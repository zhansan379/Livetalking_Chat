import time
import asyncio
import numpy as np
import resampy
import soundfile as sf
import edge_tts
from io import BytesIO

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural"
        voicename = textevent.get('tts', {}).get('ref_file',voice) #self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        # 语速：请求级 tts.rate 优先，其次回退全局 opt.SPEECH_RATE，最后引擎默认
        rate = textevent.get('tts', {}).get('rate', self.opt.SPEECH_RATE or "")
        # 兜底加固参数，均可通过 config.yaml 可选；缺省值维持原行为
        max_try      = int(getattr(self.opt, 'TTSP_RETRY', 1))           # 额外重试次数
        timeout      = float(getattr(self.opt, 'TTSP_TIMEOUT', 30) or 0) # 单次合成超时(秒)，0=不设
        min_audio_ms = float(getattr(self.opt, 'TTSP_MIN_AUDIO_MS', 300))# 低于此时长认为被截断(毫秒)

        def _warn(why, attempt):
            logger.warning(f"edgetts {why} (attempt {attempt}/{max_try+1}): {text[:20]!r}")

        # 供基类 process_tts 统一埋点：把咽下的失败(barge-in/empty/truncated/max_retries)
        # 与重试计数登记给基类；基类按它生成 tts_call 事件。
        def _fail(fail_reason, attempts, truncated=False, audio_ms=0):
            self.tts_fail(fail_reason, attempts=attempts, truncated=truncated,
                          retried=attempts > 1, audio_ms=audio_ms)

        stream = None
        retried = False      # 本句是否发生过任意类型的重试
        truncated = False    # 本句是否发生过「断流截断」重试
        audio_ms = 0.0       # 若从未到达解码赋值（如一直空音频），else 分支兜底
        for attempt in range(1, max_try + 2):
            self._reset_stream()
            t = time.time()
            try:
                loop = asyncio.new_event_loop()
                if timeout > 0:
                    loop.run_until_complete(asyncio.wait_for(
                        self.__main(voicename, text, rate), timeout=timeout))
                else:
                    loop.run_until_complete(self.__main(voicename, text, rate))
                loop.close()
            except Exception as e:
                _warn(f"合成异常: {type(e).__name__}: {e}", attempt)
                # 中途被打断(barge-in)则不再徒劳重试，直接放弃本句
                if self.state != State.RUNNING:
                    _fail("barge_in", attempt)
                    self._reset_stream()
                    return
                retried = True
                continue
            logger.info(f'-------edge tts time:{time.time()-t:.4f}s')

            if self.input_stream.getbuffer().nbytes <= 0:  #edgetts err
                _warn("返回空音频", attempt)
                if self.state != State.RUNNING:
                    _fail("barge_in", attempt)
                    self._reset_stream()
                    return
                retried = True
                continue

            self.input_stream.seek(0)
            try:
                stream = self.__create_bytes_stream(self.input_stream)
            except Exception:
                logger.exception('edgetts 解码音频失败')
                if self.state != State.RUNNING:
                    _fail("barge_in", attempt)
                    self._reset_stream()
                    return
                self._reset_stream()
                stream = None
                retried = True
                continue
            audio_ms = stream.shape[0] / self.sample_rate * 1000
            logger.info(f'[INFO] tts audio 时长 {audio_ms:.0f}ms')
            if audio_ms >= min_audio_ms:
                break
            _warn(f"音频过短({audio_ms:.0f}ms < {min_audio_ms}ms，疑似断流被截断)", attempt)
            # 断流导致的截断不属于 barge-in，允许重试
            truncated = True
            retried = True
            self._reset_stream()
            stream = None
        else:
            logger.error(f'edgetts 合成失败超过重试上限，丢弃该句: {text[:20]!r}')
            _fail("empty_audio" if retried and audio_ms == 0 else
                  ("truncated" if truncated else "max_retries"),
                  max_try + 1, truncated=truncated)
            self._reset_stream()
            return

        # 合成成功（可能经历过重试）
        self.tts_ok(audio_ms=audio_ms, attempts=attempt, retried=retried)

        # ---- 播放音频 ----
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self._reset_stream()

    def _reset_stream(self):
        """清空本次合成的音频缓冲区，避免上次残留污染下一次合成。"""
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str, rate: str = ""):
        try:
            communicate = edge_tts.Communicate(text, voicename, rate=rate or "+0%")

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception:
            logger.exception('edgetts')  # 不再吞错：记录堆栈后向上抛，触发外层重试
            raise

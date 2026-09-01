###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index
from server.action_prep import list_usable_emotions

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)

class BaseAvatar:
    # 各 sub-avatar 会存在这些「说话基地序列」cycle 成员（不存在该模型的属性会自行跳过）。
    # 表情动作 = 把这些成员整体换到另一套底座序列（data/actions/<emotion>/），口型照常贴脸生成。
    _CYCLE_ATTRS = ("frame_list_cycle", "coord_list_cycle",
                    "mask_list_cycle", "mask_coords_list_cycle",
                    "input_latent_list_cycle", "face_list_cycle")

    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms)
        self.sessionid = self.opt.sessionid

        # ── 会话级对话代际（并发安全）─────────────────────────────────
        # 每次新用户回合 +1（begin_chat）；旧回合在 TTS 喂料/落盘前用
        # is_stale 自检：一旦被新回合取代，旧回合直接放弃输出，避免
        # 「并发回合 → 历史覆盖丢失 + 前后两次回答叠读」。
        self._chat_gen = 0
        self._chat_task = None

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.msgqueues = []
        # self.custom_opt = {}
        self.__loadcustom()

        # ── 表情动作（说话时按情绪切基地帧：换一套基地序列，而非抢插）────
        # 启用与否/候选列表不再读 config.yaml，由 data/actions/<em>/action_info.json
        # 派生：存在可用底座即启用；想关掉单个表情就把其 manifest 的 enabled 设 false。
        self._neutral_cycles = {}      # 默认基地序列快照（sub-avatar init 尾部调用 _snapshot_neutral_cycles）
        self._emotion_cache = {}       # emotion → cycle dict（懒加载；加载失败记为 None 容错）
        self._emotion_bind_cache = {}  # emotion → 绑定的头像 id（None=无 manifest，视为全局/遗留）
        self._cur_emotion = None       # 当前生效表情；None = 中性默认形象
        self._last_speech_ts = 0.0     # 最近说话帧时刻（持续静音超时才回中性，避免句间小间隔抖动）
        # 一次性表情开关（最终决定见能能力目录/server 参数下发；默认关闭保持【来回镜像循环】）：
        #   emotion_once=True → 动作单向播一遍后自动回中性（不等静音）。
        #   触发粒度=「一次回答」：同一回复(trace_id)内只播一次，即使该回复被切成多句；
        #   下次新回答(trace_id 变化)才重新触发。默认关闭=原镜像循环。
        self.emotion_once_enabled = bool(getattr(opt, "emotion_once", False))
        self._emo_played_reply = None  # 已播过一次性表情的那次回复标识(trace_id)；跨回复自然失效
        # 底座溶解过渡：进出表情都做 ~0.3s addWeighted 渐变，避免一帧硬切（不同形象间
        # 呈现为形变溶解；同形象则是最自然的情绪渐变）。由 process_frames 检测 _cur_emotion
        # 变化后驱动。
        self._emo_transition_at = 0.0   # 最近一次底座切换时刻（render 侧检测到后刷新）
        self._emo_alpha_from = None     # 切换前最后一帧输出（过渡起点）

        self.batch_size = opt.batch_size
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()

        # 路由开启且多候选时返回 TTSPool（句内回退+熔断）；否则单后端（旧行为）。
        from tts import select_tts

        self.tts = select_tts(opt, parent=self)
        if self.tts is None:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict={}):
        if hasattr(self, 'tts'):
            # 新回复会取代旧的「被打断待补播」载荷：真打断后若已生成新回复，
            # 就不该再补播过期句子（避免把真实抢话后的旧话补回去）。
            try:
                if getattr(self.tts, "_interrupted", None) is not None:
                    self.tts._interrupted = None
            except Exception:  # noqa: BLE001 - 清标记失败不影响入队
                pass
            self.tts.put_msg_txt(msg, datainfo)

    def resume_talk(self) -> bool:
        """补播最近一次被打断未播完的内容（假打断 / 空转写后用）。有补播返回 True。"""
        if hasattr(self, 'tts') and hasattr(self.tts, 'resume_interrupted'):
            try:
                return bool(self.tts.resume_interrupted())
            except Exception as e:  # noqa: BLE001 - 补播失败不崩
                logger.warning("avatar resume_talk failed: %s", e)
        return False
    
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={}): # 16khz 20ms pcm
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo)

    def put_audio_file(self, filebyte, datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

    def put_audio_filepath(self, filepath, datainfo:dict={}): 
        stream = self.__create_bytes_stream(filepath)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        self.custom_audiotype = 0

    # ── 会话级对话代际 ─────────────────────────────────────────────
    def begin_chat(self) -> int:
        """开启一个新回合，返回其代际 gen；同时记录上一个回合的任务以便取消。"""
        self._chat_gen += 1
        return self._chat_gen

    def current_gen(self) -> int:
        return self._chat_gen

    def is_stale(self, gen: int) -> bool:
        """该 gen 是否已被更新的回合取代；被取代的旧回合应停止喂料/落盘。"""
        return gen < self._chat_gen

    def attach_task(self, gen: int, task):
        """记录最近一个回合的 asyncio task；取消上一个回合任务。"""
        prev = self._chat_task
        if prev and gen > 0 and not prev.done():
            prev.cancel()
        self._chat_task = task  

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def add_msgqueue(self, msgqueue):
        self.msgqueues.append(msgqueue)

    def send_msg(self, msg):
        for q in self.msgqueues:
            q.put(msg)

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            # 剔除内部观测字段（_obs 蹭在 datainfo/textevent 里传往 TTS 线程），
            # 不把它写到发往浏览器的 SSE 事件。
            ep = {k: v for k, v in eventpoint.items() if not str(k).startswith("_")}
            logger.info("notify:%s", ep)
            # 端标记送达观测：status=="end" 的零帧被 WebRTC recv 出队的此刻，obs 也补
            # 一条 tts_playback，度量「合成了 vs 真送到浏览器前的边界」。_obs 在本
            # eventpoint（_send_end 蹭来的 textevent）里，和 _run_tts_observed 同款跨
            # 线程显式 ID 打回对应 trace，从而能和 tts_call 关联出端送达率。
            if eventpoint.get('status') == 'end' and eventpoint.get('_obs'):
                try:
                    from obs import emit_explicit
                    o = eventpoint['_obs']
                    emit_explicit({
                        "type": "tts_playback",
                        "text": (ep.get('text') or "")[:40],
                        "text_len": len(ep.get('text') or ""),
                        "status": "end",
                    }, trace_id=o.get("trace_id"), session_id=o.get("session_id"),
                       parent_id=o.get("parent_id"), kind="chat")
                except Exception:  # noqa: BLE001 - 观测缺失不影响播放
                    pass
            self.send_msg(json.dumps(ep))

    def start_recording(self):
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tobytes()) #tostring()

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tobytes())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        
        record_path = os.path.join('data', 'record')
        os.makedirs(record_path, exist_ok=True)
        output_file = os.path.join(record_path, f"{self.opt.sessionid}.mp4")
        
        temp_aac = f"temp{self.opt.sessionid}.aac"
        temp_mp4 = f"temp{self.opt.sessionid}.mp4"
        
        cmd_combine_audio = f"ffmpeg -y -i {temp_aac} -i {temp_mp4} -c:v copy -c:a copy {output_file}"
        os.system(cmd_combine_audio)
        
        # 删除临时文件
        try:
            os.remove(temp_aac)
            os.remove(temp_mp4)
        except Exception as e:
            logger.error(f"Error removing temp files: {e}")

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ==================== 表情动作：基地序列切换器 ====================
    # 原理（docs/动作视频作为说话基地帧.md）：数字人只重生成脸部一块、再贴回基地全身帧。
    # 「表情动作」= 把当前说话的基地序列整体换到另一套表情底座视频的 cycle，
    # 口型照常由 inference_batch/paste_back_frame 生成并贴到该表情帧上 → 边说话边带表情。
    # set_emotion 由推理线程在算预测前触发（见 inference 的音频批循环），并清空渲染队列、
    # 归零 index，保证「算预测的底座 = 贴回的底座」，切换瞬间不错位。多个 cycle 成员逐条
    # 交换、不追求组级原子性：极端帧读到新旧混合，下一帧即自愈（长度已pad齐并由
    # mirror_index 归到各自 cycle 长度内）。

    def cycle_attrs(self):
        """返回本模型实际存在的 cycle 属性名。"""
        return [a for a in self._CYCLE_ATTRS if hasattr(self, a)]

    def _snapshot_neutral_cycles(self):
        """快照当前（默认形象）基地序列，供 set_emotion(None) 回中性时恢复。"""
        self._neutral_cycles = {}
        for a in self.cycle_attrs():
            self._neutral_cycles[a] = getattr(self, a)
        logger.info("emotion: snapshot neutral cycles -> %s", list(self._neutral_cycles))

    def _load_emotion_cycle(self, emotion):
        """从 data/actions/<emotion>/ 懒加载一套表情底座 cycle dict；失败返回 None（不回中性）。"""
        loader = getattr(self, "_load_cycle", None)  # sub-avatar 提供
        if not loader:
            return None
        try:
            payload = loader(emotion, root="data/actions")
            if not payload:
                return None
            d = {a: payload[a] for a in self.cycle_attrs() if payload.get(a) is not None}
            return d or None
        except Exception:  # noqa: BLE001 - 表情底座缺失/损坏不影响主说话链路
            logger.exception("emotion: load cycle %s failed", emotion)
            return None

    def _emotion_binding(self, emotion):
        """返回该表情底座绑定的头像 id；无 manifest/未绑定 = None（视为全局/遗留，放行）。

        读 data/actions/<emotion>/action_info.json（generate_action 生成的绑定记录），
        首次读取后缓存，避免推理线程每切一次就读文件。
        """
        if emotion not in self._emotion_bind_cache:
            bind = None
            import json, os
            p = os.path.join("data/actions", emotion, "action_info.json")
            try:
                with open(p, "r", encoding="utf-8") as f:
                    bind = (json.load(f).get("bind_avatar") or "").strip() or None
            except Exception:  # noqa: BLE001 - manifest 缺失/损坏按全局放行（容忍遗留 sad）
                bind = None
            self._emotion_bind_cache[emotion] = bind
        return self._emotion_bind_cache.get(emotion)

    def set_emotion(self, emotion):
        """把当前说话的基地序列切成指定表情；None/neutral 恢复默认形象。

        候选与启用由 data/actions 派生（list_usable_emotions，按当前形象筛 enabled+绑定）；
        无可用表情底座 = 功能关闭，静默返回。
        """
        _aid = getattr(self.opt, "avatar_id", "")
        names = list_usable_emotions(_aid)
        if not names:                       # 系统关闭（当前形象无可用表情底座）
            return
        if emotion in (None, "", "neutral"):
            if self._cur_emotion is not None:
                self._cur_emotion = None
                for a, v in self._neutral_cycles.items():
                    setattr(self, a, v)
                logger.info("emotion: reset -> neutral")
            return
        if emotion not in names:
            return  # 不在候选表情集，保持现状
        if emotion == self._cur_emotion:
            return
        # 运行时强制绑定：该底座绑定了头像，且不是当前说话形象 → 拒绝切换（保持现状）。
        bind = self._emotion_binding(emotion)
        if bind is not None and bind != getattr(self.opt, "avatar_id", None):
            logger.warning(
                "emotion: %s 绑定到头像 %s，当前形象 %s，拒绝切换",
                emotion, bind, getattr(self.opt, "avatar_id", None),
            )
            return
        if emotion not in self._emotion_cache:
            self._emotion_cache[emotion] = self._load_emotion_cycle(emotion)
        payload = self._emotion_cache.get(emotion)
        if not payload:
            logger.warning("emotion: %s 无可用底座序列，保持当前", emotion)
            return
        self._cur_emotion = emotion
        for a, v in payload.items():
            setattr(self, a, v)
        n = len(payload.get("frame_list_cycle") or [])
        logger.info("emotion: switch -> %s (base frames=%d)", emotion, n)

    def _drain_res_queue(self):
        """清空渲染队列：切换底座时丢弃用旧底座算出的在途预测，避免残留错位帧。"""
        if not getattr(self, "res_frame_queue", None):
            return
        try:
            while True:
                self.res_frame_queue.get_nowait()
        except queue.Empty:
            pass

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1
        
    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            for _ in range(self.batch_size * 2):
                audioframe:AudioFrameData = self.asr.output_queue.get()
                if audioframe.type == 0:
                    is_all_silence = False               
                audio_frames.append(audioframe)

             # 检测状态变化
            current_speaking = not is_all_silence

             # ── 表情切换：在推理前按本批音频携带的 emotion 定底座 ──────────────
            # 必须切在这里（推理线程、算预测之前），保证「用哪套底座算出的预测，就用哪套
            # 底座贴回」。若切在 process_frames（渲染线程），推理已用旧底座算出一批预测、
            # 贴回时却用新底座 → 人脸位置错位。整轮每句携带同一 emotion，set_emotion 按
            # _cur_emotion 去重，不会反复切换。
            _spk = not is_all_silence
            _rep = None  # 本次回复标识(trace_id)：同一次回答的所有句共享，跨回答自动变化
            if _spk:
                emo = None
                for _af in audio_frames:
                    if _af.type == 0 and _af.userdata:
                        if emo is None:
                            emo = _af.userdata.get("emotion") or None
                        if _rep is None:
                            _rep = (_af.userdata.get("_obs") or {}).get("trace_id") or None
                        if emo and _rep:
                            break
                # 一次性表情去重：按「一次回答(trace_id)」判定，而非静音；同一回复内只播一次。
                _already = bool(emo and self.emotion_once_enabled and _rep is not None
                                and self._emo_played_reply == _rep)
                if emo and emo != getattr(self, "_cur_emotion", None) and not _already:
                    self._drain_res_queue()   # 清掉用旧底座产出的在途预测，避免残留错位帧
                    index = 0                 # 新底座从头编排帧
                    self.set_emotion(emo)
                    # 触发当下就锁住该次回复：之后同一回复的任意句子都视为已播过，
                    # 即便动作未播满一遍、提前经 1.2s 静音回中性，也不会再重复触发。
                    if _rep is not None:
                        self._emo_played_reply = _rep
                self._last_speech_ts = time.time()
            elif getattr(self, "_cur_emotion", None) is not None \
                    and time.time() - getattr(self, "_last_speech_ts", 0.0) > 1.2 \
                    and not (self.emotion_once_enabled
                             and index < self.get_avatar_length()):
                # 一次性表情：在【播满一整遍(index≥帧数)】之前，静音空档不提前回中性，
                # 让动作借句子间隙继续往前走满一遍；播满后由下方 completion 块收尾回中性。
                self.set_emotion(None)        # 持续静音 → 回中性

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                    index = index + 1
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                pred = self.inference_batch(index, audiofeat_batch)

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], mirror_index(length, index)))
                    index = index + 1

            # 一次性表情(emotion_once)：动作单向播完一遍 → 自动回中性，不等 1.2s 静音。
            # index≥当前底座帧数即「播完一遍」，随后回报该次回复(trace_id)已触发；
            # 本回答剩余多句不再重触发（_emo_played_reply 去重），下次新回答才重新触发。
            # 开关关闭时此块为空操作，保持原【镜像循环】语义。
            if self.emotion_once_enabled:
                cur_emo = getattr(self, "_cur_emotion", None)
                if cur_emo is not None and index >= self.get_avatar_length():
                    if _rep is not None:
                        self._emo_played_reply = _rep
                    self.set_emotion(None)

            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    def process_frames(self,quit_event):
        enable_transition = False  # 设置为False禁用过渡效果，True启用

        _last_speaking = False
        _transition_start = time.time()
        # 底座溶解过渡（进出表情）：检测 _cur_emotion 变化，以切换前最后一帧为起点渐变
        _EMO_TR_DUR = 0.3          # 溶解时长(秒)
        _prev_emotion = getattr(self, "_cur_emotion", None)
        _last_out = None
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    # 防御：表情切换过程中，队列里由推理线程按中性帧数编好的 idx，
                    # 可能大于表情底座的帧数；取模钳回，宁可自愈取错一帧也不让线程崩。
                    target_frame = self.frame_list_cycle[idx % max(1, len(self.frame_list_cycle))]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            # ── 底座溶解过渡：进出表情/回中性时，一帧硬切改为 ~0.3s 渐变 ──────
            # 底座由推理线程 set_emotion 切换（_cur_emotion 变化）；这里以切换前最后
            # 一帧输出 _last_out 为起点，对新底座帧 addWeighted 渐近。切换后 _drain 清
            # 过队列，_last_out 几乎总是指向旧底座的帧，方向正确。
            cur_emotion = getattr(self, "_cur_emotion", None)
            if cur_emotion != _prev_emotion:
                if _last_out is not None:
                    self._emo_alpha_from = _last_out
                    self._emo_transition_at = time.time()
                else:
                    self._emo_alpha_from = None
                _prev_emotion = cur_emotion
            if getattr(self, "_emo_alpha_from", None) is not None:
                a = (time.time() - self._emo_transition_at) / _EMO_TR_DUR
                if a < 1.0:
                    from_f = self._emo_alpha_from
                    # 底座尺寸不齐（表情底座帧尺寸可能 ≠ 当前底座）时，先把旧帧
                    # 等比缩放(INTER_AREA 收缩/升采)到当前帧尺寸再淡出，避免
                    # addWeighted 因两张图大小不同而崩线程。仅切换的那 0.3s 生效。
                    if from_f.ndim == combine_frame.ndim and from_f.shape != combine_frame.shape:
                        from_f = cv2.resize(
                            from_f,
                            (combine_frame.shape[1], combine_frame.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    if from_f.shape == combine_frame.shape:
                        combine_frame = cv2.addWeighted(from_f, 1 - a, combine_frame, a, 0)
                    else:
                        # 通道数等仍不一致（理论少见）→ 放弃本帧淡出，直接硬切。
                        self._emo_alpha_from = None
                else:
                    self._emo_alpha_from = None  # 过渡完成
            _last_out = combine_frame.copy() if combine_frame is not None else None

            # 使用统一输出接口推送视频帧
            self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                #frame,type,eventpoint = audio_frame
                frame = (audio_frame.data * 32767).astype(np.int16)

                # 使用统一输出接口推送音频帧
                self.output.push_audio_frame(frame, audio_frame.userdata)
                self.record_audio_data(frame)
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        while not quit_event.is_set(): 
            t = time.perf_counter()
            self.asr.run_step()

            buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
            if buffer_size >= 5:
                logger.debug('sleep qsize=%d', buffer_size)
                time.sleep(0.04 * buffer_size * 0.8)
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()


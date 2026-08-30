###############################################################################
#  ChatAgent：加载历史 + 组装上下文 + 有界压缩记忆 + 完整转录持久化
###############################################################################

from datetime import datetime

from utils.logger import logger
from agent.config import get_agent_config
from agent.history import load_history, save_history


def _format_rounds(messages: list[dict]) -> str:
    """把若干条消息格式化为可压缩的文本。"""
    lines = []
    role_label = {"user": "用户", "assistant": "助手"}
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role_label.get(role, role)}：{content}")
    return "\n".join(lines)


class ChatAgent:
    """
    一个会话的记忆代理：构造时从 JSON 加载历史，负责把历史组装进上下文，
    并在到达轮次阈值时用 LLM 把旧轮次压缩成摘要（完整转录始终保留）。
    """

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._config = get_agent_config()

        summary, last_idx, messages = load_history(session_id)
        self._summary: str = summary
        self._last_compressed_index: int = last_idx
        self._messages: list[dict] = messages  # 完整转录，append-only

    @staticmethod
    def _today() -> str:
        """当前日期（本地时区），用于提醒模型以之为时间基准。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    # ─── 完整转录写入 ──────────────────────────────────────────────────────
    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    # ─── 上下文组装 ────────────────────────────────────────────────────────
    def build_messages(self, user_message: str) -> list[dict]:
        """
        上下文 = 系统提示 + 历史摘要 + 最近 keep_recent 轮原文 + 当前消息。
        同步操作，不触发压缩。
        """
        cfg = self._config
        system = (
            f"{cfg.system_prompt}\n今天的日期是 {self._today()}，回答涉及时间时以此为准。"
        )
        msgs: list[dict] = [{"role": "system", "content": system}]
        if self._summary:
            msgs.append({"role": "system", "content": f"<历史摘要>{self._summary}</历史摘要>"})
        recent = self._messages[-2 * cfg.keep_recent:]
        msgs.extend(recent)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    # ─── 长期记忆提取上下文 ───────────────────────────────────────────────
    def recent_raw_window(self, rounds: int = 4) -> str:
        """
        最近 rounds 轮原文（user/assistant 交织，append-only _messages 末段），
        供长期记忆提取器识别"自介/身份"类陈述的上下文；不含 System/摘要。
        """
        msgs = self._messages[-2 * rounds:]
        lines = []
        for m in msgs:
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{m.get('role', 'user')}: {content}")
        return "\n".join(lines)

    # ─── 压缩判断 ──────────────────────────────────────────────────────────
    def should_compress(self) -> bool:
        """自上次压缩后再积累的新轮次是否达到阈值。"""
        total_rounds = len(self._messages) // 2
        folded_rounds = self._last_compressed_index // 2
        new_rounds = total_rounds - folded_rounds
        return new_rounds >= self._config.compress_threshold

    # ─── 压缩执行（异步、后台） ────────────────────────────────────────────
    async def compress(self) -> None:
        """
        有界压缩：只处理「压缩水位 ~ 末尾 keep_recent 轮」之间的旧轮次。
        - 全程不删除 messages 里的任何原文（完整转录保留）；
        - summary 按 target_summary_chars 做有界合并/替换（见 config 说明）；
        - 更新压缩水位 last_compressed_index。
        """
        if not self._config.memory_enabled:
            return

        messages = self._messages
        fold_end = len(messages) - 2 * self._config.keep_recent
        fold_start = self._last_compressed_index
        if fold_end <= fold_start:
            return  # 没有需要折入的旧轮次

        to_compress = messages[fold_start:fold_end]
        new_text = _format_rounds(to_compress)
        if new_text == "":
            self._last_compressed_index = fold_end
            return

        target = self._config.target_summary_chars
        slack = int(target * 0.3)
        combined = (self._summary + "\n" + new_text).strip()

        if self._summary == "" or (len(self._summary) + len(new_text)) <= target + slack:
            # 摘要为空或总量仍可容纳 → 只压缩新轮次，追加到摘要
            new_summary = await self._call_summarize(new_text)
            if new_summary:
                self._summary = (self._summary + "\n" + new_summary).strip()
        else:
            # 超限 → 重写合并成一份约 target 字的摘要并替换
            self._summary = await self._call_summarize(combined)

        # 兜底护栏：异常超长时截断
        if len(self._summary) > int(target * 1.5):
            self._summary = self._summary[-target:]

        self._last_compressed_index = max(0, fold_end)
        logger.info(
            "agent[%s] compressed %d msgs -> summary chars=%d",
            self._session_id, len(to_compress), len(self._summary),
        )

    async def _call_summarize(self, text: str) -> str:
        """调用 LLM 把 text 压缩成摘要（硬上限 max_tokens + 提示词软约束）。

        压缩作为独立的 kind="summary" trace 观测，不污染用户请求的成功率/响应耗时。
        """
        from infra_ai import async_call_llm
        from obs import new_trace

        cfg = self._config
        prompt = (
            f"{cfg.summarize_prompt}（控制在大约 {cfg.target_summary_chars} 字内）"
        )
        call_kwargs: dict = {
            "use_json": False,
            "extra": {"kind": "compress"},
            "model_kwargs": {"max_tokens": cfg.max_summary_tokens},
            "capability": "compress",
        }
        if cfg.summarize_model:
            call_kwargs["model_name"] = cfg.summarize_model  # 显式单模型覆盖，优先于能力路由

        with new_trace(self._session_id, kind="summary"):
            result = await async_call_llm(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                **call_kwargs,
            )
        return (result or "").strip()

    async def compress_and_save(self) -> None:
        """后台任务入口：压缩并落盘（异常不外抛，避免后台任务静默崩溃）。"""
        try:
            await self.compress()
            self.save()
        except Exception as e:  # noqa: BLE001
            logger.exception("agent[%s] compress failed: %s", self._session_id, e)

    # ─── 持久化 ────────────────────────────────────────────────────────────
    def save(self) -> None:
        save_history(
            self._session_id,
            self._summary,
            self._last_compressed_index,
            self._messages,
        )
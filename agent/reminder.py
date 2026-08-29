###############################################################################
#  agent 域全局定时提醒 — 一次性延时 + 每日 cron，跨 session、跨重启持久化
#  设计对照：learn-claude-code/s12_cron_scheduler（persist + cron + 调度循环），但本项目
#  跑在 aiohttp 单 asyncio 事件循环上，故调度循环用 asyncio 实现；触发语义为「全局」：
#  到点时对当前所有存活会话开口，而非绑定预定时刻的某个 session。
#  对外只暴露 schedule_delay / schedule_cron / cancel / list / run，不反向依赖
#  server/session_manager 与 agent.chat 的 module 级 import（在 speak 内延迟 import，避免循环依赖）。
###############################################################################

import asyncio
import datetime
import heapq
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass

from utils.logger import logger


# ── 持久化位置 ─────────────────────────────────────────────────────────────
# 与 agent/history.py（data/chat_history）、agent/longterm.py（data/memory）一致的相对路径约定，
# 均相对服务启动时的项目根目录。
_DEFAULT_STORE_PATH = os.path.join("data", "reminders.json")


def humanize_delay(seconds: int) -> str:
    """把秒格式化成人类阅读的中文时长（用于确认话术）。"""
    seconds = int(seconds)
    if seconds >= 3600:
        h, rem = divmod(seconds, 3600)
        m = rem // 60
        return f"{h}小时{m}分钟" if m else f"{h}小时"
    if seconds >= 60:
        return f"{seconds // 60}分钟"
    return f"{seconds}秒"


# ── 到点触发的实时 agent 任务 ─────────────────────────────────────────────
# 到点时每条提醒不再说登记时固定的 content，而是当作一条实时任务去执行：
# 构造一次带工具的 LLM 回合（run_tool_loop），让模型结合可用工具生成"此刻"内容再播报。
_FIRE_FAIL_FALLBACK = "抱歉，这条定时播报这次没能准备好内容，稍后再试吧。"


def _scheduled_system_prompt(cfg) -> str:
    """定时播报任务的系统提示：沿用全局 system_prompt，追加时间基准与工具使用指引。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"{getattr(cfg, 'system_prompt', '')}\n现在是 {now}。"
        "你正在执行一条用户预定的定时播报任务：请根据任务指令调用可用工具"
        "（如天气、联网搜索）获取实时信息，然后简短、口语化地播报结果。"
        "只输出要说的内容本身，不要加任何前缀或说明。"
        "注意：本次是已到点的播报任务，无论指令怎么措辞（哪怕又是『X分钟后提醒我 Y』），"
        "都**不要**再新建/修改/取消任何提醒，只要按指令去查询并播报即可。"
    )


# ── cron 匹配 / 校验（移植自 s12_cron_scheduler/code.py）──────────────────
# 5 字段：分 时 日 月 星期（星期 Sunday=0）。支持 *、*/N、N,M、N-M、N 语法。


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    if "," in field:
        return any(_cron_field_matches(p.strip(), value) for p in field.split(","))
    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)
    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime.datetime) -> bool:
    fields = (cron_expr or "").strip().split()
    if len(fields) != 5:
        return False
    minute, hour, day, month, weekday = fields
    # Python weekday() Monday=0，但 cron 的星期 Sunday=0，换算一下：
    cron_weekday = (moment.weekday() + 1) % 7
    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False
    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches


def _validate_cron_field(field: str, lo: int, hi: int, name: str) -> str | None:
    """校验单字段语法与取值范围；合法返回 None，否则返回中文报错。"""
    for part in field.split(","):
        part = part.strip()
        if part == "*":
            continue
        if part.startswith("*/"):
            try:
                step = int(part[2:])
            except ValueError:
                return f"{name}字段 '{part}' 非法"
            if step <= 0 or step > (hi - lo + 1):
                return f"{name}字段步长 '{part}' 非法"
            continue
        if part.count("-") == 1:
            a, b = part.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
            except ValueError:
                return f"{name}字段 '{part}' 非法"
            if not (lo <= a_i <= b_i <= hi):
                return f"{name}字段 '{part}' 超出范围"
            continue
        try:
            v = int(part)
        except ValueError:
            return f"{name}字段 '{part}' 非法"
        if not (lo <= v <= hi):
            return f"{name}字段 '{part}' 超出范围"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """校验 5 字段 cron 表达式；合法返回 None，否则返回中文报错。"""
    fields = (cron_expr or "").strip().split()
    if len(fields) != 5:
        return "需 5 个字段（分 时 日 月 星期）"
    return (
        _validate_cron_field(fields[0], 0, 59, "分钟")
        or _validate_cron_field(fields[1], 0, 23, "小时")
        or _validate_cron_field(fields[2], 1, 31, "日")
        or _validate_cron_field(fields[3], 1, 12, "月")
        or _validate_cron_field(fields[4], 0, 6, "星期")
    )


def compute_next_fire(cron_expr: str, after: datetime.datetime) -> datetime.datetime | None:
    """自 after 起逐分钟前扫（上限 366 天），返回下一个 cron 匹配时刻；找不到返回 None。"""
    t = after.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    deadline = after + datetime.timedelta(days=366)
    while t <= deadline:
        if cron_matches(cron_expr, t):
            return t
        t += datetime.timedelta(minutes=1)
    return None


@dataclass
class Reminder:
    """一条持久化的定时提醒。id 全局唯一；fire_at 恒为「下一次触发」的 epoch 秒。

    content = 登记时给用户看的确认/摘要话术；
    task    = 完整、自包含的任务要求（登记时刻由模型展开，含可执行细节、去时机词），
              到点执行 _produce 只基于 task，不依赖对话历史。可为空（老记录则回退用 content）。
    """

    id: str
    content: str
    task: str = ""
    created_at: float = 0.0
    last_fired: float = 0.0
    recurring: bool = False      # False=一次性；True=按 cron 重复
    cron: str = ""               # recurring 时的 5 位 cron 表达式
    fire_at: float = 0.0         # 下一次触发（epoch 秒）；cron 型每次触发后重算 next

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "task": self.task,
            "created_at": self.created_at,
            "last_fired": self.last_fired,
            "recurring": self.recurring,
            "cron": self.cron,
            "fire_at": self.fire_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reminder":
        return cls(
            id=str(d.get("id", "")),
            content=str(d.get("content", "") or ""),
            task=str(d.get("task", "") or ""),
            created_at=float(d.get("created_at", 0) or 0),
            last_fired=float(d.get("last_fired", 0) or 0),
            recurring=bool(d.get("recurring", False)),
            cron=str(d.get("cron", "") or ""),
            fire_at=float(d.get("fire_at", 0) or 0),
        )


class ReminderManager:
    """全局定时提醒调度器（进程内单例）。

    - 状态：_reminders: dict[id, Reminder] + _heap: 小顶堆 (fire_at, id)。
    - 持久化：每次 schedule/cancel/fire 后原子落盘（临时文件 + os.replace）。
    - 触发：run() 常驻协程按堆顶 fire_at 休眠到点，_fire_due 对**所有在线会话**开口。
    """

    def __init__(self, store_path: str | None = None):
        self.store_path: str = store_path or _DEFAULT_STORE_PATH
        self._reminders: dict[str, Reminder] = {}
        self._heap: list[tuple[float, str]] = []
        self._lock = threading.RLock()

    # ── 查询/状态 ──────────────────────────────────────────────────────────
    def count(self) -> int:
        with self._lock:
            return len(self._reminders)

    def list_text(self) -> "list[str]":
        """返回人类可读的提醒清单（排查用）。方法名避免与内置 list 冲突。"""
        with self._lock:
            rows = []
            for r in sorted(self._reminders.values(), key=lambda x: x.fire_at):
                kind = f"重复 {r.cron}" if r.recurring else "一次性"
                fire = datetime.datetime.fromtimestamp(r.fire_at).strftime("%m-%d %H:%M")
                rows.append(f"{r.id} | {kind} | {fire} | {r.content}")
            return rows

    def records(self) -> list[dict]:
        """返回结构化提醒列表（HTTP 管理接口用），按下次触发时间排序。"""
        with self._lock:
            out = []
            for r in sorted(self._reminders.values(), key=lambda x: x.fire_at):
                out.append({
                    "id": r.id,
                    "content": r.content,
                    "task": r.task,
                    "recurring": r.recurring,
                    "cron": r.cron,
                    "fire_at": r.fire_at,
                    "next_fire": datetime.datetime.fromtimestamp(r.fire_at)
                        .strftime("%Y-%m-%d %H:%M"),
                    "created_at": r.created_at,
                    "last_fired": r.last_fired,
                })
            return out

    # ── 登记 / 取消 ────────────────────────────────────────────────────────
    def schedule_delay(self, delay_seconds, content: str, task: str = "") -> str:
        """登记一条一次性延时提醒（delay_seconds 秒后触发），返回 reminder id。"""
        now = time.time()
        with self._lock:
            rid = self._new_id()
            rem = Reminder(id=rid, content=content, task=task, created_at=now,
                           fire_at=now + float(delay_seconds))
            self._reminders[rid] = rem
            heapq.heappush(self._heap, (rem.fire_at, rid))
            self._save()
        logger.info("[reminder] scheduled one-shot %s in %ss: %s", rid, delay_seconds, content)
        return rid

    def schedule_cron(self, cron: str, content: str, task: str = "") -> str:
        """登记一条重复（cron）提醒，返回 reminder id；cron 非法抛 ValueError。"""
        err = validate_cron(cron)
        if err:
            raise ValueError(err)
        nxt = compute_next_fire(cron, datetime.datetime.now())
        if nxt is None:
            raise ValueError("无法计算该定时表达式的下一次触发时间")
        now = time.time()
        with self._lock:
            rid = self._new_id()
            rem = Reminder(id=rid, content=content, task=task, created_at=now,
                           fire_at=nxt.timestamp(), recurring=True, cron=cron)
            self._reminders[rid] = rem
            heapq.heappush(self._heap, (rem.fire_at, rid))
            self._save()
        logger.info("[reminder] scheduled cron %s `%s`: %s", rid, cron, content)
        return rid

    def cancel(self, rid: str) -> bool:
        """取消一条提醒；存在则移除并持久化，返回是否真的存在。"""
        with self._lock:
            if rid not in self._reminders:
                return False
            del self._reminders[rid]
            self._rebuild_heap()
            self._save()
        logger.info("[reminder] cancelled %s", rid)
        return True

    def cancel_all(self) -> int:
        with self._lock:
            n = len(self._reminders)
            self._reminders.clear()
            self._heap = []
            self._save()
        logger.info("[reminder] cancelled all (%d)", n)
        return n

    # ── 触发 ──────────────────────────────────────────────────────────────
    def speak(self, content: str) -> None:
        """全局触发：对当前所有存活会话开口（无会话则本次无听众）。"""
        try:
            # 延迟 import：避免 server.session_manager / agent.chat 形成 module 级循环依赖。
            from server.session_manager import session_manager
            from agent.chat import _feed_talk, notify_reply_start
        except Exception as e:  # noqa: BLE001 - 模块未就绪时不崩
            logger.warning("[reminder] speak deps unavailable: %s", e)
            return
        for avatar in list(session_manager.sessions.values()):
            if avatar is None:
                continue
            try:
                notify_reply_start(avatar)   # 前端清空字幕后逐句追加
                _feed_talk(avatar, content, {})
            except Exception:  # noqa: BLE001 - 单个会话失败不影响其它/不崩
                logger.exception("[reminder] speak failed for a session")

    async def _produce(self, rem: Reminder) -> str | None:
        """到点生成实际播报文本：把任务要求当指令跑一次带工具的 LLM 回合。

        instruction 优先取登记时刻由模型展开的完整任务要求 rem.task（自包含、已去时机词），
        老记录无 task 时回退用 rem.content。→ 不依赖对话历史、不二次理解为"再设提醒"。
        失败/无输出返回 None（交给调用方走失败兜底话术），绝不伪造内容。
        """
        # 延迟 import：agent.tool_loop 在模块级 import 了本模块，这里内联避免循环依赖。
        from agent.tool_loop import build_tools, list_enabled_tools, run_tool_loop
        from agent.chat import _sanitize
        from agent.config import get_agent_config

        instruction = (rem.task or "").strip() or rem.content   # 完整任务要求（已展开，非口语原话）
        cfg = get_agent_config()
        messages = [
            {"role": "system", "content": _scheduled_system_prompt(cfg)},
            {"role": "user", "content": instruction},
        ]
        # 到点回合不给「提醒管理」工具，只保留能取实时信息的只读工具，
        # 防止指令里『X分钟后提醒我Y』被当成再次新建提醒而自我繁殖。
        blocked = {"schedule_reminder", "cancel_reminder", "list_reminders"}
        tools = build_tools(n for n in list_enabled_tools(cfg) if n not in blocked)
        if tools:
            final = await run_tool_loop(messages, tools, cfg)
        else:
            from infra_ai import async_call_llm
            final = await async_call_llm(messages)
        return _sanitize(final) if final else None

    async def _handle_fire(self, rem: Reminder, now: float) -> None:
        """触发一条提醒：生成 → 全局播报 → 一次性移除/cron 重排 → 落盘。"""
        try:
            text = await self._produce(rem)
        except Exception:  # noqa: BLE001 - 生成失败走兜底，不中断调度
            logger.exception("[reminder] task produce failed for %s", rem.id)
            text = None
        try:
            self.speak(text if text else _FIRE_FAIL_FALLBACK)
        except Exception:  # noqa: BLE001 - 单会话说话失败不影响其它
            logger.exception("[reminder] speak failed for %s", rem.id)
        rem.last_fired = now
        now_dt = datetime.datetime.fromtimestamp(now)
        with self._lock:
            if rem.recurring:
                nxt = compute_next_fire(rem.cron, now_dt)
                if nxt:
                    rem.fire_at = nxt.timestamp()
                    heapq.heappush(self._heap, (rem.fire_at, rem.id))
                else:
                    self._reminders.pop(rem.id, None)
            else:
                self._reminders.pop(rem.id, None)
            self._save()
        logger.info("[reminder] handled %s at %s (%s)",
                    rem.id, now_dt.strftime("%Y-%m-%d %H:%M:%S"), rem.content[:40])

    async def _fire_due(self, now: float) -> None:
        """弹出所有到点项，每条各开一个并发 task 触发（LLM 调用不阻塞调度主循环）。"""
        with self._lock:
            due: list[Reminder] = []
            while self._heap and self._heap[0][0] <= now:
                _fire_at, rid = heapq.heappop(self._heap)
                rem = self._reminders.get(rid)
                if rem is not None:
                    due.append(rem)
        for rem in due:
            asyncio.create_task(self._handle_fire(rem, now))

    # ── 调度主循环 ────────────────────────────────────────────────────────
    async def run(self) -> None:
        """常驻调度协程：按堆顶 fire_at 休眠到点触发（由 app.py 用 create_task 拉起）。"""
        logger.info("[reminder] scheduler started (store=%s)", self.store_path)
        while True:
            with self._lock:
                if not self._heap:
                    await asyncio.sleep(1.0)
                    continue
                fire_at, _rid = self._heap[0]
            wait = fire_at - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, 30.0))
                continue
            await self._fire_due(time.time())

    # ── 持久化 ────────────────────────────────────────────────────────────
    def load(self) -> int:
        """启动时从磁盘恢复提醒：过滤非法 cron、丢弃过期一次性项、重算 cron 型 next。"""
        with self._lock:
            self._reminders = {}
            self._heap = []
            if not os.path.exists(self.store_path):
                return 0
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:  # noqa: BLE001 - 读坏文件不崩
                logger.warning("[reminder] load failed: %s", e)
                return 0
            now = time.time()
            now_dt = datetime.datetime.fromtimestamp(now)
            for item in data or []:
                rem = Reminder.from_dict(item)
                if not rem.id or not rem.content:
                    continue
                if rem.recurring:
                    if validate_cron(rem.cron):
                        continue
                    if rem.fire_at <= now:
                        nxt = compute_next_fire(rem.cron, now_dt)
                        if nxt is None:
                            continue
                        rem.fire_at = nxt.timestamp()
                else:
                    if rem.fire_at <= now:
                        continue  # 已过期的一次性提醒，不再恢复
                self._reminders[rem.id] = rem
            self._rebuild_heap()
            self._save()
            logger.info("[reminder] loaded %d reminder(s)", len(self._reminders))
            return len(self._reminders)

    def _save(self) -> None:
        # 原子写：临时文件（带 pid/线程 id 防多线程写冲突）+ os.replace。
        tmp = f"{self.store_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            payload = [r.to_dict() for r in self._reminders.values()]
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.store_path)
        except Exception as e:  # noqa: BLE001 - 落盘失败只告警，调度照常
            logger.warning("[reminder] save failed: %s", e)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ── 内部工具 ──────────────────────────────────────────────────────────
    def _rebuild_heap(self) -> None:
        self._heap = [(r.fire_at, r.id) for r in self._reminders.values()]
        heapq.heapify(self._heap)

    def _new_id(self) -> str:
        while True:
            rid = f"rem_{secrets.token_hex(4)}"
            if rid not in self._reminders:
                return rid


# 进程内单例：app.py 在事件循环上拉起 rem(inder_manager).run()。
reminder_manager = ReminderManager()
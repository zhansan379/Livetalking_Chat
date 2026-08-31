###############################################################################
#  跨会话长期记忆（参考 s09_memory/）
#
#  与会话级记忆（agent/agent.py 的摘要+压缩）正交的持久层：把若干次会话里
#  学到的持久知识（用户画像 / 工作反馈 / 项目事实 / 外部线索）落盘，
#  下次请求时按需召回，注入主对话的 system prompt。
#
#  四管道：
#    存储    每记忆一个 data/memory/<slug>.md（YAML frontmatter + 正文），
#            写入复用 history.py 的原子模式；INDEX 为人读/排查用，运行时不读。
#    召回    glob 目录 → 解析 frontmatter 得短目录 → rerank(默认)/关键词打分取
#            top-k → 只读命中的正文注入（可选 recall_mode=model 走 s09 模型选择）。
#    提取    回合结束后后台调 LLM，候选经 should_store_memory 准入才落盘。
#    整理    记忆条数达阈值后台去重/合并，带 snapshot 回滚。
#
#  全部对外入口都包在防御性 try/except 里：任何失败只影响记忆质量，
#  绝不抛出、绝不阻塞语音主回复。
###############################################################################

import asyncio
import glob
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

from utils.logger import logger
from agent.config import get_agent_config

# 允许落盘的记忆类型（user / feedback / project / reference / state）
# state = 用户持续的「情绪/心理状态」快照，采用「最新覆盖」语义（见下面提取路径）
MEMORY_TYPES = ("user", "feedback", "project", "reference", "state")

# 命中即拒的临时性短语（多语言，沿用 s09 设计）：这类内容应留在当前会话，
# 不应成为跨会话的持久规则/事实。
TEMPORARY_MARKERS = (
    "this session", "current session", "this turn", "current turn",
    "this task", "current task", "for now", "just this time", "today only",
    "本次会话", "当前会话", "这一轮", "当前轮次", "本次任务", "当前任务",
    "暂时", "临时",
)

_INDEX_NAME = "_INDEX"
_extract_round_ctr = 0  # every_n_turns 触发计数

# 整理并发/冷却护栏：防重复调度、防递归、防阈值震荡空转
_consolidating = False           # 一次整理进行中（防重入/防 consolidate 内部写回触发自己）
_consolidate_pending = False     # 已有一发整理的调度（事件循环/线程）待执行
_last_consolidate_done = 0.0     # 上次整理成功结束的时刻（monotonic 秒），作冷却基准


@dataclass
class MemoryRecord:
    """一条长期记忆。"""
    name: str
    description: str
    type: str
    body: str
    slug: str = ""
    created_at: str = ""
    updated_at: str = ""


# ════════════════════════════════════════════════════════════════════════════
#  存储原语
# ════════════════════════════════════════════════════════════════════════════

def memory_dir() -> str:
    """记忆目录：显式配置 dir > AGENT_MEMORY_DIR env > 默认 data/memory。"""
    cfg = get_agent_config()
    if cfg.longterm_dir:
        return cfg.longterm_dir
    return os.environ.get("AGENT_MEMORY_DIR", os.path.join("data", "memory"))


def memory_path_for(slug: str) -> str:
    """暴露单个记忆文件路径（调试/测试用）。"""
    return os.path.join(memory_dir(), f"{slug}.md")


def _slugify(name: str) -> str:
    """name → kebab-case slug（允许中文，方便 CJK 文件名）。"""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9一-鿿]+", "-", s).strip("-")
    return s or "memory"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """把 md 文本拆成 (frontmatter dict, body)。无 frontmatter 时返回空 dict + 原文本。"""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            meta = yaml.safe_load(text[3:end]) or {}
            return meta, text[end + 4:].lstrip("\n")
    return {}, text


def _parse_memory_file(path: str) -> MemoryRecord | None:
    """解析单个 .md 记忆文件；损坏/缺 name/type 时跳过并告警。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        meta, body = _split_frontmatter(text)
        name = (meta.get("name") or "").strip()
        mtype = str(meta.get("type") or "").strip()
        if not name or not mtype:
            return None
        return MemoryRecord(
            name=name,
            description=(meta.get("description") or "").strip(),
            type=mtype,
            body=(body or "").strip(),
            slug=os.path.splitext(os.path.basename(path))[0],
            created_at=str(meta.get("created_at") or ""),
            updated_at=str(meta.get("updated_at") or ""),
        )
    except Exception as e:  # noqa: BLE001 - 单个损坏文件不应阻塞整库
        logger.warning("memory file parse failed %s (%s), skip", path, e)
        return None


def list_memories() -> list[MemoryRecord]:
    """目录（唯一事实源）× 全部 .md 的短记录列表；运行时从这里读，不读 INDEX。"""
    records = []
    for path in sorted(glob.glob(os.path.join(memory_dir(), "*.md"))):
        rec = _parse_memory_file(path)
        if rec:
            records.append(rec)
    return records


def _write_memory_atomic(record: MemoryRecord, rebuild: bool) -> None:
    """原子写入单个记忆文件（临时文件 + os.replace）。"""
    path = memory_path_for(record.slug)
    document = (
        "---\n"
        + yaml.safe_dump(
            {
                "name": record.name,
                "description": record.description,
                "type": record.type,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            },
            allow_unicode=True, default_flow_style=False, sort_keys=False,
        ).strip()
        + "\n---\n\n"
        + record.body.strip()
        + "\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".mem-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(document)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if rebuild:
        rebuild_index()


def write_memory(record: MemoryRecord, rebuild: bool = True) -> MemoryRecord:
    """写入一条记忆（slug 未给则从 name 生成），随后重建 INDEX。
    写入后统一判定是否触发后台整理——任何入口（extract / interview / 其它能力）
    经此落盘都会走向 consolidate，而不再只有 extract 一条路径能触发。"""
    now = _now_iso()
    record.slug = record.slug or _slugify(record.name)
    record.created_at = record.created_at or now
    record.updated_at = now
    _write_memory_atomic(record, rebuild)
    _schedule_consolidation_if_needed()
    return record


def _schedule_consolidation_if_needed(cfg=None) -> None:
    """统一收敛触发点：库达标、且不在整理中/冷却期/无 pending 时，调度一次 consolidate。

    事件循环感知：有 running loop → create_task；否则（同步上下文，如 interview tool）
    起守护线程 asyncio.run 兜底。供 write_memory 及 extract 路径调用。
    """
    global _consolidate_pending
    cfg = cfg or get_agent_config()
    if _consolidating or _consolidate_pending:
        return
    if cfg.longterm_consolidate_cooldown and (
        time.monotonic() - _last_consolidate_done < cfg.longterm_consolidate_cooldown
    ):
        return
    if len(list_memories()) < cfg.longterm_consolidate_threshold:
        return
    _consolidate_pending = True
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(target=_run_consolidate_in_thread, daemon=True).start()
    else:
        asyncio.create_task(_consolidate_task())
    logger.info("longterm consolidate scheduled (threshold=%d)", cfg.longterm_consolidate_threshold)


async def _consolidate_task() -> None:
    """事件循环侧的一次性整理任务：跑完复位 pending（consolidate 内部再置 _consolidating）。"""
    global _consolidate_pending
    try:
        await consolidate_memories(None)
    finally:
        _consolidate_pending = False


def _run_consolidate_in_thread() -> None:
    """无事件循环上下文时（同步调用方）的兜底线程：私有 loop 跑一次整理。"""
    if _consolidating:
        return
    try:
        asyncio.run(consolidate_memories(None))
    except Exception as e:  # noqa: BLE001 - 整理失败由 consolidate 内部兜底
        logger.warning("longterm consolidate thread failed: %s", e)
    finally:
        _consolidate_pending = False


def delete_memory(slug: str, rebuild: bool = True) -> None:
    """删除一条记忆（测试/维护/整理用）。"""
    try:
        os.remove(memory_path_for(slug))
    except FileNotFoundError:
        return
    if rebuild:
        rebuild_index()


def _clear_state_memories() -> None:
    """清掉库里全部 state 型记忆：state 是「最新情绪快照」，覆盖式更新，只存一份。"""
    for rec in list_memories():
        if rec.type == "state":
            delete_memory(rec.slug, rebuild=False)


def rebuild_index() -> None:
    """重建 data/memory/_INDEX：人读/排查用的目录汇总，运行时永不加载它。"""
    records = list_memories()
    lines = [f"- [{r.type}] {r.name} — {r.description}（{r.slug}.md）" for r in records]
    path = os.path.join(memory_dir(), f"{_INDEX_NAME}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".idx-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# 长期记忆目录（自动生成）\n" + ("\n".join(lines) or "(empty)") + "\n")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  召回：短目录 → 打分选 top-k → 按需加载正文
# ════════════════════════════════════════════════════════════════════════════

async def recall_longterm_memories(user_text: str) -> str:
    """
    召回与本轮用户请求相关的长期记忆，返回要注入的正文文本（多行）；
    无命中或任何异常返回 ""。不抛异常、不阻塞主回复。
    """
    cfg = get_agent_config()
    if not cfg.longterm_enabled:
        return ""
    if not user_text or not user_text.strip():
        return ""
    try:
        records = list_memories()
        if not records:
            return ""
        docs = [f"[{r.type}] {r.name}：{r.description}" for r in records]
        top_idx = await _select_indices(user_text, records, docs, cfg)
        if not top_idx:
            return ""
        return _build_injectable(records, top_idx, cfg.longterm_recall_char_limit)
    except Exception as e:  # noqa: BLE001
        logger.exception("recall_longterm_memories failed: %s", e)
        return ""


async def _select_indices(query: str, records: list[MemoryRecord], docs: list[str],
                          cfg) -> list[int]:
    """选中最相关的记忆下标（顺序），recall_mode=model 时走 LLM 选择。"""
    if cfg.longterm_recall_mode == "model":
        return await _select_by_model(query, docs, cfg)
    return await _select_by_score(query, records, docs, cfg)


async def _select_by_score(query: str, records: list[MemoryRecord], docs: list[str],
                           cfg) -> list[int]:
    """零延迟打分：rerank(默认/确诊可用) 失败退关键词。只对短目录打分，不读正文。"""
    k = max(1, cfg.longterm_recall_top_k)
    if cfg.longterm_recall_backend != "keyword" and _rerank_ready():
        try:
            from infra_ai import async_rerank
            resp = await async_rerank(query, docs, top_n=k)
            idxs = [r.index for r in resp.results if 0 <= r.index < len(records)]
            if idxs:
                return idxs[:k]
            logger.warning("rerank recall returned empty, fallback keyword")
        except Exception as e:  # noqa: BLE001
            logger.warning("rerank recall failed (%s), fallback keyword", e)
    return _select_by_keyword(query, records, docs, k)


def _rerank_ready() -> bool:
    """rerank 是否真正可用（至少一条 enabled 候选且 api_key 非空）。

    否则直接跳过多路重试、走关键词——避免未配置/缺 key 时 async_rerank
    内部 2s+4s 退避重试把「零延迟」召回拖慢成 ~6s。
    """
    try:
        from infra_ai.rerank import _get_rerank_candidates
        for c in _get_rerank_candidates("text"):
            if c.api_key:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _select_by_keyword(query: str, records: list[MemoryRecord], docs: list[str],
                       k: int) -> list[int]:
    """关键词兜底：对短目录（name+description）做重叠分词打分。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored: list[tuple[int, int]] = []
    for i, doc in enumerate(docs):
        d_tokens = set(_tokenize(doc))
        hits = sum(1 for t in q_tokens if t in d_tokens)
        if hits:
            scored.append((i, hits))
    scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    return [i for i, _ in scored[:k]]


def _tokenize(text: str) -> list[str]:
    """把查询/短目录切成可比较的词：≥3 英文字母的词 + 连续中文字串。"""
    tokens = []
    for m in re.finditer(r"[a-zA-Z]{3,}|[一-鿿]+", text.lower()):
        t = m.group(0)
        if len(t) >= 2:
            tokens.append(t)
    return tokens


async def _select_by_model(query: str, docs: list[str], cfg) -> list[int]:
    """s09 模型选择路径：轻量 LLM 调用返回相关索引，再按需加载正文。"""
    from infra_ai import async_call_llm
    k = max(1, cfg.longterm_recall_top_k)
    catalog = "\n".join(f"{i}: {d}" for i, d in enumerate(docs))
    prompt = (
        "下面是长期记忆目录。选出与当前用户请求相关的条目索引，"
        f"最多 {k} 条。只输出 JSON 数组，如 [0,2]。无相关输出 []。\n"
        f"目录：\n{catalog}\n当前请求：{query}"
    )
    raw = await async_call_llm(
        [
            {"role": "system", "content": "You select relevant memory indexes."},
            {"role": "user", "content": prompt},
        ],
        use_json=True,
        extra={"kind": "longterm_select"},
        model_kwargs={"max_tokens": 64},
    )
    arr = _extract_json_array(raw or "")
    out = []
    for v in arr:
        if isinstance(v, int) and 0 <= v < len(docs) and v not in out:
            out.append(v)
        if len(out) >= k:
            break
    return out


def _build_injectable(records: list[MemoryRecord], top_idx: list[int], char_limit: int) -> str:
    """把命中的正文拼成有界的注入文本（超限截断），不含外层 <长期记忆> 标签。"""
    lines: list[str] = []
    total = 0
    for i in top_idx:
        r = records[i]
        if not r.body:
            continue
        piece = f"[{r.type}] {r.name}：{r.body.replace('\\n', ' ')}"
        if total + len(piece) > char_limit and lines:
            break
        lines.append(piece)
        total += len(piece)
        if total >= char_limit:
            break
    return "\n".join(lines)


def inject_memory_block(messages: list[dict], block: str) -> list[dict]:
    """
    把长期记忆块做成 system 消息插入 messages，紧随第一条非 system 之前（即在
    历史摘要/系统提示之后）。block 为空时原样返回。
    """
    if not block:
        return messages
    block_msg = {"role": "system", "content": f"<长期记忆>\n{block}\n</长期记忆>"}
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            msgs.insert(i, block_msg)
            return msgs
    msgs.append(block_msg)
    return msgs


# ════════════════════════════════════════════════════════════════════════════
#  提取：回合结束后后台调用，候选经准入才落盘
# ════════════════════════════════════════════════════════════════════════════

# 明显寒暄/单字回应，不值得开一次后台提取 LLM
_TRIVIAL_UTTERANCES = frozenset({
    "嗯", "嗯嗯", "额", "哦", "噢", "好", "好的", "好吧", "对", "对的", "ok", "okay",
    "在", "在的", "在吗", "哈", "嗨", "hi", "hello", "你好", "再见", "拜拜",
})


def _has_durable_signal(user_msg: str) -> bool:
    """低成本预滤：去掉寒暄/标点后无实质内容则跳过，不调度提取 LLM（挡空轮）。"""
    text = (user_msg or "").strip().strip("。，,.!！?？")
    if not text:
        return False
    if text.lower() in _TRIVIAL_UTTERANCES:
        return False
    return bool(_tokenize(text))


def _should_extract_now(user_msg: str) -> bool:
    """触发节奏：every_n_turns 按模块计数取余且信号非空；every_turn 也过空轮预滤。"""
    global _extract_round_ctr
    cfg = get_agent_config()
    if cfg.longterm_extract_trigger != "every_n_turns":
        return _has_durable_signal(user_msg)
    n = cfg.longterm_extract_every_n
    if n <= 0:
        return _has_durable_signal(user_msg)
    _extract_round_ctr += 1
    if _extract_round_ctr % n != 0:
        return False
    return _has_durable_signal(user_msg)


def _emit_memory_meta(mtype: str, payload: dict) -> None:
    """OBS 观测：往当前 trace 发一条记忆维护元数据（业务量），失败静默不影响主流程。"""
    try:
        from obs import emit
        ev = {"type": mtype}
        ev.update(payload)
        emit(ev)
    except Exception:  # noqa: BLE001 - 观测失败不影响记忆主流程
        pass


async def extract_longterm_memory(session_id: str, last_user_msg: str, reply: str,
                                  context: str = "") -> None:
    """后台任务入口（与 compress_and_save 同款）：提取并落盘，异常不外抛。
    context：最近几轮原文（含本轮），供提取器识别自介/身份类陈述；空则退回单轮。"""
    cfg = get_agent_config()
    if not cfg.longterm_enabled:
        return
    if not reply or not reply.strip():
        return
    if not _should_extract_now(last_user_msg):
        return
    try:
        from obs import new_trace
        with new_trace(session_id, kind="longterm_extract"):
            candidates = await _call_extract(last_user_msg, reply, cfg, context)
            existing = list_memories()
            sink_slugs = {m.slug for m in existing}
            seen_bodies = {_norm(m.body) for m in existing}
            written = 0
            skipped = 0
            state_cleared = False
            for cand in (candidates or []):
                rec = _candidate_to_record(cand)
                is_update = (cand.get("op") or "add") == "update"
                target = None
                if is_update:
                    target = _match_existing(cand, existing)
                    if target is not None:
                        # 覆写既有条目：保留 slug/created_at，body/description/type 以本轮为准
                        rec.slug = target.slug
                        rec.created_at = target.created_at
                        rec.name = rec.name or target.name
                ok, reason = should_store_memory(
                    cand, rec, sink_slugs, seen_bodies, allow_update=(target is not None)
                )
                if not ok:
                    if rec.type == "state":
                        logger.debug("longterm state skip (%s): %s", rec.name, reason)
                    skipped += 1
                    continue
                # state 是「最新情绪快照」：写入前清掉库里旧的情绪记忆，
                # 无论新条目名是否与前一条相同，都以本轮为准覆盖。
                if rec.type == "state" and not state_cleared:
                    _clear_state_memories()
                    sink_slugs = {m.slug for m in list_memories()}
                    state_cleared = True
                write_memory(rec, rebuild=False)   # 内部已统一判定是否触发整理
                sink_slugs.add(rec.slug)
                seen_bodies.add(_norm(rec.body))
                written += 1
                verb = "updated" if is_update else "saved"
                logger.info("longterm %s [%s] %s", verb, rec.type, rec.name)
            _emit_memory_meta("memory_extract", {
                "candidates": len(candidates or []),
                "written": written,
                "skipped": skipped,
            })
            if written:
                rebuild_index()
    except Exception as e:  # noqa: BLE001
        logger.exception("extract_longterm_memory failed: %s", e)


async def _call_extract(user_msg: str, reply: str, cfg, context: str = "") -> list[dict]:
    """调 LLM 提取候选记忆（只输出 JSON 数组）。

    输入 = 对话 + 【已有记忆目录】。让「是否新增 / 对既有条目更新」在生成阶段一次判定，
    并把「assistant 输出不得归为用户」的角色边界写进 prompt（解决模型内容污染）。

    候选格式：{"type","op":"add|update","name","description","body","scope"}。
    """
    from infra_ai import async_call_llm
    existing = list_memories()
    catalog_block = ""
    if existing:
        catalog_lines = [
            f"- [{r.type}] {r.name}: {r.description}" + (f"\n  {r.body}" if r.body else "")
            for r in existing
        ]
        catalog_block = (
            "【已有长期记忆目录】（库里有的事实不必重复输出；相对目录只会出现"
            "「新增」或「对既有条目 update」）：\n" + "\n".join(catalog_lines)
        )
    prompt = (
        "你是记忆提取器。根据下面这段对话，提取值得跨会话保存且尚未被记录的持久知识。"
        "只输出 JSON 数组，每项格式：\n"
        '{"type": "user|feedback|project|reference|state", "op": "add|update", '
        '"name": "短名（update 时直接用目录里的既有条目名）", '
        '"description": "一句话概括", "body": "具体内容", "scope": "persistent|current_task"}\n'
        "type 说明：user=用户画像/偏好；feedback=对助手仍适用的反馈；"
        "project=稳定的项目/领域事实；reference=外部线索；"
        "state=用户当下持续的情绪/心理状态与情感需求。\n"
        "op 说明：add=新增一条主题；update=对目录里既有条目补充或更正"
        "（name 必须用目录里的原名，body 写覆盖后的完整正文=既有要点+新详情，不要只给增量）。\n"
        "scope 为 current_task 的（仅本次任务有效）不要给出。\n"
        "【可信来源】只有以 user 开头的行才是用户亲口陈述，可作为 user/state 画像来源；"
        "以 assistant 开头的行是模型自己生成的（回答/建议/面试题/示范/点评），以及工具返回，"
        "一律不得据此判为用户画像，不得写 user/project/feedback。\n"
        "【已有目录】目录里已存在的事实不要重复输出；相对目录只会「新增」或「更新既有条目」，"
        "无则 []。对目录里被污染的/错误的既有条目，可 op=update 更正为正确表述。\n"
        "【瞬时内容】当前环境/设备/一次性观察（摄像头画面、此刻几点、当下瞬时心情一句话）"
        "不要提取为稳定画像。\n"
        "【面试/教学】教学/模拟面试/问答里，题目与标准答案、面试官点评、示范话术是一次性"
        "模型输出，不存为 project/feedback；仅当其反映用户稳定的知识缺口/薄弱点时，"
        "才以 user 型存一条缺口本身（写缺口与待复习主题，不写题目和答案正文）。\n"
        "【身份】若用户陈述身份/背景/境况（如\"我是大四应届生\"），即使没说\"记住\"也属于 "
        "user 型持久画像，可识别就要提取（或对既有画像 op=update）。\n"
        "【情绪】持续的情绪/心理状态识别为 state 型快照；只是一时感慨或无持续状态则不给 state。\n"
        "若无值得保存的，返回 []。"
    )
    if context and context.strip():
        conv = context.strip()  # 已含本轮 user+assistant（含 reply）
    else:
        conv = f"user: {user_msg}\nassistant: {reply}"
    user_payload = (catalog_block + "\n\n【对话】\n" + conv) if catalog_block else conv
    raw = await async_call_llm(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_payload},
        ],
        use_json=True,
        extra={"kind": "longterm_extract"},
        model_name=cfg.longterm_extract_model,  # 显式单模型覆盖，优先于能力路由
        model_kwargs={"max_tokens": 1024},
        capability="extract",
    )
    arr = _extract_json_array(raw or "")
    out = []
    for item in arr:
        if isinstance(item, dict):
            out.append(item)
    if not out:
        logger.warning("extract LLM returned no candidates (mode=%s)", cfg.longterm_recall_mode)
    return out


def _candidate_to_record(cand: dict) -> MemoryRecord:
    now = _now_iso()
    rec = MemoryRecord(
        name=(cand.get("name") or "").strip()[:80],
        description=(cand.get("description") or "").strip()[:200],
        type=(cand.get("type") or "").strip(),
        body=(cand.get("body") or "").strip(),
        created_at=now,
        updated_at=now,
    )
    if rec.name:
        rec.slug = _slugify(rec.name)
    return rec


def _match_existing(cand: dict, existing: list[MemoryRecord]) -> MemoryRecord | None:
    """按 cand 声明的 name（update 目标）匹配既有条目；_norm 大小写/标点兜底。"""
    name = (cand.get("name") or "").strip()
    if not name:
        return None
    n = _norm(name)
    if not n:
        return None
    for r in existing:
        if n == _norm(r.name) or (r.slug and n == _norm(r.slug)):
            return r
    return None


def should_store_memory(cand: dict, rec: MemoryRecord, sink_slugs: set[str],
                        seen_bodies: set[str], allow_update: bool = False) -> tuple[bool, str]:
    """准入判断：scope 非 persistent / 字段不全 / 临时性 / 类型不允 → 拒绝。
    精确查重（slug/body）仅对 add 生效；update 覆写既有条目时放行覆盖。"""
    if (cand.get("scope") or "persistent") != "persistent":
        return False, "scope != persistent"
    if rec.type not in MEMORY_TYPES:
        return False, f"bad type {rec.type!r}"
    cfg = get_agent_config()
    if cfg.longterm_store_types and rec.type not in cfg.longterm_store_types:
        return False, f"type {rec.type} not in store_types"
    if not rec.name or not rec.body:
        return False, "missing name/body"
    if allow_update:
        return True, "ok (update)"
    rec.slug = _slugify(rec.name)
    if not rec.slug or rec.slug in sink_slugs:
        return False, "duplicate slug"
    body_norm = _norm(rec.body)
    if body_norm in seen_bodies:
        return False, "duplicate body"
    hay = _norm(f"{rec.name} {rec.description} {rec.body}")
    for marker in TEMPORARY_MARKERS:
        if marker in hay:
            return False, f"temporary marker: {marker}"
    return True, "ok"


def _norm(s: str) -> str:
    """归一化用于查重：去空白/标点/换行、小写。"""
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def _extract_json_array(text: str) -> list[Any]:
    """从模型输出里稳妥取出 JSON 数组；失败返回 []。"""
    text = text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            pass
    return []


def _extract_json_obj(text: str) -> dict:
    """从模型输出里稳妥取出 JSON 对象；失败返回 {}。"""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


# ════════════════════════════════════════════════════════════════════════════
#  整理：阈值触发，后台增量归并（removals/merges/untouched），snapshot 回滚
# ════════════════════════════════════════════════════════════════════════════

def _resolve_record(by_key: dict[str, MemoryRecord], name: str) -> MemoryRecord | None:
    """在 {_norm: record} 索引里按名称解析记录；查不到返回 None。"""
    n = _norm(name or "")
    if not n:
        return None
    return by_key.get(n)


async def consolidate_memories(session_id: str) -> None:
    """后台整理（独立 trace、异常不外抛、失败回滚）：**增量归并**，不做全库重写。

    由 LLM 决策 removals/merges/untouched 三操作集，应用时只删/并涉及条目，
    untouched 的稳定主题字节级不动。入口置 _consolidating 防重入；完成（含失败）
    都推进冷却基准，避免频闪重试。
    """
    global _consolidating, _last_consolidate_done
    if _consolidating:
        return
    cfg = get_agent_config()
    try:
        records = list_memories()
        if len(records) < cfg.longterm_consolidate_threshold:
            return
        _consolidating = True
        from obs import new_trace
        with new_trace(session_id, kind="longterm_consolidate"):
            ops = await _call_consolidate(records, cfg)
            _emit_memory_meta("memory_consolidate", {
                "before": len(records),
                "removals": len(ops.get("removals") or []),
                "merges": len(ops.get("merges") or []),
                "untouched": len(ops.get("untouched") or []),
            })
            if not (ops.get("removals") or ops.get("merges")):
                return
            snapshot = _snapshot_files()
            try:
                by_key = {_norm(r.name): r for r in records}
                # 存活集 = untouched ∪ merge 目标；未被提及的条目视为被删（确定性收敛）
                kept: set[str] = set()
                for name in ops.get("untouched") or []:
                    r = _resolve_record(by_key, name)
                    if r:
                        kept.add(r.slug)
                for m in ops.get("merges") or []:
                    t = _resolve_record(by_key, m.get("into"))
                    if t:
                        kept.add(t.slug)
                for r in records:
                    if r.slug not in kept:
                        delete_memory(r.slug, rebuild=False)
                        by_key.pop(_norm(r.name), None)
                # 应用 merge：保留目标 slug/created_at，覆写 body/description
                for m in ops.get("merges") or []:
                    t = _resolve_record(by_key, m.get("into"))
                    if not t:
                        continue
                    rec = MemoryRecord(
                        name=t.name,
                        description=m.get("into_desc") or t.description,
                        type=t.type,
                        body=m.get("body") or t.body,
                        slug=t.slug,
                        created_at=t.created_at,
                    )
                    write_memory(rec, rebuild=False)
                    by_key[_norm(rec.name)] = rec
                rebuild_index()
                logger.info("longterm consolidated %d -> %d",
                            len(records), len(list_memories()))
            except Exception:
                _restore_snapshot(snapshot)
                raise
    except Exception as e:  # noqa: BLE001
        logger.exception("consolidate_memories failed: %s", e)
    finally:
        _last_consolidate_done = time.monotonic()
        _consolidating = False


async def _call_consolidate(records: list[MemoryRecord],
                            cfg=None) -> dict:
    """调 LLM 决策**收敛操作集**（removals/merges/untouched），而非清理后的整库。

    只有超出阈值时才被调用；要求 LLM 只做必要收敛、稳定条目进 untouched，
    merge 目标 body 保留多句细节（不精炼成一句话）。返回 dict，应用方解析。
    """
    cfg = cfg or get_agent_config()
    keep = max(1, cfg.longterm_consolidate_keep)
    from infra_ai import async_call_llm
    present = "\n".join(
        f"- [{r.type}] {r.name}: {r.description}\n  {r.body}" for r in records
    )
    prompt = (
        "下面是长期记忆库（已超出上限条数）。只做【必要收敛】，不要全库重写、"
        "不要复述未变化的条目。只输出一个 JSON 对象：\n"
        '{"removals": ["确已过时/一次性/已失效的条目名", ...],\n'
        ' "merges": [{"into": "保留的既有条目名", "into_desc": "（可改的一句话概括）", '
        '"from": ["被并进/删除的条目名", ...], "body": "合并后的完整正文"}],\n'
        ' "untouched": ["未变化的条目名", ...]}\n'
        "规则：含义相近才 merge 成一条；into 必须取既有条目名，body 保留足够细节、可多句，"
        "把 from 各条要点并入，不要只剩一句话；确已过时才 removal；其余稳定条目一律进 untouched，"
        "原文不动。合并后总条数必须少于当前并 ≤ "
        f"{keep}。state 型（当前情绪/状态快照）至多保留最新一条。"
        "不要新增对话里没有的内容。"
    )
    raw = await async_call_llm(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": present},
        ],
        use_json=True,
        extra={"kind": "longterm_merge"},
        model_kwargs={"max_tokens": 4096},
        capability="consolidate",
    )
    ops = _extract_json_obj(raw or "")
    by_key = {_norm(r.name): r for r in records}
    removals = [x for x in ops.get("removals") or [] if isinstance(x, str)]
    merges = []
    for m in ops.get("merges") or []:
        if not (isinstance(m, dict) and (m.get("into") or "").strip()):
            continue
        if _resolve_record(by_key, m.get("into")) is None:
            continue  # into 解析不到既有条目，丢弃该 merge
        merges.append({
            "into": (m.get("into") or "").strip(),
            "into_desc": (m.get("into_desc") or "").strip(),
            "from": [x for x in m.get("from") or [] if isinstance(x, str)],
            "body": (m.get("body") or "").strip(),
        })
    untouched = [x for x in ops.get("untouched") or [] if isinstance(x, str)]
    return {"removals": removals, "merges": merges, "untouched": untouched}


def _snapshot_files() -> dict[str, str]:
    """把当前全部记忆文件内容读进内存，供失败恢复。"""
    snap: dict[str, str] = {}
    for path in glob.glob(os.path.join(memory_dir(), "*.md")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap[os.path.basename(path)] = f.read()
        except OSError:
            pass
    return snap


def _restore_snapshot(snapshot: dict[str, str]) -> None:
    """用 snapshot 覆盖回写（重建被破坏的记忆库）。"""
    for slug in [os.path.splitext(basename)[0] for basename in snapshot]:
        delete_memory(slug, rebuild=False)
    for basename, content in snapshot.items():
        path = os.path.join(memory_dir(), basename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".rst-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    rebuild_index()
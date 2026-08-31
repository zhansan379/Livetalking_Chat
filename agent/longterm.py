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

def _should_extract_now() -> bool:
    """触发节奏：every_turn 恒真；every_n_turns 按模块计数取余。"""
    global _extract_round_ctr
    cfg = get_agent_config()
    if cfg.longterm_extract_trigger != "every_n_turns":
        return True
    n = cfg.longterm_extract_every_n
    if n <= 0:
        return True
    _extract_round_ctr += 1
    return _extract_round_ctr % n == 0


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
    if not _should_extract_now():
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
                ok, reason = should_store_memory(cand, rec, sink_slugs, seen_bodies)
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
                logger.info("longterm saved [%s] %s", rec.type, rec.name)
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
    """调 LLM 提取候选记忆（只输出 JSON 数组），解析失败返回 []。"""
    from infra_ai import async_call_llm
    prompt = (
        "你是记忆提取器。从下面这段对话中，提取值得跨会话保存的持久知识，"
        "忽略一次性/临时性信息。只输出 JSON 数组，每项格式：\n"
        '{"type": "user|feedback|project|reference|state", "name": "短名", '
        '"description": "一句话概括", "body": "具体内容", "scope": "persistent|current_task"}\n'
        'type 说明：user=用户画像/偏好；feedback=对助手仍适用的反馈；'
        "project=稳定的项目/领域事实；reference=外部线索；"
        "state=用户当下持续的情绪/心理状态与情感需求。\n"
        "scope 为 current_task 的（仅本次任务有效）不要给出。\n"
        "特别注意：当前环境/设备/一次性观察（如摄像头画面拍到什么、此刻几点、"
        "当下瞬时的心情一句话）不要提取为 user 画像——这些属于当前会话的瞬时状态，"
        "不应成为跨会话的稳定画像。\n"
        "特别注意：教学/模拟面试/问答场景里，题目与标准答案、示范话术是一次性内容，"
        "不要存为 project/feedback；仅当其反映用户稳定的知识缺口/薄弱点时，"
        "才以 user 型存一条缺口本身（写缺口与待复习主题，不写题目和答案正文）。\n"
        "特别注意：若对话中用户陈述了自己的身份/背景/境况（如\"我是大四应届生\"、"
        "\"我学会计\"），即使没说\"记住\"也属于 user 型持久画像，可识别就要提取。\n"
        "情绪状态方面：若用户流露出正在经历/持续的情绪或心理状态（如职业迷茫、"
        "自我怀疑、压抑、焦虑、低落，或需要倾诉/鼓励/陪伴），且看似会持续一段时间"
        "而非一时感慨，识别为 state 型快照，写清具体情绪与希望得到怎样的回应；"
        "若只是一次性情绪或无明显持续状态，不要给 state。\n"
        "若无值得保存的，返回 []。"
    )
    if context and context.strip():
        conv = context.strip()  # 已含本轮 user+assistant（含 reply）
    else:
        conv = f"用户：{user_msg}\n助手：{reply}"
    raw = await async_call_llm(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conv},
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


def should_store_memory(cand: dict, rec: MemoryRecord, sink_slugs: set[str],
                        seen_bodies: set[str]) -> tuple[bool, str]:
    """准入判断：scope 非 persistent / 字段不全 / 临时性 / 类型不允 / 重复 → 拒绝。"""
    if (cand.get("scope") or "persistent") != "persistent":
        return False, "scope != persistent"
    if rec.type not in MEMORY_TYPES:
        return False, f"bad type {rec.type!r}"
    cfg = get_agent_config()
    if cfg.longterm_store_types and rec.type not in cfg.longterm_store_types:
        return False, f"type {rec.type} not in store_types"
    if not rec.name or not rec.body:
        return False, "missing name/body"
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


# ════════════════════════════════════════════════════════════════════════════
#  整理：阈值触发，后台去重/合并，snapshot 回滚
# ════════════════════════════════════════════════════════════════════════════

async def consolidate_memories(session_id: str) -> None:
    """后台整理（与 compress 同款模板：独立 trace、异常不外抛、失败回滚）。
    入口置 _consolidating 防重入；完成（含失败）都推进冷却基准，避免频闪重试。"""
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
            cleaned = await _call_consolidate(records, cfg)
            _emit_memory_meta("memory_consolidate", {
                "before": len(records), "after": len(cleaned),
            })
            if not cleaned:
                return
            snapshot = _snapshot_files()
            try:
                for r in records:
                    delete_memory(r.slug, rebuild=False)
                for r in cleaned:
                    write_memory(r, rebuild=False)
                rebuild_index()
                logger.info("longterm consolidated %d -> %d", len(records), len(cleaned))
            except Exception:
                _restore_snapshot(snapshot)
                raise
    except Exception as e:  # noqa: BLE001
        logger.exception("consolidate_memories failed: %s", e)
    finally:
        _last_consolidate_done = time.monotonic()
        _consolidating = False


async def _call_consolidate(records: list[MemoryRecord],
                            cfg=None) -> list[MemoryRecord]:
    """调 LLM 把整库去重/合并，返回清理后的记录列表（条数硬封顶，保证收敛）。

    keep = longterm_consolidate_keep（须 < threshold）：LLM 一旦照抄不删，
    这里强制截到 keep 条以内，写回后必 < threshold，从而终结「整理完又触发」空转。
    """
    cfg = cfg or get_agent_config()
    keep = cfg.longterm_consolidate_keep
    from infra_ai import async_call_llm
    present = "\n".join(
        f"- [{r.type}] {r.name}: {r.description}\n  {r.body}" for r in records
    )
    prompt = (
        "下面是长期记忆库。请去重、合并含义相近的条目、应用较新的更正、"
        "剔除不再有用或已失效的内容。只输出清理后的 JSON 数组，每项格式：\n"
        '{"type": "user|feedback|project|reference|state", "name": "短名", '
        '"description": "一句话概括", "body": "具体内容", "scope": "persistent"}\n'
        f"合并后总条数必须严格 ≤ {keep} 条。含义相近的多条必须并成一条；"
        "过时/一次性/环境观察直接删除；每条 body 精炼成一句话。"
        "宁可丢细节，总量也必须 ≤ " + str(keep) + "；不要新增对话里没有的内容。"
        "state 型条目（当前情绪/状态快照）至多保留最新一条。"
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
    arr = _extract_json_array(raw or "")
    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        rec = _candidate_to_record(item)
        if rec.type in MEMORY_TYPES and rec.name and rec.body and rec.slug:
            out.append(rec)
    # 硬封顶：无论 LLM 照抄几条，都截到 keep 以内 → 写回后 <threshold，终结空转
    keep = max(1, cfg.longterm_consolidate_keep)
    if len(out) > keep:
        logger.warning("longterm consolidate LLM returned %d (cap %d), trim", len(out), keep)
        out = out[:keep]
    return out


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
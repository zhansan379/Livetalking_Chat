###############################################################################
#  模拟面试 · 题库摄入 + chromadb 向量化存储与检索
#
#  题库来源改为真实数据 CSV（data/question_bank.csv 目录 + data/question_essay.csv 题目），
#  建索引时按 bank_id 关联出分类，灌进 chromadb（本地 onnx 内置模型向量化）落盘。
#  整本题库永不被塞进 prompt：开场仅经向量检索取 top-k。
#
#  约定：
#    · 建索引走独立脚本 scripts/build_interview_index.py（可反复重建，幂等）；
#    · 运行期只 search；若库缺失在 search 内兜底自动建，避免空手上阵。
#    · 外部可用配置 interview_bank_csv / interview_essay_csv / interview_index_dir 覆盖。
###############################################################################

import csv
import os

from utils.logger import logger

DEFAULT_BANK_CSV = os.path.join("data", "question_bank.csv")
DEFAULT_ESSAY_CSV = os.path.join("data", "question_essay.csv")

_ANSWER_TRUNC = 2000    # metadata 里参考答案置顶截断，控 chromadb 体积
_BATCH = 500            # 逐批 add，避免一次性超大 payload

# 题库名清洗：去掉固定尾缀（按长到短），得到可作分类标签的技能主体。
_BANK_NAME_TAILS = (
    "面试题及答案整理，最新面试题",
    "实战，答案整理，最新面试题",
    "新特性实战，答案整理，最新面试题",
    "，答案整理，最新面试题",
    "面试题及答案整理",
    "答案整理，最新面试题",
    "，最新面试题",
    "，高级面试题",
    "最新面试题",
    "答案整理",
    "面试题",
)


def _clean_bank_name(name: str) -> str:
    """题库名 → 分类标签：先去 '_来源注释' 段，再循环剥固定尾缀，保留技能主体。"""
    s = (name or "").strip()
    if not s:
        return s
    # 纯 ASCII（如 interview_questions）不做切分，仅把下划线换空格
    if s.replace("_", "").isascii() and not any(ord(c) > 127 for c in s):
        return s.replace("_", " ").strip()
    if "_" in s:
        s = s.split("_")[0].strip()
    changed = True
    while changed:
        changed = False
        for tail in _BANK_NAME_TAILS:
            if s.endswith(tail):
                s = s[: -len(tail)].strip()
                changed = True
                break
    return s.strip(" ，、-_")


def _index_dir(cfg) -> str:
    return getattr(cfg, "interview_index_dir", None) or "data/capabilities/interview/chroma"


def _bank_csv(cfg) -> str:
    return getattr(cfg, "interview_bank_csv", None) or DEFAULT_BANK_CSV


def _essay_csv(cfg) -> str:
    return getattr(cfg, "interview_essay_csv", None) or DEFAULT_ESSAY_CSV


def _reader(path: str):
    """utf-8-sig 打开 + 规整列名（去掉 BOM 前缀），兼容不同导出文件。"""
    f = open(path, "r", encoding="utf-8-sig")
    reader = csv.DictReader(f)
    if reader.fieldnames:
        reader.fieldnames = [str(n or "").lstrip("﻿").strip() for n in reader.fieldnames]
    return reader


def _load_bank_catalog(path: str) -> dict[str, dict]:
    """读题库目录：question_bank_id → {category(题库名), channel(广泛分类 relative_position)}。

    插入分类按题库名(question_bank_name)走；relative_position 仅作为广泛的兜底分类保留在 channel。
    """
    catalog: dict[str, dict] = {}
    try:
        for row in _reader(path):
            bid = (row.get("question_bank_id") or "").strip()
            if not bid:
                continue
            catalog[bid] = {
                "category": _clean_bank_name(row.get("question_bank_name") or ""),
                "channel": (row.get("relative_position") or "").strip(),
            }
    except Exception as e:  # noqa: BLE001 - 目录读失败按无目录处理，仅题目可用
        logger.warning("interview bank catalog load failed (%s): %s", path, e)
    return catalog


def load_records(cfg) -> list[dict]:
    """读题目 CSV，按 bank_id 关联目录，归一化为入库记录（剔除 is_deleted）。"""
    catalog = _load_bank_catalog(_bank_csv(cfg))
    records: list[dict] = []
    try:
        for row in _reader(_essay_csv(cfg)):
                if (row.get("is_deleted") or "0").strip() != "0":
                    continue
                qid = (row.get("question_id") or "").strip()
                title = (row.get("title") or "").strip()
                if not qid or not title:
                    continue
                bid = (row.get("bank_id") or "").strip()
                meta = catalog.get(bid) or {}
                records.append({
                    "id": qid,
                    "text": title,
                    "category": meta.get("category") or "",
                    "channel": meta.get("channel") or bid,
                    "keywords": (row.get("keyword") or "").strip(),
                    "answer": (row.get("answer") or "").strip()[:_ANSWER_TRUNC],
                    "bank_id": bid,
                })
    except Exception as e:  # noqa: BLE001 - 题目读失败返回空，能力可降级
        logger.warning("interview essay load failed (%s): %s", _essay_csv(cfg), e)
    logger.info("interview bank records loaded: %d", len(records))
    return records


def _doc_text(rec: dict) -> str:
    """向量化用文档串：题干 + 分类 + 关键词 + 参考答案（语义锚点）。"""
    parts = [str(rec.get("text") or "")]
    if rec.get("category"):
        parts.append(str(rec["category"]))
    if rec.get("keywords"):
        parts.append(str(rec["keywords"]))
    if rec.get("answer"):
        parts.append(str(rec["answer"])[:500])
    return " | ".join(p for p in parts if p)


def _collection(client, create: bool):
    """在 client 上获取或创建 interview_bank 集合（chromadb 延迟 import，避免未安装时崩导入）。"""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    kw = dict(embedding_function=DefaultEmbeddingFunction(),
              metadata={"hnsw:space": "cosine"})
    if create:
        return client.get_or_create_collection("interview_bank", **kw)
    return client.get_collection("interview_bank")


def get_collection(cfg):
    """获取 interview 题库 chromadb 集合（本地 onnx 内置模型向量化）。"""
    from chromadb import PersistentClient
    client = PersistentClient(path=_index_dir(cfg))
    return _collection(client, create=True)


def build_index(cfg) -> int:
    """全量摄入 CSV → 重建向量集合（幂等）。返回入库条数。"""
    records = load_records(cfg)
    if not records:
        logger.warning("interview build_index: no records to index")
        return 0
    from chromadb import PersistentClient
    client = PersistentClient(path=_index_dir(cfg))
    try:
        client.delete_collection("interview_bank")
    except Exception:  # noqa: BLE001 - 首次无集合
        pass
    col = _collection(client, create=True)
    for i in range(0, len(records), _BATCH):
        chunk = records[i:i + _BATCH]
        col.add(
            ids=[r["id"] for r in chunk],
            documents=[_doc_text(r) for r in chunk],
            metadatas=[_meta(r) for r in chunk],
        )
    logger.info("interview indexed %d records into chromadb", len(records))
    return len(records)


def _meta(rec: dict) -> dict:
    """chromadb 允许的标量键值 metadata（全转 str，避免非标量入库报错）。"""
    return {
        "id": str(rec.get("id") or ""),
        "text": str(rec.get("text") or ""),
        "category": str(rec.get("category") or ""),
        "channel": str(rec.get("channel") or ""),
        "keywords": str(rec.get("keywords") or ""),
        "answer": str(rec.get("answer") or ""),
    }


def search(cfg, query: str, top_k: int) -> list[dict]:
    """向量检索：query 语义近邻 → top_k 条记录（确定性排序回退）。
    库缺失则兜底自动建一次（幂等），避免空手上阵。
    """
    if top_k <= 0:
        return []
    try:
        col = get_collection(cfg)
        if col.count() == 0:
            logger.info("interview chromadb empty, auto-build index first")
            build_index(cfg)
        resp = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        metas = resp.get("metadatas") or [[]]
        return [{k: (m.get(k) or "") for k in
                 ("id", "text", "category", "channel", "keywords", "answer")}
                for m in (metas[0] if metas else [])]
    except Exception as e:  # noqa: BLE001 - 检索失败返回空，调用方已降级
        logger.warning("interview bank search failed: %s", e)
        return []
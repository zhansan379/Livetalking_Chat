###############################################################################
#  模拟面试 · 题库加载与确定性过滤
#
#  整本题库永不被塞进 prompt：召回阶段按需读取。内置 bank_data.yaml 默认，
#  用户可用 data/interview_bank.yaml（或覆盖配置 bank_override）覆盖（沿用
#  config 的"内置默认 + 外部覆盖"惯例）。
###############################################################################

import os

import yaml

from utils.logger import logger

_BANK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_data.yaml")


def _load_yaml(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return list(data.get("bank") or [])
    except Exception as e:  # noqa: BLE001 - 题库加载失败按空处理，能力可降级
        logger.warning("interview bank load failed (%s): %s", path, e)
        return []


def load_bank(cfg) -> list[dict]:
    """加载题库：外部覆盖优先，否则内置。返回归一化后的题目列表。"""
    override = getattr(cfg, "interview_bank_override", None)
    if override and os.path.isfile(override):
        items = _load_yaml(override)
        if items:
            return items
    return _load_yaml(_BANK_FILE)


def filter_bank(cfg, role: str | None, level: str | None,
                category: str | None = None) -> list[dict]:
    """确定性过滤（轻量）：role/level/category 命中即保留（role='*' 表通用）。

    缩小候选池供后续 rerank/个性化作 top-k。无任何题可用时返回全库。
    """
    bank = load_bank(cfg)
    role = (role or "").strip()
    level = (level or "").strip()
    category = (category or "").strip()

    def _hit(q: dict) -> bool:
        qr = [str(x).strip() for x in (q.get("role") or [])]
        ql = [str(x).strip() for x in (q.get("level") or [])]
        qc = (q.get("category") or "").strip()
        if category and qc and qc != category:
            return False
        if role and qr and "*" not in qr and role not in qr:
            return False
        if level and ql and level not in ql:
            return False
        return True

    out = [q for q in bank if _hit(q)]
    return out or bank


def question_text(q: dict) -> str:
    """题目的可作为 prompt/模型理解的描述串（用于召回时拼 query）。"""
    return q.get("text", "")
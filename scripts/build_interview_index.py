#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模拟面试题库向量化建索引脚本（离线、幂等）。

读 data/question_bank.csv（目录）+ data/question_essay.csv（题目），按 bank_id 关联，
灌进 chromadb（本地 onnx 内置模型向量化），落盘到 index_dir。

用法：
    python scripts/build_interview_index.py
    python scripts/build_interview_index.py --bank-csv ... --essay-csv ... --out ...

可反复运行重建（每次先删旧集合再全量重建）。
"""

import argparse
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使 capabilities/config 等以项目根可导入


def _cfg(args) -> SimpleNamespace:
    """构造 bank.py 所需的轻量 cfg（覆盖脚本参数；缺省走内置 data 路径）。"""
    return SimpleNamespace(
        interview_bank_csv=args.bank_csv,
        interview_essay_csv=args.essay_csv,
        interview_index_dir=args.out,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 interview 题库 chromadb 向量索引")
    parser.add_argument("--bank-csv", default=None,
                        help="题库目录 CSV（默认 data/question_bank.csv）")
    parser.add_argument("--essay-csv", default=None,
                        help="题目 CSV（默认 data/question_essay.csv）")
    parser.add_argument("--out", default="data/capabilities/interview/chroma",
                        help="chromadb 落盘目录（默认 data/capabilities/interview/chroma）")
    args = parser.parse_args()

    os.chdir(_ROOT)  # 相对 data/ 路径以项目根为准
    from capabilities.interview import bank

    n = bank.build_index(_cfg(args))
    print(f"interview 向量索引构建完成：共入库 {n} 题 → {args.out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
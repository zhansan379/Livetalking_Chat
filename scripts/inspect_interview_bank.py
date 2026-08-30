#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""interview 题库 chromadb 巡检/演示（纯本地，无色网依赖）。

读 data/capabilities/interview/chroma 下已建好的题库向量库，做四件事：
  ① stats    ：总体规模 + 按分类(category)的题量分布
  ② sample   ：随机/指定抽几题预览（题干/分类/参考答案片段）
  ③ query    ：跑一条向量检索，展示 top-k 及相似度(距离)
  ④ probe    ：对某个 id 看它在库里的原文与元数据

用法：
    python scripts/inspect_interview_bank.py              # 全部默认：stats + sample 5
    python scripts/inspect_interview_bank.py stats
    python scripts/inspect_interview_bank.py query "Java 并发 线程池" --k 8
    python scripts/inspect_interview_bank.py query "vite 构建优化" --k 5 --json
    python scripts/inspect_interview_bank.py sample --n 8
    python scripts/inspect_interview_bank.py probe 15870
文案全部中文，控制台(如 PowerShell)乱码时可用 `--o utf8` 直接写到文件再打开。
"""

import argparse
import itertools
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _cfg(args) -> SimpleNamespace:
    return SimpleNamespace(interview_index_dir=args.out)


def _col(args):
    from capabilities.interview import bank
    return bank.get_collection(_cfg(args))


# ── ① 规模 + 分类分布 ────────────────────────────────────────────────
def _stats(col) -> None:
    n = col.count()
    print(f"题库总量：{n} 题")
    metas = col.get(include=["metadatas"])["metadatas"]
    from collections import Counter
    dist = Counter((m or {}).get("category") or "(未分类)" for m in metas)
    print(f"分类数：{len(dist)}")
    for cat, cnt in dist.most_common():
        bar = "#" * (int(cnt / max(1, n) * 50))
        print(f"  {cat:<24} {cnt:>6}  {bar}")


# ── ② 抽样预览 ───────────────────────────────────────────────────────
def _sample(col, n: int) -> None:
    total = col.count()
    n = max(1, min(n, total))
    got = col.get(include=["metadatas"], limit=n)
    ids = got["ids"]
    metas = got["metadatas"]
    print(f"\n抽取 {len(ids)} 题预览：")
    for i, (qid, m) in enumerate(zip(ids, metas), 1):
        m = m or {}
        ans = (m.get("answer") or "").replace("\n", " ").strip()
        print(f"#{i} [{qid}] ({m.get('category') or '无'}) {m.get('text') or ''}")
        if ans:
            print(f"     参考答案：{ans[:60]}{'…' if len(ans) > 60 else ''}")


# ── ③ 向量检索演示 ───────────────────────────────────────────────────
def _query(col, query: str, k: int, as_json: bool) -> None:
    k = max(1, min(k, col.count()))
    resp = col.query(query_texts=[query], n_results=k,
                     include=["metadatas", "distances"])
    ids = resp["ids"][0]
    dists = resp["distances"][0]
    metas = resp["metadatas"][0]
    if as_json:
        import json
        out = [{"id": i, "text": (m or {}).get("text"),
                "category": (m or {}).get("category"), "distance": round(float(d), 4)}
               for i, d, m in zip(ids, dists, metas)]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print(f"\n检索「{query}」，返回 {len(ids)} 条（distance 越小越相似）：")
    for rank, (i, d, m) in enumerate(zip(ids, dists, metas), 1):
        m = m or {}
        print(f"  #{rank:>2} d={float(d):.3f} [{i}] ({m.get('category') or '无'}) {m.get('text') or ''}")
    ds = [float(d) for d in dists]
    top, tail = ds[0], ds[-1]
    spread = tail - top
    print(f"  -- top1={top:.3f} top{len(ds)}={tail:.3f} 距离差={spread:.3f}"
          f"（差小=候选同质；差大=top1 明显胜出）")


# ── ④ 按 id 探原文 ───────────────────────────────────────────────────
def _probe(col, id_: str) -> None:
    got = col.get(ids=[id_], include=["metadatas", "documents"])
    if not got["ids"]:
        print(f"未找到 id={id_}")
        return
    m = got["metadatas"][0] or {}
    print(f"\nid={id_} 元数据：")
    for key in ("text", "category", "channel", "keywords"):
        print(f"  {key:<10}: {m.get(key) or ''}")
    ans = m.get("answer") or ""
    print(f"  参考答案 : {ans[:200]}{'…' if len(ans) > 200 else ''}  (共 {len(ans)} 字)")
    if got["documents"] and got["documents"][0]:
        print(f"  入库向量文档: {got['documents'][0][:160]}")


def main() -> int:
    p = argparse.ArgumentParser(description="interview 题库 chromadb 巡检/演示（本地）")
    p.add_argument("--out", default="data/capabilities/interview/chroma",
                   help="chromadb 落盘目录")
    p.add_argument("--o", dest="outfile", default=None,
                   help="把输出写到此文件(UTF-8)，控制台乱码时用")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("stats", help="规模+分类分布")
    q = sub.add_parser("query", help="向量检索演示")
    q.add_argument("text", help="检索文本，如 'Java 并发'")
    q.add_argument("--k", type=int, default=8, help="返回条数")
    q.add_argument("--json", action="store_true", help="输出 JSON")
    sm = sub.add_parser("sample", help="抽样预览")
    sm.add_argument("--n", type=int, default=5, help="抽几条")
    pr = sub.add_parser("probe", help="按 id 查原文")
    pr.add_argument("id", help="题目 id（如 15870）")

    args = p.parse_args()

    if args.outfile:
        sys.stdout = open(args.outfile, "w", encoding="utf-8")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")   # 保证管道/终端正确显示中文
        except Exception:
            pass

    col = _col(args)
    cmd = args.cmd or "default"
    if cmd in ("stats", "default"):
        _stats(col)
    if cmd in ("sample", "default"):
        _sample(col, args.n if cmd == "sample" else 5)
    if cmd == "query":
        _query(col, args.text, args.k, args.json)
    if cmd == "probe":
        _probe(col, args.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
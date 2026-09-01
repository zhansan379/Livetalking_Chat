#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载中文关键词唤醒（KWS）模型并落到本项目 models/kws/<model_id>/。

主路径（推荐）：[0] GitHub release 单文件 tar.bz2（tag=kws-models）→ 下载并解压，本机实测可达。
回退路径（任一成功即停）：
  [1] ModelScope git clone / 文件（该 kws 模型不在 MS 上，通常失败）
  [2] hf-mirror / huggingface file（该 kws 模型不在 HF 仓库上，通常失败）

用法：
  python scripts/fetch_kws_model.py       # 从 GitHub tar 装整套模型（约 32MB）

装好即完全离线可用；重启后端即可启用关键词唤醒（server/kws.py）。
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = os.environ.get(
    "KWS_MODEL_ID", "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
)
DEST = ROOT / "models" / "kws" / MODEL_ID

# 该模型的必需文件（encoder/joiner 用 int8 版减小体积；decoder 用 fp32）
_FILES = {
    "encoder": "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder": "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    "joiner": "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "tokens": "tokens.txt",
}

_SOURCES = [
    ("hf-mirror", "https://hf-mirror.com/{repo}/resolve/main/{file}"),
    ("huggingface", "https://huggingface.co/{repo}/resolve/main/{file}"),
]
_REPO = "k2-fsa/" + MODEL_ID

# KWS 官方不在 HF 仓库发布，而是以已编译 tar.bz2 打在 GitHub releases（tag=kws-models）。
# GitHub 在本机可达；这是最可靠的主路径。
_GITHUB_TAR = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_ID}.tar.bz2"
)


def progress_block(filename, count, block_size, total):
    done = min(count * block_size, total)
    pct = 100 * done / total if total else 0
    sys.stdout.write(f"\r  {filename}  {pct:5.1f}%  {done/1e6:.1f}MB/{total/1e6:.1f}MB")
    sys.stdout.flush()


def _auth_headers() -> dict:
    """若设置了 HF_TOKEN，返回带 Bearer 鉴权的请求头（用于 HF gated 仓库）。"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    token = token.strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    headers.update(_auth_headers())
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(f"    [x] {e.code} — gated/需鉴权；确认已设 HF_TOKEN 且账号已接受该模型条款")
        else:
            print(f"    [x] {e}")
        if dest.exists():
            dest.unlink()
        return False
    except urllib.error.URLError as e:
        print(f"    [x] {e}")
        if dest.exists():
            dest.unlink()
        return False


def try_modelscope() -> bool:
    """ModelScope 上有社区镜像则优先；未知命名时静默失败即可。"""
    ms_url = f"https://modelscope.cn/{_REPO}/resolve/master/"
    ok = all(
        download(ms_url + fname, DEST / fname)
        for fname in _FILES.values()
    )
    return ok


def discover_files() -> dict:
    """尽量用 HF API 实时列举仓库文件，挑出 encoder/decoder/joiner/tokens.txt；
    API 不可达（本机被墙）时回退到已知的 int8 文件名。"""
    names: list = []
    headers = {"User-Agent": "Mozilla/5.0"}
    headers.update(_auth_headers())   # gated 仓库 API 也需鉴权
    for api in (f"https://huggingface.co/api/models/{_REPO}",
                f"https://hf-mirror.com/api/models/{_REPO}"):
        try:
            req = urllib.request.Request(api, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            names = [s.get("rfilename", "") for s in (data or {}).get("siblings", [])]
            if names:
                break
        except Exception:  # noqa: BLE001
            continue
    if not names:
        return dict(_FILES)
    out: dict = {}
    for n in names:
        if n == "tokens.txt":
            out.setdefault("tokens.txt", n)
        elif n.startswith("encoder-") and n.endswith(".onnx"):
            out.setdefault("encoder", n)
        elif n.startswith("decoder-") and n.endswith(".onnx"):
            out.setdefault("decoder", n)
        elif n.startswith("joiner-") and n.endswith(".onnx"):
            out.setdefault("joiner", n)
    # 缺失键用回退默认名补齐
    return {**_FILES, **out}


def try_github_tar() -> bool:
    """GitHub releases 上 k2-fsa 把整套 KWS 模型打成单文件 tar.bz2（tag=kws-models），
    下到临时文件解压到 DEST。这是最可靠的主路径（GitHub 本机可达）。"""
    if DEST.is_dir() and (DEST / "tokens.txt").exists():
        return True  # 已装好
    tmp = DEST.parent / (MODEL_ID + ".tar.bz2.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"[0] GitHub release tar: {_GITHUB_TAR}")
    try:
        req = urllib.request.Request(_GITHUB_TAR, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 18)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(
                        f"\r  下载 {done/1e6:.1f}MB/{total/1e6:.1f}MB")
                    sys.stdout.flush()
        print()
        import tarfile
        # tar 内多为同名子目录（<MODEL_ID>/...），解到 DEST.parent 即正好落到 DEST
        with tarfile.open(str(tmp), "r:bz2") as t:
            os.makedirs(DEST.parent, exist_ok=True)
            t.extractall(DEST.parent)
        tmp.unlink()
        return (DEST / "tokens.txt").exists()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] GitHub tar 下载/解压失败: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="仅装必需文件（本来就只装必需）")
    args = ap.parse_args()

    # 首选：GitHub release 单文件 tar（官方分发方式）
    if try_github_tar():
        _summarize(_FILES)
        return

    files = discover_files()
    print(f"目标目录: {DEST}")
    print("待下载文件: " + ", ".join(sorted(set(files.values()))))
    os.makedirs(DEST, exist_ok=True)

    # 次选：ModelScope git clone（若有）
    if shutil.which("git"):
        clone_url = f"https://www.modelscope.cn/{_REPO}.git"
        print(f"[1] 尝试 ModelScope git clone: {clone_url}")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(DEST)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and (DEST / "tokens.txt").exists():
            print("    [OK] 完成")
            _summarize(files); return
        else:
            print("    [FAIL]", r.stderr.strip()[:200] or "未托管于 ModelScope")

    # 逐个源下载必需文件
    fnames = list(dict.fromkeys(files.values()))   # 去重保序
    missing = list(fnames)
    for name, fmt in _SOURCES:
        todo = [f for f in missing]
        if not todo:
            break
        print(f"[2] 尝试 {name}")
        for fname in todo:
            url = fmt.format(repo=_REPO, file=fname)
            if download(url, DEST / fname):
                missing = [f for f in missing if f != fname]
                print(f"  [OK] {fname}")
            else:
                print(f"  [FAIL] {fname}")

    # 兜底：ModelScope SDK/文件下载
    if missing:
        print("[3] 尝试 ModelScope 文件下载")
        if try_modelscope():
            missing = []
        else:
            for fname in list(missing):
                if (DEST / fname).exists():
                    missing = [f for f in missing if f != fname]

    if missing:
        print("\n仍有文件缺失:", missing)
        print("请在可达外网的机器上运行本脚本，或手动放入模型，再拷回本机。")
        sys.exit(1)

    _summarize(files)


def _summarize(files: dict):
    print("\n模型就绪：")
    for f in dict.fromkeys(files.values()):
        p = DEST / f
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f}MB" if p.exists() else f"  {f}: 缺失")
    print("重启后端即可启用关键词唤醒（server/kws.py）。")


if __name__ == "__main__":
    main()
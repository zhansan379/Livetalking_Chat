#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把本地 model/形象/动作 打成一个 release zip，方便上传到 GitHub Releases，
让克隆者跑 download_models.py 即可恢复运行时数据，"直接运行项目"。

内容：models/ + data/（含 avatars 形象、actions 动作、capabilities 面试库、music）
说明：这些大文件不提交进 git（.gitignore 已忽略 models/ 与 data/），
      而是随 GitHub Release 附件分发，避开 git 100MB 单文件硬限与 LFS 配额。

产物：dist/livetalking-assets-<version>.zip
用法：
    python scripts/package_release.py                     # 自动取 git 最近 tag 或 HEAD
    python scripts/package_release.py --version v1.0.0    # 指定版本号
    python scripts/package_release.py --sources models data  # 自定义打包目录

上传：在 GitHub 仓库 Release 页面新建 release，把 dist/*.zip 作为附件上传，
      然后用 scripts/download_models.py 拉取。
"""
import argparse
import os
import sys
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
# 默认打包的全部运行时数据（用户已确认"全部打包"）
DEFAULT_SOURCES = ["models", "data"]


def current_version():
    """取 git 最近 tag；无 tag 则用 HEAD 短哈希凑一个占位版本号。"""
    try:
        import subprocess
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        if tag:
            return tag
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"git-{short}"
    except Exception:
        return "latest"


def main():
    ap = argparse.ArgumentParser(description="打包 models/ 与 data/ 为 GitHub Release 附件")
    ap.add_argument("--version", default=current_version(),
                    help="zip 名里的版本号（默认取最近 git tag）")
    ap.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES,
                    help=f"要打包的顶层目录（默认: {' '.join(DEFAULT_SOURCES)}）")
    ap.add_argument("--fast", action="store_true",
                    help="直存不压缩（默认）。图片/模型大多已压缩，节省打包时间")
    args = ap.parse_args()

    os.makedirs(DIST, exist_ok=True)
    name = f"livetalking-assets-{args.version}"
    out = os.path.join(DIST, name + ".zip")

    compression = zipfile.ZIP_STORED if args.fast else zipfile.ZIP_DEFLATED
    print(f"[package] 写入 {out}")
    print(f"[package] 打包目录: {', '.join(args.sources)}")

    total = 0
    with zipfile.ZipFile(out, "w", compression=compression,
                         allowZip64=True) as zf:
        for src in args.sources:
            base = os.path.join(ROOT, src)
            if not os.path.isdir(base):
                print(f"[package] 跳过不存在目录: {src}", file=sys.stderr)
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                # 跳过缓存/临时目录，避免混入 __pycache__、.DS_Store 等
                dirnames[:] = [d for d in dirnames
                               if d not in ("__pycache__", ".pytest_cache")]
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, ROOT).replace(os.sep, "/")
                    zf.write(full, arc)
                    total += os.path.getsize(full)

    size = os.path.getsize(out)
    print(f"[package] 完成: {out}")
    print(f"[package]   源数据 {total / 1048576:.1f} MB -> zip {size / 1048576:.1f} MB")
    print("[package] 下一步：在 GitHub Release 页把该 zip 作为附件上传，")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 GitHub Releases 拉取模型/形象/动作 zip 并解压到仓库根，恢复运行时数据。

配合 scripts/package_release.py 使用：先运行 package_release.py 打 zip 并作为
Release 附件上传，运行者再执行本脚本即可一键补齐 models/ 与 data/，直接 run.bat。

纯 Python 标准库实现，无需额外依赖。公开仓库无需 token；私有仓库可传 --token。
用法：
    python scripts/download_models.py                        # 拉最新版本，解压到仓库根
    python scripts/download_models.py --version v1.0.0       # 指定 release tag
    python scripts/download_models.py --list                 # 仅列出可用版本，不下载
"""
import argparse
import json
import os
import sys
import urllib.request
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = os.environ.get("LIVETALKING_REPO", "zhansan379/Livetalking_Chat")


def _request(url, token=None):
    """带 User-Agent（GitHub API 要求）；私有仓叠加 token。返回 response。"""
    headers = {"User-Agent": "livetalking-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req)


def list_releases(repo, token=None):
    url = f"https://api.github.com/repos/{repo}/releases"
    with _request(url, token) as r:
        return json.load(r)


def pick_asset(release, name):
    """从 release 里挑出名字以 name 开头的 zip 资产。"""
    for a in release.get("assets", []):
        if a["name"].startswith(name) and a["name"].endswith(".zip"):
            return a
    # 老版本不一定有 assets 字段，回退到源码 tarball
    raise SystemExit(
        f"[dl] 在 {release.get('tag_name')} 里没找到 {name}*.zip；"
        "请先运行 scripts/package_release.py 打包并作为 Release 附件上传。"
    )


def download(url, dest, token=None):
    """流式下载并显示进度条。"""
    with _request(url, token) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    sys.stdout.write(f"\r[dl] {done / 1048576:.1f}/{total / 1048576:.1f} MB ({pct}%)")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    print(f"[dl] 已下载: {dest}")


def main():
    ap = argparse.ArgumentParser(description="从 GitHub Releases 拉取并解压模型/形象/动作数据")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub 仓库，如 owner/name")
    ap.add_argument("--version", default="latest",
                    help="release tag；默认 latest")
    ap.add_argument("--asset", default="livetalking-assets",
                    help="zip 资产名前缀")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                    help="GitHub token（私有仓库必需）")
    ap.add_argument("--dest", default=ROOT, help="解压目标目录（默认仓库根）")
    ap.add_argument("--list", action="store_true", help="仅列出可用版本并退出")
    args = ap.parse_args()

    print(f"[dl] 查询 {args.repo} releases ...")
    releases = list_releases(args.repo, args.token)
    if args.list:
        for rel in releases:
            tags = ", ".join(a["name"] for a in rel.get("assets", []))
            print(f"  {rel['tag_name']:<20} {tags}")
        return

    release = releases[0] if args.version == "latest" else next(
        (r for r in releases if r["tag_name"] == args.version), None)
    if not release:
        raise SystemExit(f"[dl] 找不到版本 {args.version}")

    asset = pick_asset(release, args.asset)
    print(f"[dl] 版本 {release['tag_name']}: {asset['name']} "
          f"({asset['size'] / 1048576:.1f} MB)")
    if not asset["name"].endswith(".zip"):
        raise SystemExit("[dl] 资产不是 zip，须用 package_release.py 生成的包")

    tmp = os.path.join(args.dest, os.path.basename(asset["name"]))
    if not (os.path.exists(tmp) and os.path.getsize(tmp) == asset["size"]):
        download(asset["browser_download_url"], tmp, args.token)
    else:
        print(f"[dl] 本地已存在完整文件，跳过下载: {tmp}")

    print(f"[dl] 解压到 {args.dest} ...（这将创建/覆盖 models/ 与 data/）")
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(args.dest)
    print("[dl] 完成。现在可以运行 run.bat 启动，或 ./run.bat 。")
    print("[dl] 注意：仍需在 .env 配置你自己的 TENCENT/DASHSCOPE/DOUBAO 密钥。")


if __name__ == "__main__":
    main()
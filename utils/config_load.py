"""
通用配置加载工具：从 asr/config_loader.py 抽出的可复用部分。

- `resolve_env(obj)`：递归解析 dict/list/str 里的 `${ENV}` / `${VAR:-default}` 占位。
- `load_yaml(path)`：读取并 safe_load 一个 YAML 文件。

`load_dotenv` 逻辑保留在各 loader 里（属各自模块的单例初始化），此处只放无侵入的共用函数。
"""

import os
from pathlib import Path
from typing import Any

import yaml


def resolve_env(obj: Any) -> Any:
    """递归解析 dict/list/str 中的 `${ENV}` 占位符（支持 `${VAR:-default}` 嵌套回退）。"""
    if isinstance(obj, dict):
        return {k: resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env(i) for i in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        inner = obj[2:-1]
        if ":-" in inner:
            var, _, default = inner.partition(":-")
            default = resolve_env(default) if default else ""
            return os.environ.get(var.strip(), default)
        return os.environ.get(inner.strip(), "")
    return obj


def load_yaml(path: str | Path) -> dict:
    """读取并解析一个 YAML 配置文件，返回 dict（空文件 → {}）。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
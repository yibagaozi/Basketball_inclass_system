"""Configuration loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"


@lru_cache(maxsize=1)
def load_models_config() -> dict[str, Any]:
    """进程内只解析一次。配置改动需重启进程才生效。"""
    path = CONFIGS / "models.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_path(relative: str) -> Path:
    return ROOT / relative


@lru_cache(maxsize=32)
def load_yaml(name: str) -> dict[str, Any]:
    """进程内按文件名缓存。返回的 dict 是共享对象,调用方不要就地修改。

    热路径上 _identity_cfg() 等每帧都会调用本函数,不缓存的话每次都是
    一次磁盘读 + PyYAML 纯 Python 解析,在 GIL 下会成为主要瓶颈。
    配置改动需重启进程才生效;测试里改配置后调用 load_yaml.cache_clear()。
    """
    path = CONFIGS / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(*parts: str) -> Path:
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

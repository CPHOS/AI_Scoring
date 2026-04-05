"""从 YAML 资源文件加载 prompt 定义."""

from __future__ import annotations

import functools
from importlib.resources import files
from typing import Any

import yaml


@functools.cache
def load_prompt(name: str) -> dict[str, Any]:
    """按名称（不含扩展名）加载 prompt YAML 文件并缓存。"""
    resource = files("src.prompts").joinpath(f"{name}.yaml")
    text = resource.read_text("utf-8")
    return yaml.safe_load(text)

"""公用数据类型."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputAsset:
    """单张输入图片/页面的 base64 编码资产."""

    source_path: str
    source_index: int
    page_index: int
    media_type: str
    filename: str
    base64_data: str


@dataclass
class TranscriptionResult:
    """单次 VLM 转录的结果."""

    request_id: str
    transcription: str
    source_files: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = "openrouter"
    usage: dict[str, Any] = field(default_factory=dict)

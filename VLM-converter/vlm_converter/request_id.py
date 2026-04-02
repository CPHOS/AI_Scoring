from __future__ import annotations

import re
from pathlib import Path


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def build_request_id(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("At least one input path is required")

    normalized = []
    for path in sorted(paths, key=lambda item: item.name.lower()):
        stem = path.stem.lower()
        cleaned = _NON_ALNUM_RE.sub("-", stem).strip("-")
        normalized.append(cleaned or "input")

    collapsed = "__".join(normalized)
    if len(collapsed) <= 96:
        return collapsed

    return collapsed[:96].rstrip("-")

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_UPPI_RE = re.compile(r"\\uppi\b")
_TEXT_UNIT_RE = re.compile(r"\$([^$]*?)\\text\{([^{}]+)\}([^$]*?)\$")


def normalize_transcription(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    normalized = _UPPI_RE.sub(r"\\pi", normalized)
    normalized = _normalize_unit_spacing(normalized)
    return normalized.strip()


def _normalize_unit_spacing(text: str) -> str:
    return _TEXT_UNIT_RE.sub(lambda match: f"${match.group(1)}\\text{{{match.group(2).strip()}}}{match.group(3)}$", text)

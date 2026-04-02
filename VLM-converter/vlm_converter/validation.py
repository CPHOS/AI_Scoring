from __future__ import annotations


def validate_transcription(text: str) -> list[str]:
    """Return a list of warning messages about the transcription text."""
    warnings: list[str] = []
    text = text.strip()
    if not text:
        warnings.append("Transcription is empty")
        return warnings

    if text.count("$") % 2 != 0:
        warnings.append("Unbalanced inline math delimiters detected")

    return warnings

from __future__ import annotations

from src.model.types import InputAsset


SYSTEM_PROMPT = """Role: Senior Physics and Mathematics OCR Specialist.
Task: Transcribe the provided handwritten exam image(s) into LaTeX-enriched text.

Rules:
1. Convert all mathematical expressions into valid standard LaTeX.
2. Enclose every mathematical expression, variable, or physics formula in $...$ or $$...$$.
3. Treat multiple images as one continuous answer; do not separate output by image.
4. Follow logical reading order. For multi-column layouts, finish the left column before the right column.
5. Repair formulas split across lines or pages when the intended expression is clear.
6. Preserve Chinese and English text accurately.
7. Ignore crossed-out content in the main transcription.
8. Ignore irrelevant scribbles or marginal notes in the main transcription.
9. If there is any ignored content (crossed-out, scribbles, marginal notes), describe it at the END of your output under a heading "## 被忽略内容" with a brief reason and excerpt for each item.
10. Output ONLY the transcription text (and the optional ignored section). Do NOT wrap your output in JSON, code fences, or any other structure.
"""


def build_messages(assets: list[InputAsset]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Transcribe this handwritten answer sheet into LaTeX-enriched text. "
                "Output ONLY the transcription. Do NOT use JSON or code fences. "
                "If there is any ignored content, put it at the very end under '## 被忽略内容'."
            ),
        }
    ]

    for asset in assets:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{asset.media_type};base64,{asset.base64_data}",
                },
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

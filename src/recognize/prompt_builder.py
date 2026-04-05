from __future__ import annotations

from src.model.types import InputAsset
from src.prompt import load_prompt


def build_messages(assets: list[InputAsset]) -> list[dict[str, object]]:
    prompts = load_prompt("recognize")

    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": prompts["user"],
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
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": content},
    ]

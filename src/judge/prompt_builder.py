"""构建 judge 模块的 System / User prompt（同时支持文本和直接 VLM 评分）."""

from __future__ import annotations

from src.model.types import InputAsset
from src.prompt import load_prompt

from .types import ScoringItem, ScoringRubric


def build_judge_messages(
    rubric: ScoringRubric,
    student_text: str | None = None,
    *,
    assets: list[InputAsset] | None = None,
    verbosity: int = 1,
    strictness: int = 1,
) -> list[dict[str, object]]:
    """Construct the messages list for the OpenRouter chat completion call.

    当 *assets* 非空且 *student_text* 为 None 时进入 **direct 模式**
    （VLM 直接读取图片评分），否则为文本评分模式。
    """
    direct_mode = assets is not None and student_text is None
    prompts = load_prompt("judge")
    system_prompt = prompts["direct_system"] if direct_mode else prompts["system"]

    user_text = _build_user_prompt(rubric, student_text, verbosity, strictness, direct_mode, prompts)

    if direct_mode:
        content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def _build_user_prompt(
    rubric: ScoringRubric,
    student_text: str | None,
    verbosity: int,
    strictness: int,
    direct_mode: bool,
    prompts: dict[str, object],
) -> str:
    sections: list[str] = []

    # ── 0. 给分策略（strictness） ──
    strictness_rules = prompts.get("strictness_rules", {})
    rule_text = strictness_rules.get(strictness, strictness_rules.get(1, ""))
    if rule_text:
        sections.append(rule_text.strip())

    # ── 1. 题目描述 ──
    if rubric.problem_statement:
        sections.append(
            f"## 题目描述\n\n{rubric.problem_title}\n\n{rubric.problem_statement}"
        )

    # ── 2. 评分标准 ──
    primary_items = [it for it in rubric.items if not it.alternative]
    alt_items = [it for it in rubric.items if it.alternative]

    rubric_lines = ["## 评分标准（主解法）", ""]
    rubric_lines.append(f"本题满分 {rubric.total_score} 分。逐项评分如下：\n")
    for item in primary_items:
        rubric_lines.append(_format_scoring_item(item))

    if alt_items:
        rubric_lines.append("")
        rubric_lines.append("## 替代解法评分标准")
        rubric_lines.append("")
        rubric_lines.append(
            "学生可能采用以下替代解法，如果学生使用替代解法，"
            "请按替代解法标准给分（将替代解法的评分点映射到主解法同位项）。"
        )
        rubric_lines.append("")
        for item in alt_items:
            rubric_lines.append(_format_scoring_item(item))

    sections.append("\n".join(rubric_lines))

    # ── 3. 标准解答参考 ──
    if rubric.solution_text:
        sections.append(f"## 标准解答（参考）\n\n{rubric.solution_text}")

    # ── 4. 学生答案（模式依赖） ──
    if direct_mode:
        sections.append(prompts["student_images_section"].strip())
    else:
        sections.append(f"## 学生答案\n\n{student_text}")

    # ── 5. 输出要求 ──
    output_spec = _build_output_spec(primary_items, verbosity)
    sections.append(output_spec)

    return "\n\n---\n\n".join(sections)


def _format_scoring_item(item: ScoringItem) -> str:
    type_label = "方程" if item.item_type == "equation" else "文字/讨论"
    sol_note = f" (解法{item.solution_index + 1})" if item.alternative else ""
    line = (
        f"- **{item.item_id}** [{type_label}] "
        f"(满分 {item.max_score} 分){sol_note}："
    )
    if item.context:
        line += f" {item.context}"
    return line


def _build_output_spec(
    primary_items: list[ScoringItem], verbosity: int
) -> str:
    item_ids = [it.item_id for it in primary_items]
    ids_str = ", ".join(f'"{i}"' for i in item_ids)

    lines = ["## 输出要求", ""]
    lines.append("请输出一个 JSON 对象，格式如下：")
    lines.append("```json")
    lines.append("{")
    lines.append(f'  "scores": {{')

    example_id = item_ids[0] if item_ids else "1"
    if verbosity == 0:
        lines.append(f'    "{example_id}": <int>,')
        lines.append(f'    ...')
        lines.append(f'  }}')
    else:
        reasoning_desc = "简要一句话说明" if verbosity == 1 else "详细推理过程"
        lines.append(f'    "{example_id}": {{"score": <int>, "reasoning": "{reasoning_desc}"}},')
        lines.append(f'    ...')
        lines.append(f'  }}')

    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append(f"需要评分的 item_id 列表: [{ids_str}]")
    lines.append("")
    lines.append("注意：")
    lines.append("- 每项分数为 0 到该项满分之间的整数")
    lines.append("- 输出纯 JSON，不要包含任何其他文字或 markdown 标记")
    lines.append("- 不要用 ```json 等代码块包裹，直接输出 JSON 对象")

    if verbosity == 2:
        lines.append("- reasoning 字段请保持在 2-3 句话以内")

    return "\n".join(lines)

"""构建 judge 模块的 System / User prompt."""

from __future__ import annotations

from .types import ScoringItem, ScoringRubric


_SYSTEM_PROMPT = """\
Role: 你是一位经验丰富的物理竞赛阅卷专家。

Task: 根据提供的评分标准，对学生手写答卷的转录文本进行逐项评分。

Rules:
1. 严格按照评分标准中的每一个评分点逐项给分。
2. 每个评分点的得分为 0 到该项满分之间的整数，允许部分给分。
3. 评分只依据学生答案的物理和数学内容是否正确，不考虑书写整洁度。
4. 如果学生使用了与标准答案不同但等价正确的方法，应当给分。
5. 如果标准答案提供了多种解法，识别学生使用的解法并按对应标准给分。
6. 对于方程类评分点：学生写出的方程在物理含义和数学形式上与标准一致即可给分，\
不要求完全一致的符号或排列，允许等价变形。
7. 对于文字/讨论类评分点：学生给出了正确的物理解释或结论即可给分。
8. 输出必须严格为 JSON 格式，不要添加任何额外文字。
"""


def build_judge_messages(
    rubric: ScoringRubric,
    student_text: str,
    verbosity: int = 1,
) -> list[dict[str, object]]:
    """Construct the messages list for the OpenRouter chat completion call."""
    user_text = _build_user_prompt(rubric, student_text, verbosity)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _build_user_prompt(
    rubric: ScoringRubric,
    student_text: str,
    verbosity: int,
) -> str:
    sections: list[str] = []

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

    # ── 4. 学生答案 ──
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

"""解析 .tex 标准答案文件中的评分结构."""

from __future__ import annotations

import re
from pathlib import Path

from .types import ScoringItem, ScoringRubric


# ── regex patterns ──────────────────────────────────────────────────────────

_PROBLEM_RE = re.compile(
    r"\\begin\{problem\}\s*\[(\d+)\]\s*\{([^}]*)\}", re.DOTALL
)
_PROBLEM_STATEMENT_RE = re.compile(
    r"\\begin\{problemstatement\}(.*?)\\end\{problemstatement\}", re.DOTALL
)
_SOLUTION_RE = re.compile(
    r"\\begin\{solution\}(.*?)\\end\{solution\}", re.DOTALL
)
_MULTISOL_RE = re.compile(
    r"\\begin\{multisol\}(.*?)\\end\{multisol\}", re.DOTALL
)

_SOLSUBQ_RE = re.compile(r"\\solsubq\{([^}]*)\}\{(\d+)\}")
_SOLSUBSUBQ_RE = re.compile(r"\\solsubsubq\{([^}]*)\}\{(\d+)\}")
_EQTAGSCORE_RE = re.compile(r"\\eqtagscore\{([^}]*)\}\{(\d+)\}")
_EQTAG_RE = re.compile(r"\\eqtag\{([^}]*)\}")
_ADDTEXT_RE = re.compile(r"\\addtext\{([^}]*)\}\{(\d+)\}")

_EQUATION_ENV_RE = re.compile(
    r"\\begin\{equation\}(.*?)\\end\{equation\}", re.DOTALL
)


def parse_scoring_rubric(tex_path: Path) -> ScoringRubric:
    """Parse a .tex file and extract the scoring rubric."""
    text = tex_path.read_text(encoding="utf-8")

    problem_title, total_score = _extract_problem_header(text)
    problem_statement = _extract_environment(_PROBLEM_STATEMENT_RE, text)
    solution_text = _extract_environment(_SOLUTION_RE, text)

    items = _extract_scoring_items(solution_text)

    return ScoringRubric(
        problem_title=problem_title,
        total_score=total_score,
        problem_statement=problem_statement,
        solution_text=solution_text,
        items=items,
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _extract_problem_header(text: str) -> tuple[str, int]:
    match = _PROBLEM_RE.search(text)
    if not match:
        raise ValueError("Cannot find \\begin{problem}[score]{title} in .tex file")
    return match.group(2).strip(), int(match.group(1))


def _extract_environment(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_scoring_items(solution_text: str) -> list[ScoringItem]:
    """Walk through the solution text and extract all scoring items."""
    # First, handle multisol regions: replace them with a placeholder then
    # process them separately so that items outside multisol are handled
    # normally.
    multisol_blocks: list[str] = []
    PLACEHOLDER = "\x00MULTISOL_{}\x00"

    def _capture_multisol(m: re.Match[str]) -> str:
        idx = len(multisol_blocks)
        multisol_blocks.append(m.group(1))
        return PLACEHOLDER.format(idx)

    working_text = _MULTISOL_RE.sub(_capture_multisol, solution_text)

    items: list[ScoringItem] = []
    current_subq = ""
    addtext_counter: dict[str, int] = {}  # label -> sequential counter

    # Tokenize the working text by scanning for relevant commands in order
    _TOKEN_RE = re.compile(
        r"(?P<solsubsubq>\\solsubsubq\{(?P<sssq_id>[^}]*)\}\{(?P<sssq_score>\d+)\})"
        r"|(?P<solsubq>\\solsubq\{(?P<ssq_id>[^}]*)\}\{(?P<ssq_score>\d+)\})"
        r"|(?P<multisol_ph>\x00MULTISOL_(?P<ms_idx>\d+)\x00)"
        r"|(?P<eqtagscore>\\eqtagscore\{(?P<ets_id>[^}]*)\}\{(?P<ets_score>\d+)\})"
        r"|(?P<addtext>\\addtext\{(?P<at_label>[^}]*)\}\{(?P<at_score>\d+)\})"
        r"|(?P<equation>\\begin\{equation\})"
    )

    for token in _TOKEN_RE.finditer(working_text):
        if token.group("solsubq"):
            current_subq = token.group("ssq_id")
        elif token.group("solsubsubq"):
            current_subq = token.group("sssq_id")
        elif token.group("multisol_ph"):
            ms_idx = int(token.group("ms_idx"))
            ms_items = _parse_multisol_block(
                multisol_blocks[ms_idx], current_subq, addtext_counter
            )
            items.extend(ms_items)
        elif token.group("eqtagscore"):
            item_id = token.group("ets_id")
            score = int(token.group("ets_score"))
            context = _find_enclosing_equation(working_text, token.start())
            items.append(ScoringItem(
                item_type="equation",
                item_id=item_id,
                max_score=score,
                context=context,
                sub_question=current_subq,
            ))
        elif token.group("addtext"):
            label = token.group("at_label")
            score = int(token.group("at_score"))
            addtext_counter[label] = addtext_counter.get(label, 0) + 1
            item_id = f"{label}_{addtext_counter[label]}"
            context = _find_preceding_text(working_text, token.start())
            items.append(ScoringItem(
                item_type="text",
                item_id=item_id,
                max_score=score,
                context=context,
                sub_question=current_subq,
            ))

    return items


def _parse_multisol_block(
    block_text: str, sub_question: str, addtext_counter: dict[str, int]
) -> list[ScoringItem]:
    """Parse a multisol block into scoring items from multiple solutions."""
    # Split by \item — the first segment before any \item is preamble (ignored)
    parts = re.split(r"\\item\b", block_text)
    # parts[0] is text before the first \item (usually empty), skip it
    solution_parts = parts[1:] if len(parts) > 1 else []

    items: list[ScoringItem] = []

    for sol_idx, part in enumerate(solution_parts):
        is_alternative = sol_idx > 0

        for eq_match in _EQUATION_ENV_RE.finditer(part):
            eq_body = eq_match.group(1)
            for es_match in _EQTAGSCORE_RE.finditer(eq_body):
                item_id = es_match.group(1)
                score = int(es_match.group(2))
                context = _clean_equation(eq_body)
                items.append(ScoringItem(
                    item_type="equation",
                    item_id=item_id,
                    max_score=score,
                    context=context,
                    sub_question=sub_question,
                    alternative=is_alternative,
                    solution_index=sol_idx,
                ))

        for at_match in _ADDTEXT_RE.finditer(part):
            label = at_match.group(1)
            score = int(at_match.group(2))
            addtext_counter[label] = addtext_counter.get(label, 0) + 1
            item_id = f"{label}_{addtext_counter[label]}"
            context = _find_preceding_text(part, at_match.start())
            items.append(ScoringItem(
                item_type="text",
                item_id=item_id,
                max_score=score,
                context=context,
                sub_question=sub_question,
                alternative=is_alternative,
                solution_index=sol_idx,
            ))

    return items


def _find_enclosing_equation(text: str, pos: int) -> str:
    """Find the equation environment enclosing the position *pos*."""
    for m in _EQUATION_ENV_RE.finditer(text):
        if m.start() <= pos <= m.end():
            return _clean_equation(m.group(1))
    return ""


def _clean_equation(raw: str) -> str:
    """Strip scoring tags and LaTeX comments from equation text."""
    cleaned = _EQTAGSCORE_RE.sub("", raw)
    cleaned = _EQTAG_RE.sub("", cleaned)
    # Remove LaTeX line comments (% to end of line)
    cleaned = re.sub(r"%[^\n]*", "", cleaned)
    return cleaned.strip()


def _find_preceding_text(text: str, pos: int) -> str:
    """Extract a meaningful context snippet before *pos*."""
    # Take up to 300 chars before the position
    start = max(0, pos - 300)
    segment = text[start:pos]
    # Find the last sentence boundary or paragraph break
    lines = segment.split("\n")
    # Return the last non-empty line(s) as context
    meaningful: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if meaningful:
                break
            continue
        # Skip lines that are just LaTeX commands
        if stripped.startswith("\\solsubq") or stripped.startswith("\\solsubsubq"):
            continue
        if stripped.startswith("\\begin{") or stripped.startswith("\\end{"):
            continue
        meaningful.append(stripped)
        if len(meaningful) >= 3:
            break
    meaningful.reverse()
    return " ".join(meaningful)

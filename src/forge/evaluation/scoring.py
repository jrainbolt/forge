"""Transparent deterministic scoring for evaluation tasks."""

from __future__ import annotations

import re

from forge.evaluation.types import (
    DimensionScore,
    EvaluationTask,
    TaskScores,
    ToolRecord,
)


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def score_task(
    task: EvaluationTask,
    answer: str,
    files_inspected: tuple[str, ...],
    tools: tuple[ToolRecord, ...],
    *,
    completed: bool,
) -> TaskScores:
    """Score one task using only declared ground truth and observed metadata."""
    normalized = normalize(answer)
    facts = sum(
        any(normalize(alternative) in normalized for alternative in fact.alternatives)
        for fact in task.required_facts
    )
    inspected = set(files_inspected)
    grounding = sum(path in inspected for path in task.required_files)
    location_targets = (*task.expected_answer_files, *task.expected_symbols)
    localized = sum(normalize(target) in normalized for target in location_targets)
    return TaskScores(
        correctness=DimensionScore(facts, len(task.required_facts)),
        grounding=DimensionScore(grounding, len(task.required_files)),
        localization=DimensionScore(localized, len(location_targets)),
        efficiency=DimensionScore(int(len(tools) <= task.max_tool_calls), 1),
        completion=DimensionScore(int(completed), 1),
    )


def answer_mentions_file(answer: str, path: str) -> bool:
    """Public helper used by tests and report diagnostics."""
    return normalize(path) in normalize(answer)


def answer_words(answer: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", answer.casefold()))

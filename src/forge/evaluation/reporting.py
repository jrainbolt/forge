"""Human- and machine-readable evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from forge.evaluation.types import EvaluationRun, TaskResult


def render_terminal_report(run: EvaluationRun, *, verbose: bool = False) -> str:
    lines = [
        f"Forge Evaluation: {run.suite}",
        f"Model: {run.model_profile}",
        "",
    ]
    for result in run.task_results:
        label = (
            "PASS"
            if result.success and result.scores.total == result.scores.maximum
            else "PARTIAL"
            if result.success
            else "FAIL"
        )
        lines.append(
            f"{result.task_id}  {label:<7} "
            f"{result.scores.total}/{result.scores.maximum}  "
            f"{result.tool_count} tools"
        )
        if verbose:
            lines.append(f"  files: {', '.join(result.files_inspected) or '-'}")
            lines.append(f"  answer: {result.answer}")
            if result.failure_reason is not None:
                lines.append(f"  reason: {result.failure_reason.value}")
    lines.extend(
        (
            "",
            f"Total: {run.score}/{run.maximum_score}",
            f"Completed: {run.completed}/{len(run.task_results)}",
            f"Tool calls: {run.tool_calls}",
            f"Protocol corrections: {run.protocol_corrections}",
            f"Elapsed: {run.elapsed_seconds:.1f}s",
            f"Input tokens: {_optional(run.usage.input_tokens)}",
            f"Output tokens: {_optional(run.usage.output_tokens)}",
        )
    )
    return "\n".join(lines)


def write_json_report(run: EvaluationRun, path: Path) -> None:
    """Write a report only to the caller's explicit path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_to_dict(run: EvaluationRun) -> dict[str, object]:
    return {
        "schema_version": run.schema_version,
        "suite": run.suite,
        "suite_version": run.suite_version,
        "model_profile": run.model_profile,
        "elapsed_seconds": run.elapsed_seconds,
        "summary": {
            "tasks": len(run.task_results),
            "completed": run.completed,
            "failed": run.failed,
            "score": run.score,
            "maximum_score": run.maximum_score,
            "tool_calls": run.tool_calls,
            "protocol_corrections": run.protocol_corrections,
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
        },
        "tasks": [_task_to_dict(result) for result in run.task_results],
    }


def _task_to_dict(result: TaskResult) -> dict[str, object]:
    return {
        "task_id": result.task_id,
        "category": result.category.value,
        "success": result.success,
        "answer": result.answer,
        "files_inspected": list(result.files_inspected),
        "tools": [
            {
                "name": tool.name,
                "status": tool.status,
                "evidence": tool.evidence,
                "path": tool.path,
                "returned_bytes": tool.returned_bytes,
                "returned_lines": tool.returned_lines,
            }
            for tool in result.tools
        ],
        "tool_count": result.tool_count,
        "orchestration_steps": result.orchestration_steps,
        "protocol_corrections": result.protocol_corrections,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "elapsed_seconds": result.elapsed_seconds,
        "scores": {
            name: {"earned": score.earned, "maximum": score.maximum}
            for name, score in (
                ("correctness", result.scores.correctness),
                ("grounding", result.scores.grounding),
                ("localization", result.scores.localization),
                ("efficiency", result.scores.efficiency),
                ("completion", result.scores.completion),
            )
        },
        "total_score": result.scores.total,
        "maximum_score": result.scores.maximum,
        "failure_reason": (
            result.failure_reason.value if result.failure_reason is not None else None
        ),
        "failure_message": result.failure_message,
    }


def _optional(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"

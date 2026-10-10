"""Production structural-completeness enforcement boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StructuralCompletenessMode(Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class StructuralCompletenessDecision(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class StructuralCompletenessEvaluation:
    decision: StructuralCompletenessDecision
    failed_check_ids: tuple[str, ...] = ()
    operational_failures: tuple[str, ...] = ()
    latency_seconds: float = 0.0
    subprocess_count: int = 0

    def __post_init__(self) -> None:
        if (
            self.decision is not StructuralCompletenessDecision.FAIL
            and self.failed_check_ids
        ):
            raise ValueError(
                "only proven structural failure may carry failed check IDs"
            )
        if self.latency_seconds < 0 or self.subprocess_count < 0:
            raise ValueError("structural evaluation metrics must be non-negative")
        object.__setattr__(self, "failed_check_ids", tuple(self.failed_check_ids))
        object.__setattr__(
            self, "operational_failures", tuple(self.operational_failures)
        )


@dataclass(frozen=True, slots=True)
class StructuralCompletenessRecord:
    mode: str = StructuralCompletenessMode.SHADOW.value
    executed: bool = False
    decision: str = "not_run"
    failed_check_ids: tuple[str, ...] = ()
    operational_failures: tuple[str, ...] = ()
    latency_seconds: float = 0.0
    subprocess_count: int = 0
    enforced_rejection: bool = False


StructuralCompletenessEvaluator = Callable[[Path], StructuralCompletenessEvaluation]


def evaluate_structural_completeness(
    mode: StructuralCompletenessMode,
    workspace: Path,
    evaluator: StructuralCompletenessEvaluator | None,
) -> StructuralCompletenessRecord:
    """Execute one contained post-transaction structural observation."""
    if mode is StructuralCompletenessMode.OFF:
        return StructuralCompletenessRecord(mode=mode.value)
    if evaluator is None:
        return StructuralCompletenessRecord(
            mode=mode.value,
            executed=True,
            decision=StructuralCompletenessDecision.NOT_APPLICABLE.value,
        )
    started = time.perf_counter()
    try:
        result = evaluator(workspace)
        if not isinstance(result, StructuralCompletenessEvaluation):
            raise TypeError("structural evaluator returned an invalid result")
    except Exception:
        return StructuralCompletenessRecord(
            mode=mode.value,
            executed=True,
            decision=StructuralCompletenessDecision.PARTIAL.value,
            operational_failures=("CHECKER_INTERNAL_ERROR",),
            latency_seconds=time.perf_counter() - started,
        )
    latency = max(result.latency_seconds, time.perf_counter() - started)
    rejected = (
        mode is StructuralCompletenessMode.ENFORCE
        and result.decision is StructuralCompletenessDecision.FAIL
        and not result.operational_failures
    )
    return StructuralCompletenessRecord(
        mode=mode.value,
        executed=True,
        decision=result.decision.value,
        failed_check_ids=result.failed_check_ids,
        operational_failures=result.operational_failures,
        latency_seconds=latency,
        subprocess_count=result.subprocess_count,
        enforced_rejection=rejected,
    )

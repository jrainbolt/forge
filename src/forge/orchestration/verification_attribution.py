"""Conservative, model-free comparison of trusted project verification runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from forge.tools.project import PreparedProjectCommand, project_environment

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_SUMMARY_CASE = re.compile(r"^\s*\d+\s+-\s+.+\((?:Failed|Not Run|Timeout)\)\s*$", re.I)
_FAILURE_LINE = re.compile(
    r"(?:\bFAILED\b|\bFAILURE\b|\bAssertionError\b|\berror:|"
    r"\bUnable to find executable:|\bnot found\b|\bfailed\b)",
    re.I,
)
_MAX_LINES = 128
_MAX_LINE_CHARS = 240


class AttributionResult(Enum):
    NOT_APPLICABLE = "not_applicable"
    NO_BASELINE = "no_baseline"
    MUTATION_ASSOCIATED = "mutation_associated"
    PREEXISTING_OR_UNRELATED = "preexisting_or_unrelated"
    UNATTRIBUTED = "unattributed"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    COMMAND_MISMATCH = "command_mismatch"


@dataclass(frozen=True, slots=True)
class VerificationBaselineEvidence:
    """A caller-owned A10 execution, bound to one workspace and generation."""

    workspace: Path
    generation: int
    command_identity: str
    result: str
    failure_fingerprint: str | None
    duration_seconds: float
    executed: bool


@dataclass(frozen=True, slots=True)
class VerificationPlanBaseline:
    plan_id: str
    operations: tuple[str, ...]
    steps: tuple[VerificationBaselineEvidence, ...]


def attribute_plan_failure(
    baseline: VerificationPlanBaseline | None,
    plan_id: str,
    operations: tuple[str, ...],
    successful_prior_steps: int,
    prepared: PreparedProjectCommand | None,
    status: str,
    output: Mapping[str, object] | None,
    pre_mutation_generation: int,
) -> VerificationAttribution:
    """Compare only the identical failed step after identical passing prerequisites."""
    if baseline is None:
        return attribute_failure(
            None, prepared, status, output, pre_mutation_generation
        )
    if baseline.plan_id != plan_id or baseline.operations != operations:
        return VerificationAttribution(AttributionResult.COMMAND_MISMATCH)
    if len(baseline.steps) <= successful_prior_steps or any(
        step.result != "passed" for step in baseline.steps[:successful_prior_steps]
    ):
        return VerificationAttribution(AttributionResult.UNATTRIBUTED)
    return attribute_failure(
        baseline.steps[successful_prior_steps],
        prepared,
        status,
        output,
        pre_mutation_generation,
    )


@dataclass(frozen=True, slots=True)
class VerificationAttribution:
    result: AttributionResult = AttributionResult.NO_BASELINE
    fingerprint_equal: bool | None = None
    post_duration_seconds: float = 0.0


def command_identity(prepared: PreparedProjectCommand) -> str:
    """Bind operation, argv, cwd, timeout, and relevant process environment."""
    environment = project_environment()
    value = (
        prepared.operation,
        prepared.argv,
        str(prepared.workspace.resolve()),
        prepared.timeout_seconds,
        tuple(sorted(environment.items())),
    )
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def failure_fingerprint(
    operation: str, output: Mapping[str, object], workspace: Path
) -> str | None:
    """Hash bounded explicit failure markers; refuse equality on incomplete output."""
    outcome = output.get("outcome")
    if outcome not in {"nonzero_exit", "timeout"}:
        return None
    if output.get("stdout_truncated") is True or output.get("stderr_truncated") is True:
        return None
    lines: list[str] = []
    for stream in ("stdout", "stderr"):
        raw = output.get(stream)
        if not isinstance(raw, str):
            continue
        for item in _ANSI.sub("", raw).splitlines():
            line = item.replace(str(workspace.resolve()), "<WORKSPACE>").strip()
            if not line or line.startswith("Total Test time"):
                continue
            if _SUMMARY_CASE.fullmatch(line) or _FAILURE_LINE.search(line):
                if len(line) > _MAX_LINE_CHARS or len(lines) >= _MAX_LINES:
                    return None
                lines.append(line)
    if not lines:
        return None
    material = (
        operation,
        outcome,
        output.get("exit_code"),
        output.get("timed_out"),
        tuple(lines),
    )
    return hashlib.sha256(json.dumps(material).encode()).hexdigest()


def baseline_evidence(
    prepared: PreparedProjectCommand,
    status: str,
    output: Mapping[str, object] | None,
    generation: int,
) -> VerificationBaselineEvidence:
    outcome = output.get("outcome") if output is not None else None
    result = (
        "passed"
        if status == "success" and outcome == "success"
        else "failed"
        if status == "failure" and outcome in {"nonzero_exit", "timeout"}
        else "unavailable"
    )
    duration = output.get("duration_seconds") if output is not None else None
    return VerificationBaselineEvidence(
        prepared.workspace.resolve(),
        generation,
        command_identity(prepared),
        result,
        failure_fingerprint(prepared.operation, output, prepared.workspace)
        if result == "failed" and output is not None
        else None,
        float(duration) if isinstance(duration, (int, float)) else 0.0,
        result in {"passed", "failed"},
    )


def attribute_failure(
    baseline: VerificationBaselineEvidence | None,
    prepared: PreparedProjectCommand | None,
    status: str,
    output: Mapping[str, object] | None,
    pre_mutation_generation: int,
) -> VerificationAttribution:
    duration = output.get("duration_seconds") if output is not None else None
    elapsed = float(duration) if isinstance(duration, (int, float)) else 0.0
    if status == "success":
        return VerificationAttribution(AttributionResult.NOT_APPLICABLE, None, elapsed)
    if baseline is None:
        return VerificationAttribution(AttributionResult.NO_BASELINE, None, elapsed)
    if prepared is None:
        return VerificationAttribution(
            AttributionResult.COMMAND_MISMATCH, None, elapsed
        )
    if baseline.result == "unavailable":
        return VerificationAttribution(
            AttributionResult.BASELINE_UNAVAILABLE, None, elapsed
        )
    if (
        baseline.workspace != prepared.workspace.resolve()
        or baseline.generation != pre_mutation_generation
        or baseline.command_identity != command_identity(prepared)
    ):
        return VerificationAttribution(
            AttributionResult.COMMAND_MISMATCH, None, elapsed
        )
    if (
        status != "failure"
        or output is None
        or output.get("outcome") not in {"nonzero_exit", "timeout"}
    ):
        return VerificationAttribution(AttributionResult.UNATTRIBUTED, None, elapsed)
    if baseline.result == "passed":
        return VerificationAttribution(
            AttributionResult.MUTATION_ASSOCIATED, None, elapsed
        )
    current = failure_fingerprint(prepared.operation, output, prepared.workspace)
    if baseline.failure_fingerprint is None or current is None:
        return VerificationAttribution(AttributionResult.UNATTRIBUTED, None, elapsed)
    equal = baseline.failure_fingerprint == current
    return VerificationAttribution(
        AttributionResult.PREEXISTING_OR_UNRELATED
        if equal
        else AttributionResult.UNATTRIBUTED,
        equal,
        elapsed,
    )

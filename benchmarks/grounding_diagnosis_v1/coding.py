"""A63 C0/C1 sufficient-context coding diagnostics."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from benchmarks.grounding_diagnosis_v1.suite import HistoricalCase
from benchmarks.real_repository_pilot_v1.suite import (
    create_snapshot,
    snapshot_description,
)
from benchmarks.real_repository_pilot_v2.suite import PilotV2Task
from benchmarks.realistic_coding_v2.runner import RecordingModel, envelope_shape
from forge.conversation import Message, MessageRole
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    apply_task_setup,
)
from forge.models import (
    GenerationConfig,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    MutationRepresentationPolicy,
)
from forge.orchestration.protocol import build_mutation_ready_output


@dataclass(frozen=True, slots=True)
class CodingCondition:
    structural_valid: bool
    transaction: bool
    verification_pass: bool
    repair_attempted: bool
    semantic_pass: bool
    failure: str
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    generation_seconds: float


@dataclass(frozen=True, slots=True)
class CodingDiagnostic:
    case_id: str
    profile: str
    task_id: str
    source_paths: tuple[str, ...]
    representation: str
    same_task: bool
    same_source: bool
    same_representation: bool
    c0: CodingCondition
    c1: CodingCondition
    interpretation: str


class StaticResponseModel(Model):
    def __init__(self, backend: Model, response: ModelResponse):
        self.backend = backend
        self.response = response
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return self.backend.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, _request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return self.response

    def close(self) -> None:
        """The shared diagnostic backend is owned by the outer runner."""


def _condition(raw, recorder, structural: bool) -> CodingCondition:  # type: ignore[no-untyped-def]
    metrics = raw.metrics
    transaction = bool(metrics.mutations) and not raw.unexpected_paths
    verification = (
        metrics.verification_plan_result == "pass"
        or metrics.reverification_result == "pass"
    )
    semantic = raw.oracle is EvaluationOutcome.PASS
    repair = bool(metrics.repair_attempts)
    if not structural:
        failure = "PROTOCOL_SCHEMA"
    elif metrics.line_range_attempts and not metrics.line_range_materialized:
        failure = "INCORRECT_LINE_RANGE_CONSTRUCTION"
    elif not transaction:
        failure = "INCOMPLETE_REQUIRED_OPERATION_SET"
    elif repair and not semantic:
        failure = "VERIFICATION_TRIGGERED_REPAIR_FAILURE"
    elif not semantic:
        failure = "SEMANTIC_IMPLEMENTATION_ERROR"
    else:
        failure = "PASS"
    return CodingCondition(
        structural,
        transaction,
        verification,
        repair,
        semantic,
        failure,
        recorder.calls,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        round(recorder.generation_seconds, 3),
    )


def _structural_envelope(
    text: str, paths: tuple[str, ...]
) -> tuple[bool, dict[str, object]]:
    envelope = envelope_shape(text, frozenset(paths))
    kind = envelope.get("type")
    if kind in {"structured_edit", "line_range_edit"}:
        children = envelope.get("children", [])
        actual = {item.get("path") for item in children if isinstance(item, dict)}
        return actual.issubset(set(paths)) and bool(actual), envelope
    if kind in {"multi_file_structured_edit", "multi_file_line_range_edit"}:
        children = envelope.get("children", [])
        actual = {item.get("path") for item in children if isinstance(item, dict)}
        return actual == set(paths), envelope
    return False, envelope


def _bounded_sources(
    workspace: Path, paths: tuple[str, ...], *, character_budget: int = 18_000
) -> str:
    """Render only production-acquired files within the unchanged context budget."""
    share = max(1, character_budget // len(paths))
    rendered = []
    for path in paths:
        text = (workspace / path).read_text(encoding="utf-8")
        if len(text) > share:
            text = text[:share] + "\n[bounded source excerpt]\n"
        rendered.append(f"FILE {path}\n{text}")
    return "\n\n".join(rendered)


def run_coding_diagnostic(
    case: HistoricalCase,
    definition: PilotV2Task,
    canonical: Path,
    model: Model,
    representation: MutationRepresentationPolicy,
) -> CodingDiagnostic:
    fixed_task = replace(
        definition.production_task,
        expected_files=case.acquired_paths,
        allowed_paths=case.acquired_paths,
        expected_changed_paths=case.acquired_paths,
        required_candidate_paths=case.acquired_paths,
    )
    c0_recorder = RecordingModel(model, frozenset(case.acquired_paths))
    c0_raw = (
        RealWorldEvaluationRunner(
            case.profile,
            c0_recorder,
            canonical,
            mutation_representation=representation,
        )
        .run((fixed_task,), snapshot_description(canonical))
        .results[0]
    )
    c0_structural = bool(
        c0_raw.metrics.structured_mutation_valid
        or c0_raw.metrics.preview_created
        or c0_raw.metrics.mutations
    )
    c0 = _condition(c0_raw, c0_recorder, c0_structural)

    with tempfile.TemporaryDirectory(prefix=f"forge-a63-{case.case_id}-c1-") as name:
        workspace = create_snapshot(canonical, Path(name) / "foundation")
        apply_task_setup(workspace, fixed_task.setup)
        sources = _bounded_sources(workspace, case.acquired_paths)
        request = ModelRequest(
            (
                Message(
                    MessageRole.SYSTEM,
                    "Return only the JSON mutation required by the supplied schema.",
                ),
                Message(
                    MessageRole.USER,
                    f"{fixed_task.prompt}\n\nAuthoritative source:\n{sources}",
                ),
            ),
            GenerationConfig(max_tokens=512, temperature=0.0, seed=42),
            build_mutation_ready_output(
                case.acquired_paths, representation=representation
            ),
        )
        started = time.perf_counter()
        response = model.generate(request)
        c1_seconds = time.perf_counter() - started
    c1_structural, _envelope = _structural_envelope(response.text, case.acquired_paths)
    static = StaticResponseModel(model, response)
    c1_recorder = RecordingModel(static, frozenset(case.acquired_paths))
    c1_raw = (
        RealWorldEvaluationRunner(
            case.profile,
            c1_recorder,
            canonical,
            mutation_representation=representation,
        )
        .run((fixed_task,), snapshot_description(canonical))
        .results[0]
    )
    c1 = replace(
        _condition(c1_raw, c1_recorder, c1_structural),
        generation_seconds=round(c1_seconds, 3),
    )
    if not c0.structural_valid and c1.structural_valid:
        interpretation = "PRODUCTION_CONTINUITY_OR_FRAMING_ISSUE"
    elif not c0.structural_valid and not c1.structural_valid:
        interpretation = "MODEL_PROTOCOL_ISSUE"
    elif (
        c0.structural_valid
        and c1.structural_valid
        and (not c0.semantic_pass or not c1.semantic_pass)
    ):
        interpretation = "SEMANTIC_CODING_LIMITATION"
    else:
        interpretation = "MIXED_POST_GROUNDING_FAILURE"
    return CodingDiagnostic(
        case.case_id,
        case.profile,
        case.task_id,
        case.acquired_paths,
        representation.value,
        True,
        True,
        True,
        c0,
        c1,
        interpretation,
    )

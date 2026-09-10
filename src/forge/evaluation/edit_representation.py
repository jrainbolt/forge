"""Evaluation-only comparison of exact-text, line-range, and span edits."""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.protocol_compatibility import (
    DIAGNOSTIC_GENERATION,
    SYNTHETIC_FIXTURES,
    DiagnosticFixture,
    ProtocolLayerResult,
    run_exact_edit_fixture,
    run_foundation_exact_edit,
    run_synthetic_oracle,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldTask,
    apply_task_setup,
    copy_repository,
    run_oracle,
)
from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    ModelUsage,
    OutputSpecification,
    ResponseFormat,
)
from forge.orchestration.structured_edit import MAX_EDIT_LINES, MAX_EDIT_TEXT_BYTES

EDIT_REPRESENTATION_V1 = "edit-representation-v1"
EDIT_REPRESENTATION_PROMPT_VERSION = "edit-representation-diagnostic-v1"
EDIT_REPRESENTATION_SUITE_VERSION = 1
EDIT_REPRESENTATION_SCHEMA_VERSION = 1


class EditRepresentation(Enum):
    EXACT_TEXT = "R1_EXACT_TEXT"
    LINE_RANGE = "R2_LINE_RANGE"
    SOURCE_SPAN = "R3_SOURCE_SPAN"


class EditFailure(Enum):
    GENERATION_FAILED = "GENERATION_FAILED"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    TARGET_INVALID = "TARGET_INVALID"
    TARGET_WRONG = "TARGET_WRONG"
    NO_DELTA = "NO_DELTA"
    SEMANTIC_FAILURE = "SEMANTIC_FAILURE"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    span_id: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class EditRepresentationResult:
    model_profile: str
    task_id: str
    representation: str
    seed: int
    generation_status: str
    failure: str
    response_valid: bool
    target_selectable: bool
    replacement_valid: bool
    target_region_correct: bool | None
    material_delta: bool
    oracle: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class EditRepresentationAggregate:
    model_profile: str
    representation: str
    task_count: int
    valid_representations: int
    correct_targets: int | None
    material_deltas: int
    oracle_passes: int
    mean_latency_seconds: float


@dataclass(frozen=True, slots=True)
class RepresentationGain:
    model_profile: str
    r1_oracle_rate: float
    r2_oracle_gain: float
    r3_oracle_gain: float
    exact_copy_penalty_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EditRepresentationRun:
    suite: str
    suite_version: int
    schema_version: int
    prompt_version: str
    generation: GenerationConfig
    context_capacity: int
    results: tuple[EditRepresentationResult, ...]
    aggregates: tuple[EditRepresentationAggregate, ...]
    gains: tuple[RepresentationGain, ...]


R2_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "new_text": {"type": "string"},
        },
        "required": ["start_line", "end_line", "new_text"],
        "additionalProperties": False,
    },
)

R3_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            "span_id": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["span_id", "new_text"],
        "additionalProperties": False,
    },
)


def run_edit_representation_diagnostics(
    model_profile: str,
    model: Model,
    *,
    foundation_repository: Path | None = None,
    foundation_task: RealWorldTask | None = None,
) -> tuple[EditRepresentationResult, ...]:
    """Run R1/R2/R3 with equivalent task, source, settings, and oracle."""
    results: list[EditRepresentationResult] = []
    for fixture in SYNTHETIC_FIXTURES:
        results.extend(_run_fixture(model_profile, model, fixture))
    if (foundation_repository is None) != (foundation_task is None):
        raise ValueError("Foundation repository and task must be supplied together")
    if foundation_repository is not None and foundation_task is not None:
        fixture = _foundation_fixture(foundation_repository, foundation_task)
        results.append(
            _from_r1(
                run_foundation_exact_edit(
                    model_profile,
                    model,
                    fixture,
                    foundation_task,
                    foundation_repository,
                )
            )
        )
        results.append(
            _generate_and_evaluate(
                model_profile,
                model,
                fixture,
                EditRepresentation.LINE_RANGE,
                foundation_repository,
                foundation_task,
            )
        )
        results.append(
            _generate_and_evaluate(
                model_profile,
                model,
                fixture,
                EditRepresentation.SOURCE_SPAN,
                foundation_repository,
                foundation_task,
            )
        )
    return tuple(results)


def build_edit_representation_run(
    results: tuple[EditRepresentationResult, ...],
    *,
    context_capacity: int = 8192,
) -> EditRepresentationRun:
    profiles = tuple(dict.fromkeys(item.model_profile for item in results))
    representations = tuple(item.value for item in EditRepresentation)
    aggregates = tuple(
        aggregate_edit_representation(profile, representation, results)
        for profile in profiles
        for representation in representations
    )
    return EditRepresentationRun(
        EDIT_REPRESENTATION_V1,
        EDIT_REPRESENTATION_SUITE_VERSION,
        EDIT_REPRESENTATION_SCHEMA_VERSION,
        EDIT_REPRESENTATION_PROMPT_VERSION,
        DIAGNOSTIC_GENERATION,
        context_capacity,
        results,
        aggregates,
        tuple(_gain(profile, results) for profile in profiles),
    )


def aggregate_edit_representation(
    profile: str,
    representation: str,
    results: tuple[EditRepresentationResult, ...],
) -> EditRepresentationAggregate:
    selected = tuple(
        item
        for item in results
        if item.model_profile == profile and item.representation == representation
    )
    target_values = tuple(
        item.target_region_correct
        for item in selected
        if item.target_region_correct is not None
    )
    return EditRepresentationAggregate(
        profile,
        representation,
        len(selected),
        sum(item.response_valid for item in selected),
        sum(target_values) if target_values else None,
        sum(item.material_delta for item in selected),
        sum(item.oracle == EvaluationOutcome.PASS.value for item in selected),
        sum(item.latency_seconds for item in selected) / len(selected)
        if selected
        else 0.0,
    )


def edit_representation_to_dict(run: EditRepresentationRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_edit_representation_json(run: EditRepresentationRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(edit_representation_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_edit_representation(run: EditRepresentationRun) -> str:
    lines = [
        "Model       Task  Representation  Valid Target Delta "
        "Oracle  Failure           Time"
    ]
    for item in run.results:
        target = (
            "n/a"
            if item.target_region_correct is None
            else ("yes" if item.target_region_correct else "no")
        )
        lines.append(
            f"{item.model_profile:<11} {item.task_id:<5} "
            f"{item.representation:<15} {'yes' if item.response_valid else 'no':<5} "
            f"{target:<6} {'yes' if item.material_delta else 'no':<5} "
            f"{item.oracle:<7} {item.failure:<17} {item.latency_seconds:.2f}s"
        )
    lines.append("")
    lines.append("Representation gains")
    for gain in run.gains:
        penalty = ",".join(gain.exact_copy_penalty_tasks) or "none"
        lines.append(
            f"{gain.model_profile}: R1={gain.r1_oracle_rate:.3f}; "
            f"R2-R1={gain.r2_oracle_gain:+.3f}; "
            f"R3-R1={gain.r3_oracle_gain:+.3f}; exact-copy={penalty}"
        )
    return "\n".join(lines)


def construct_source_spans(source: str) -> tuple[SourceSpan, ...]:
    """Partition all source into at most eight opaque contiguous regions."""
    lines = source.splitlines(keepends=True)
    if not lines:
        return ()
    blocks: list[tuple[int, int]] = []
    start = 1
    for number, line in enumerate(lines, 1):
        if not line.strip():
            blocks.append((start, number))
            start = number + 1
    if start <= len(lines):
        blocks.append((start, len(lines)))
    if not blocks:
        blocks.append((1, len(lines)))
    if len(blocks) < min(3, len(lines)):
        blocks = _equal_line_ranges(len(lines), min(3, len(lines)))
    if len(blocks) > 8:
        blocks = _equal_line_ranges(len(lines), 8)
    return tuple(
        SourceSpan(f"S{index}", start, end)
        for index, (start, end) in enumerate(blocks, 1)
    )


def _equal_line_ranges(line_count: int, count: int) -> list[tuple[int, int]]:
    return [
        (
            math.floor(index * line_count / count) + 1,
            math.floor((index + 1) * line_count / count),
        )
        for index in range(count)
    ]


def materialize_line_range(
    source: str, start_line: int, end_line: int, new_text: str
) -> str:
    lines = source.splitlines(keepends=True)
    if (
        isinstance(start_line, bool)
        or isinstance(end_line, bool)
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < start_line
        or end_line > len(lines)
    ):
        raise ValueError("invalid source line range")
    _validate_replacement(new_text)
    start = sum(len(line) for line in lines[: start_line - 1])
    end = sum(len(line) for line in lines[:end_line])
    updated = source[:start] + new_text + source[end:]
    if updated == source:
        raise ValueError("replacement produces no delta")
    return updated


def materialize_source_span(
    source: str, spans: tuple[SourceSpan, ...], span_id: str, new_text: str
) -> tuple[str, SourceSpan]:
    span = next((item for item in spans if item.span_id == span_id), None)
    if span is None:
        raise ValueError("invalid source span")
    return (
        materialize_line_range(source, span.start_line, span.end_line, new_text),
        span,
    )


def _run_fixture(
    profile: str, model: Model, fixture: DiagnosticFixture
) -> tuple[EditRepresentationResult, ...]:
    return (
        _from_r1(run_exact_edit_fixture(profile, model, fixture)),
        _generate_and_evaluate(profile, model, fixture, EditRepresentation.LINE_RANGE),
        _generate_and_evaluate(profile, model, fixture, EditRepresentation.SOURCE_SPAN),
    )


def _from_r1(result: ProtocolLayerResult) -> EditRepresentationResult:
    generated = result.generation_status == "generated"
    status = result.representation_status
    response_valid = generated and status not in {
        "could_not_produce_fields",
        "missing_field",
        "schema_invalid",
    }
    if not generated:
        failure = EditFailure.GENERATION_FAILED
    elif not response_valid:
        failure = EditFailure.SCHEMA_INVALID
    elif not result.representation_valid:
        failure = (
            EditFailure.NO_DELTA if status == "no_op" else EditFailure.TARGET_INVALID
        )
    elif result.oracle == EvaluationOutcome.PASS.value:
        failure = EditFailure.PASS
    else:
        failure = EditFailure.SEMANTIC_FAILURE
    return EditRepresentationResult(
        result.model_profile,
        result.task_id,
        EditRepresentation.EXACT_TEXT.value,
        result.seed,
        result.generation_status,
        failure.value,
        response_valid,
        result.representation_valid,
        result.material_delta,
        None,
        result.material_delta,
        result.oracle,
        result.latency_seconds,
        result.input_tokens,
        result.output_tokens,
    )


def _generate_and_evaluate(
    profile: str,
    model: Model,
    fixture: DiagnosticFixture,
    representation: EditRepresentation,
    repository: Path | None = None,
    task: RealWorldTask | None = None,
) -> EditRepresentationResult:
    spans = construct_source_spans(fixture.source)
    if representation is EditRepresentation.LINE_RANGE:
        source = _numbered_source(fixture.source)
        instruction = (
            "Choose the current source line range that should be replaced and "
            "provide the replacement text."
        )
        output = R2_OUTPUT
    else:
        source = _spanned_source(fixture.source, spans)
        instruction = (
            "Choose the source span that should be replaced and provide the "
            "replacement text."
        )
        output = R3_OUTPUT
    prompt = (
        f"Coding task:\n{fixture.task}\n\nCurrent path:\n{fixture.path}"
        f"\n\nCurrent source:\n{source}\n{instruction}"
    )
    started = time.perf_counter()
    try:
        response = model.generate(
            ModelRequest(
                (Message(MessageRole.USER, prompt),), DIAGNOSTIC_GENERATION, output
            )
        )
    except Exception:
        return _failed_generation(profile, fixture.task_id, representation, started)
    latency = time.perf_counter() - started
    return evaluate_representation_response(
        profile,
        fixture,
        representation,
        response.text,
        latency,
        response.usage,
        spans=spans,
        repository=repository,
        task=task,
    )


def evaluate_representation_response(
    profile: str,
    fixture: DiagnosticFixture,
    representation: EditRepresentation,
    text: str,
    latency_seconds: float,
    usage: ModelUsage,
    *,
    spans: tuple[SourceSpan, ...] | None = None,
    repository: Path | None = None,
    task: RealWorldTask | None = None,
) -> EditRepresentationResult:
    """Strictly parse, materialize, and score one R2 or R3 response."""
    try:
        values = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return _invalid_result(
            profile,
            fixture.task_id,
            representation,
            EditFailure.SCHEMA_INVALID,
            latency_seconds,
            usage,
        )
    if not isinstance(values, dict):
        return _invalid_result(
            profile,
            fixture.task_id,
            representation,
            EditFailure.SCHEMA_INVALID,
            latency_seconds,
            usage,
        )
    target_correct = False
    try:
        if representation is EditRepresentation.LINE_RANGE:
            if set(values) != {"start_line", "end_line", "new_text"}:
                raise TypeError
            start_line = values["start_line"]
            end_line = values["end_line"]
            new_text = values["new_text"]
            if (
                isinstance(start_line, bool)
                or isinstance(end_line, bool)
                or not isinstance(start_line, int)
                or not isinstance(end_line, int)
                or not isinstance(new_text, str)
            ):
                raise TypeError
            line_count = len(fixture.source.splitlines(keepends=True))
            if start_line < 1 or end_line < start_line or end_line > line_count:
                return _invalid_result(
                    profile,
                    fixture.task_id,
                    representation,
                    EditFailure.TARGET_INVALID,
                    latency_seconds,
                    usage,
                    response_valid=True,
                )
            target_correct = start_line <= fixture.expected_line <= end_line
            updated = materialize_line_range(
                fixture.source, start_line, end_line, new_text
            )
        else:
            if set(values) != {"span_id", "new_text"}:
                raise TypeError
            span_id = values["span_id"]
            new_text = values["new_text"]
            if not isinstance(span_id, str) or not isinstance(new_text, str):
                raise TypeError
            available_spans = spans or construct_source_spans(fixture.source)
            span = next(
                (item for item in available_spans if item.span_id == span_id), None
            )
            if span is None:
                return _invalid_result(
                    profile,
                    fixture.task_id,
                    representation,
                    EditFailure.TARGET_INVALID,
                    latency_seconds,
                    usage,
                    response_valid=True,
                )
            target_correct = span.start_line <= fixture.expected_line <= span.end_line
            updated, span = materialize_source_span(
                fixture.source,
                available_spans,
                span_id,
                new_text,
            )
    except TypeError:
        return _invalid_result(
            profile,
            fixture.task_id,
            representation,
            EditFailure.SCHEMA_INVALID,
            latency_seconds,
            usage,
        )
    except ValueError:
        failure = EditFailure.NO_DELTA
        return _invalid_result(
            profile,
            fixture.task_id,
            representation,
            failure,
            latency_seconds,
            usage,
            response_valid=True,
            target_selectable=True,
            target_correct=target_correct,
        )
    oracle = _run_materialized_oracle(fixture, updated, repository, task)
    failure = (
        EditFailure.TARGET_WRONG
        if not target_correct
        else EditFailure.PASS
        if oracle is EvaluationOutcome.PASS
        else EditFailure.SEMANTIC_FAILURE
    )
    return EditRepresentationResult(
        profile,
        fixture.task_id,
        representation.value,
        42,
        "generated",
        failure.value,
        True,
        True,
        bool(new_text) and updated != fixture.source,
        target_correct,
        updated != fixture.source,
        oracle.value,
        latency_seconds,
        usage.input_tokens,
        usage.output_tokens,
    )


def _run_materialized_oracle(
    fixture: DiagnosticFixture,
    updated: str,
    repository: Path | None,
    task: RealWorldTask | None,
) -> EvaluationOutcome:
    if repository is not None and task is not None:
        with tempfile.TemporaryDirectory(prefix="forge-edit-representation-") as name:
            workspace = copy_repository(repository, Path(name).resolve() / "workspace")
            apply_task_setup(workspace, task.setup)
            if run_oracle(workspace, task.setup_commands) is EvaluationOutcome.FAIL:
                raise RuntimeError("Foundation diagnostic setup failed")
            (workspace / fixture.path).write_text(updated, encoding="utf-8")
            return run_oracle(workspace, task.oracle_commands)
    with tempfile.TemporaryDirectory(prefix="forge-edit-representation-") as name:
        workspace = Path(name).resolve()
        (workspace / fixture.path).write_text(updated, encoding="utf-8")
        return run_synthetic_oracle(fixture, workspace)


def _foundation_fixture(repository: Path, task: RealWorldTask) -> DiagnosticFixture:
    if task.task_id != "E04" or len(task.expected_files) != 1 or len(task.setup) != 1:
        raise ValueError("Foundation diagnostic requires unchanged E04")
    with tempfile.TemporaryDirectory(prefix="forge-edit-representation-") as name:
        workspace = copy_repository(repository, Path(name).resolve() / "workspace")
        apply_task_setup(workspace, task.setup)
        if run_oracle(workspace, task.setup_commands) is EvaluationOutcome.FAIL:
            raise RuntimeError("Foundation diagnostic setup failed")
        source = (workspace / task.expected_files[0]).read_text(encoding="utf-8")
    expected = task.setup[0].replacement
    offset = source.find(expected)
    if offset < 0:
        raise RuntimeError("Foundation E04 expected source location missing")
    return DiagnosticFixture(
        "E04",
        "C",
        task.expected_files[0],
        task.prompt,
        source,
        (),
        (),
        "foundation",
        source.count("\n", 0, offset) + 1,
    )


def _numbered_source(source: str) -> str:
    return "".join(
        f"{number:>4} | {line}"
        for number, line in enumerate(source.splitlines(keepends=True), 1)
    )


def _spanned_source(source: str, spans: tuple[SourceSpan, ...]) -> str:
    lines = source.splitlines(keepends=True)
    chunks = []
    for span in spans:
        chunks.append(f"[{span.span_id}]\n")
        chunks.append("".join(lines[span.start_line - 1 : span.end_line]))
        if chunks[-1] and not chunks[-1].endswith("\n"):
            chunks.append("\n")
        chunks.append(f"[/{span.span_id}]\n")
    return "".join(chunks)


def _validate_replacement(new_text: str) -> None:
    if not isinstance(new_text, str) or not new_text:
        raise ValueError("replacement must be non-empty")
    try:
        encoded = new_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("replacement encoding is invalid") from error
    lines = new_text.count("\n") + (0 if new_text.endswith("\n") else 1)
    if len(encoded) > MAX_EDIT_TEXT_BYTES or lines > MAX_EDIT_LINES:
        raise ValueError("replacement is too large")


def _invalid_result(
    profile: str,
    task_id: str,
    representation: EditRepresentation,
    failure: EditFailure,
    latency: float,
    usage: ModelUsage,
    *,
    response_valid: bool = False,
    target_selectable: bool = False,
    target_correct: bool = False,
) -> EditRepresentationResult:
    return EditRepresentationResult(
        profile,
        task_id,
        representation.value,
        42,
        "generated",
        failure.value,
        response_valid,
        target_selectable,
        False,
        target_correct,
        False,
        EvaluationOutcome.NOT_RUN.value,
        latency,
        usage.input_tokens,
        usage.output_tokens,
    )


def _failed_generation(
    profile: str,
    task_id: str,
    representation: EditRepresentation,
    started: float,
) -> EditRepresentationResult:
    return EditRepresentationResult(
        profile,
        task_id,
        representation.value,
        42,
        "generation_failure",
        EditFailure.GENERATION_FAILED.value,
        False,
        False,
        False,
        False,
        False,
        EvaluationOutcome.NOT_RUN.value,
        time.perf_counter() - started,
        0,
        0,
    )


def _gain(
    profile: str, results: tuple[EditRepresentationResult, ...]
) -> RepresentationGain:
    selected = tuple(item for item in results if item.model_profile == profile)
    by_rep = {
        representation.value: tuple(
            item for item in selected if item.representation == representation.value
        )
        for representation in EditRepresentation
    }
    rates = {
        key: sum(item.oracle == EvaluationOutcome.PASS.value for item in values)
        / len(values)
        if values
        else 0.0
        for key, values in by_rep.items()
    }
    r1_by_task = {
        item.task_id: item for item in by_rep[EditRepresentation.EXACT_TEXT.value]
    }
    alternatives = (
        *by_rep[EditRepresentation.LINE_RANGE.value],
        *by_rep[EditRepresentation.SOURCE_SPAN.value],
    )
    penalty = tuple(
        sorted(
            {
                item.task_id
                for item in alternatives
                if item.target_region_correct
                and item.oracle == EvaluationOutcome.PASS.value
                and item.task_id in r1_by_task
                and r1_by_task[item.task_id].oracle != EvaluationOutcome.PASS.value
            }
        )
    )
    r1 = rates[EditRepresentation.EXACT_TEXT.value]
    return RepresentationGain(
        profile,
        r1,
        rates[EditRepresentation.LINE_RANGE.value] - r1,
        rates[EditRepresentation.SOURCE_SPAN.value] - r1,
        penalty,
    )

"""Evaluation-only separation of reasoning, edit, protocol, and orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldTask,
    RealWorldTaskResult,
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
from forge.orchestration import (
    MutationCandidate,
    StructuredEditProposal,
    ToolCallOutcome,
    parse_model_output,
    validate_structured_edit,
)
from forge.orchestration.protocol import build_mutation_ready_output

PROTOCOL_COMPATIBILITY_V1 = "protocol-compatibility-v1"
PROTOCOL_DIAGNOSTIC_PROMPT_VERSION = "protocol-diagnostic-v1"
PROTOCOL_COMPATIBILITY_SUITE_VERSION = 1
PROTOCOL_COMPATIBILITY_SCHEMA_VERSION = 1
DIAGNOSTIC_GENERATION = GenerationConfig(max_tokens=512, temperature=0.0, seed=42)


class DiagnosticLayer(Enum):
    L1_REASONING = "L1"
    L2_EDIT = "L2"
    L3_PROTOCOL = "L3"
    L4_FORGE = "L4"


class GenerationStatus(Enum):
    GENERATED = "generated"
    FAILED = "generation_failure"


class RepresentationStatus(Enum):
    NOT_APPLICABLE = "not_applicable"
    CONCEPT_IDENTIFIED = "concept_identified"
    CONCEPT_MISSING = "concept_missing"
    CONTRADICTORY = "contradictory"
    FIELDS_INVALID = "could_not_produce_fields"
    SCHEMA_INVALID = "schema_invalid"
    WRONG_RESPONSE_TYPE = "wrong_response_type"
    MISSING_FIELD = "missing_field"
    PATH_MISMATCH = "path_mismatch"
    OLD_TEXT_MISMATCH = "old_text_mismatch"
    NO_OP = "no_op"
    INVALID_DELTA = "invalid_delta"
    VALID_DELTA = "valid_delta"
    FULL_FORGE = "full_forge"


@dataclass(frozen=True, slots=True)
class DiagnosticFixture:
    task_id: str
    language: str
    path: str
    task: str
    source: str
    concept_groups: tuple[tuple[str, ...], ...]
    contradictory_groups: tuple[tuple[str, ...], ...]
    oracle: str
    expected_line: int


@dataclass(frozen=True, slots=True)
class ProtocolLayerResult:
    model_profile: str
    task_id: str
    layer: str
    seed: int
    generation_status: str
    representation_status: str
    representation_valid: bool
    material_delta: bool
    oracle: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ProtocolModelAggregate:
    model_profile: str
    task_count: int
    l1_concepts: int
    l2_valid_edits: int
    l2_oracle_passes: int
    l3_schema_valid: int
    l3_valid_edits: int
    l3_oracle_passes: int
    l4_valid_edits: int
    l4_oracle_passes: int
    l1_to_l2_drop: int
    l2_to_l3_drop: int
    l3_to_l4_drop: int


@dataclass(frozen=True, slots=True)
class ProtocolCompatibilityRun:
    suite: str
    suite_version: int
    schema_version: int
    prompt_version: str
    generation: GenerationConfig
    results: tuple[ProtocolLayerResult, ...]
    aggregates: tuple[ProtocolModelAggregate, ...]


SYNTHETIC_FIXTURES = (
    DiagnosticFixture(
        "P01",
        "C",
        "boundary.c",
        "The boundary value itself should be allowed.",
        "#include <stdbool.h>\nbool allowed(int value, int limit) {\n"
        "    return value > limit;\n}\n",
        (("boundary", "equal", "limit"), ("allow", "include", ">=")),
        (("exclude equality", "keep >"),),
        "boundary",
        3,
    ),
    DiagnosticFixture(
        "P02",
        "Python",
        "feature.py",
        "A true flag should enable the feature and a false flag should disable it.",
        "def enabled(flag: bool) -> bool:\n    return not flag\n",
        (("true",), ("enable",), ("false", "disable")),
        (("invert", "not flag"),),
        "boolean",
        2,
    ),
    DiagnosticFixture(
        "P03",
        "C",
        "retry.c",
        "The default retry limit should permit three attempts.",
        "int retry_limit(void) {\n    return 1;\n}\n",
        (("three", "3"), ("retry", "attempt")),
        (("one attempt", "return 1"),),
        "constant",
        2,
    ),
)

L2_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["old_text", "new_text"],
        "additionalProperties": False,
    },
)


def run_protocol_diagnostics(
    model_profile: str,
    model: Model,
    *,
    foundation_repository: Path | None = None,
    foundation_task: RealWorldTask | None = None,
    l4_result: RealWorldTaskResult | None = None,
) -> tuple[ProtocolLayerResult, ...]:
    results: list[ProtocolLayerResult] = []
    for fixture in SYNTHETIC_FIXTURES:
        results.extend(_run_fixture(model_profile, model, fixture))
    if foundation_repository is not None and foundation_task is not None:
        results.extend(
            _run_foundation_layers(
                model_profile, model, foundation_repository, foundation_task
            )
        )
    if l4_result is not None:
        results.append(_l4(model_profile, l4_result))
    return tuple(results)


def run_exact_edit_fixture(
    model_profile: str, model: Model, fixture: DiagnosticFixture
) -> ProtocolLayerResult:
    """Run the existing A33 exact-text edit layer for one synthetic fixture."""
    return _edit_layer(model_profile, model, fixture, DiagnosticLayer.L2_EDIT)


def run_foundation_exact_edit(
    model_profile: str,
    model: Model,
    fixture: DiagnosticFixture,
    task: RealWorldTask,
    repository: Path,
) -> ProtocolLayerResult:
    """Run the existing A33 exact-text edit layer for Foundation E04."""
    return _foundation_edit(model_profile, model, fixture, task, repository, False)


def run_synthetic_oracle(
    fixture: DiagnosticFixture, workspace: Path
) -> EvaluationOutcome:
    """Run the existing A33 deterministic semantic oracle."""
    return _synthetic_oracle(fixture, workspace)


def build_protocol_compatibility_run(
    results: tuple[ProtocolLayerResult, ...],
) -> ProtocolCompatibilityRun:
    profiles = tuple(dict.fromkeys(result.model_profile for result in results))
    return ProtocolCompatibilityRun(
        PROTOCOL_COMPATIBILITY_V1,
        PROTOCOL_COMPATIBILITY_SUITE_VERSION,
        PROTOCOL_COMPATIBILITY_SCHEMA_VERSION,
        PROTOCOL_DIAGNOSTIC_PROMPT_VERSION,
        DIAGNOSTIC_GENERATION,
        results,
        tuple(aggregate_protocol_results(profile, results) for profile in profiles),
    )


def aggregate_protocol_results(
    profile: str, results: tuple[ProtocolLayerResult, ...]
) -> ProtocolModelAggregate:
    selected = tuple(result for result in results if result.model_profile == profile)
    l1 = tuple(result for result in selected if result.layer == "L1")
    l2 = tuple(result for result in selected if result.layer == "L2")
    l3 = tuple(result for result in selected if result.layer == "L3")
    l4 = tuple(result for result in selected if result.layer == "L4")
    l1_pass = sum(item.representation_status == "concept_identified" for item in l1)
    l2_valid = sum(item.representation_valid for item in l2)
    l3_schema = sum(
        item.representation_status not in {"schema_invalid", "wrong_response_type"}
        for item in l3
    )
    l3_valid = sum(item.representation_valid for item in l3)
    l4_valid = sum(item.representation_valid for item in l4)
    return ProtocolModelAggregate(
        profile,
        len({item.task_id for item in selected}),
        l1_pass,
        l2_valid,
        sum(item.oracle == "PASS" for item in l2),
        l3_schema,
        l3_valid,
        sum(item.oracle == "PASS" for item in l3),
        l4_valid,
        sum(item.oracle == "PASS" for item in l4),
        _transition_drop(l1, l2),
        _transition_drop(l2, l3),
        _transition_drop(l3, l4),
    )


def _transition_drop(
    left: tuple[ProtocolLayerResult, ...],
    right: tuple[ProtocolLayerResult, ...],
) -> int:
    left_by_task = {item.task_id: item for item in left}
    right_by_task = {item.task_id: item for item in right}
    shared = set(left_by_task) & set(right_by_task)
    return sum(left_by_task[task].representation_valid for task in shared) - sum(
        right_by_task[task].representation_valid for task in shared
    )


def protocol_compatibility_to_dict(run: ProtocolCompatibilityRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_protocol_compatibility_json(
    run: ProtocolCompatibilityRun, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol_compatibility_to_dict(run), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def render_protocol_compatibility(run: ProtocolCompatibilityRun) -> str:
    lines = ["Model       Task  Layer  Representation       Delta  Oracle  Time"]
    for result in run.results:
        lines.append(
            f"{result.model_profile:<11} {result.task_id:<5} {result.layer:<6} "
            f"{result.representation_status:<20} "
            f"{'yes' if result.material_delta else 'no':<6} "
            f"{result.oracle:<7} {result.latency_seconds:.2f}s"
        )
    lines.append("")
    lines.append("Model aggregates")
    for item in run.aggregates:
        lines.append(
            f"{item.model_profile}: L1={item.l1_concepts}; "
            f"L2={item.l2_valid_edits}/{item.l2_oracle_passes}; "
            f"L3={item.l3_valid_edits}/{item.l3_oracle_passes}; "
            f"L4={item.l4_valid_edits}/{item.l4_oracle_passes}; "
            f"drops={item.l1_to_l2_drop},{item.l2_to_l3_drop},"
            f"{item.l3_to_l4_drop}"
        )
    return "\n".join(lines)


def _run_fixture(
    profile: str, model: Model, fixture: DiagnosticFixture
) -> tuple[ProtocolLayerResult, ...]:
    return (
        _l1(profile, model, fixture),
        _edit_layer(profile, model, fixture, DiagnosticLayer.L2_EDIT),
        _edit_layer(profile, model, fixture, DiagnosticLayer.L3_PROTOCOL),
    )


def _l1(profile: str, model: Model, fixture: DiagnosticFixture) -> ProtocolLayerResult:
    prompt = _base_prompt(fixture) + (
        "\nConcise answer: what behavior is wrong and what behavior must change? "
        "Do not provide a patch or step-by-step reasoning."
    )
    started = time.perf_counter()
    try:
        response = model.generate(
            ModelRequest((Message(MessageRole.USER, prompt),), DIAGNOSTIC_GENERATION)
        )
    except Exception:
        return _generation_failure(profile, fixture.task_id, "L1", started)
    generation_latency = time.perf_counter() - started
    normalized = response.text.casefold()
    contradictory = any(
        all(term in normalized for term in group)
        for group in fixture.contradictory_groups
    )
    concepts = all(
        any(term in normalized for term in group) for group in fixture.concept_groups
    )
    status = (
        RepresentationStatus.CONTRADICTORY
        if contradictory
        else RepresentationStatus.CONCEPT_IDENTIFIED
        if concepts
        else RepresentationStatus.CONCEPT_MISSING
    )
    return _result(
        profile,
        fixture.task_id,
        "L1",
        status,
        status is RepresentationStatus.CONCEPT_IDENTIFIED,
        False,
        EvaluationOutcome.NOT_RUN,
        generation_latency,
        response.usage,
    )


def _edit_layer(
    profile: str,
    model: Model,
    fixture: DiagnosticFixture,
    layer: DiagnosticLayer,
) -> ProtocolLayerResult:
    l3 = layer is DiagnosticLayer.L3_PROTOCOL
    contract = (
        "Return the structured_edit action required by the supplied schema."
        if l3
        else "Return only old_text and new_text using the supplied JSON schema."
    )
    request = ModelRequest(
        (Message(MessageRole.USER, _base_prompt(fixture) + "\n" + contract),),
        DIAGNOSTIC_GENERATION,
        build_mutation_ready_output((fixture.path,)) if l3 else L2_OUTPUT,
    )
    started = time.perf_counter()
    try:
        response = model.generate(request)
    except Exception:
        return _generation_failure(profile, fixture.task_id, layer.value, started)
    generation_latency = time.perf_counter() - started
    proposal, status = _proposal(response.text, fixture.path, l3=l3)
    if proposal is None:
        return _result(
            profile,
            fixture.task_id,
            layer.value,
            status,
            False,
            False,
            EvaluationOutcome.NOT_RUN,
            generation_latency,
            response.usage,
        )
    validation, oracle = _validate_fixture(fixture, proposal)
    failure_status = _validation_status(
        validation.failure.value if validation.failure else None
    )
    return _result(
        profile,
        fixture.task_id,
        layer.value,
        failure_status,
        validation.valid,
        validation.valid,
        oracle,
        generation_latency,
        response.usage,
    )


def _proposal(
    text: str, path: str, *, l3: bool
) -> tuple[StructuredEditProposal | None, RepresentationStatus]:
    if l3:
        try:
            parsed = parse_model_output(text)
        except Exception:
            return None, RepresentationStatus.SCHEMA_INVALID
        if parsed.outcome is not ToolCallOutcome.STRUCTURED_EDIT:
            return None, RepresentationStatus.WRONG_RESPONSE_TYPE
        assert parsed.structured_edit is not None
        values = parsed.structured_edit
        if values.get("path") != path:
            return None, RepresentationStatus.PATH_MISMATCH
        return StructuredEditProposal(**values), RepresentationStatus.VALID_DELTA
    try:
        values = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None, RepresentationStatus.FIELDS_INVALID
    if not isinstance(values, dict) or set(values) != {"old_text", "new_text"}:
        return None, RepresentationStatus.MISSING_FIELD
    if not all(isinstance(values[key], str) for key in values):
        return None, RepresentationStatus.FIELDS_INVALID
    return (
        StructuredEditProposal(path, values["old_text"], values["new_text"]),
        RepresentationStatus.VALID_DELTA,
    )


def _validate_fixture(fixture: DiagnosticFixture, proposal: StructuredEditProposal):
    with tempfile.TemporaryDirectory(prefix="forge-protocol-") as name:
        workspace = Path(name).resolve()
        path = workspace / fixture.path
        path.write_text(fixture.source, encoding="utf-8")
        candidate = MutationCandidate(
            fixture.path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            0,
            "evaluator-source",
        )
        validation = validate_structured_edit(proposal, (candidate,), workspace, 0)
        if not validation.valid:
            return validation, EvaluationOutcome.NOT_RUN
        updated = fixture.source.replace(proposal.old_text, proposal.new_text, 1)
        path.write_text(updated, encoding="utf-8")
        return validation, _synthetic_oracle(fixture, workspace)


def _synthetic_oracle(fixture: DiagnosticFixture, workspace: Path) -> EvaluationOutcome:
    if fixture.oracle == "boolean":
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); import feature; "
            "assert feature.enabled(True) is True; "
            "assert feature.enabled(False) is False"
        )
        command = (shutil.which("python3") or "python3", "-c", script, str(workspace))
    else:
        function = (
            "allowed(5, 5) && !allowed(4, 5)"
            if fixture.oracle == "boundary"
            else "retry_limit() == 3"
        )
        declaration = (
            "#include <stdbool.h>\nbool allowed(int,int);"
            if fixture.oracle == "boundary"
            else "int retry_limit(void);"
        )
        harness = workspace / "oracle.c"
        harness.write_text(
            declaration + f"\nint main(void) {{ return {function} ? 0 : 1; }}\n"
        )
        executable = workspace / "oracle"
        compiler = shutil.which("cc")
        if compiler is None:
            return EvaluationOutcome.FAIL
        completed = subprocess.run(
            [
                compiler,
                str(workspace / fixture.path),
                str(harness),
                "-o",
                str(executable),
            ],
            capture_output=True,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            return EvaluationOutcome.FAIL
        command = (str(executable),)
    completed = subprocess.run(
        command, cwd=workspace, capture_output=True, check=False, shell=False
    )
    return (
        EvaluationOutcome.PASS if completed.returncode == 0 else EvaluationOutcome.FAIL
    )


def _run_foundation_layers(
    profile: str, model: Model, repository: Path, task: RealWorldTask
) -> tuple[ProtocolLayerResult, ...]:
    if task.task_id != "E04" or len(task.expected_files) != 1:
        raise ValueError("Foundation diagnostic requires unchanged E04")
    with tempfile.TemporaryDirectory(prefix="forge-protocol-foundation-") as name:
        workspace = copy_repository(repository, Path(name).resolve() / "workspace")
        apply_task_setup(workspace, task.setup)
        if run_oracle(workspace, task.setup_commands) is EvaluationOutcome.FAIL:
            raise RuntimeError("Foundation diagnostic setup failed")
        source = (workspace / task.expected_files[0]).read_text(encoding="utf-8")
        fixture = DiagnosticFixture(
            "E04",
            "C",
            task.expected_files[0],
            task.prompt,
            source,
            (("advance", "tick"), ("ordinary", "maximum", "max")),
            (("only maximum", "equal maximum"),),
            "foundation",
            _line_containing(source, task.setup[0].replacement),
        )
        l1 = _l1(profile, model, fixture)
        l2 = _foundation_edit(profile, model, fixture, task, repository, False)
        l3 = _foundation_edit(profile, model, fixture, task, repository, True)
        return l1, l2, l3


def _foundation_edit(
    profile: str,
    model: Model,
    fixture: DiagnosticFixture,
    task: RealWorldTask,
    repository: Path,
    l3: bool,
) -> ProtocolLayerResult:
    layer = DiagnosticLayer.L3_PROTOCOL if l3 else DiagnosticLayer.L2_EDIT
    request = ModelRequest(
        (
            Message(
                MessageRole.USER,
                _base_prompt(fixture)
                + "\n"
                + (
                    "Return the structured_edit action required by the supplied schema."
                    if l3
                    else "Return only old_text and new_text using the supplied "
                    "JSON schema."
                ),
            ),
        ),
        DIAGNOSTIC_GENERATION,
        build_mutation_ready_output((fixture.path,)) if l3 else L2_OUTPUT,
    )
    started = time.perf_counter()
    try:
        response = model.generate(request)
    except Exception:
        return _generation_failure(profile, "E04", layer.value, started)
    generation_latency = time.perf_counter() - started
    proposal, status = _proposal(response.text, fixture.path, l3=l3)
    if proposal is None:
        return _result(
            profile,
            "E04",
            layer.value,
            status,
            False,
            False,
            EvaluationOutcome.NOT_RUN,
            generation_latency,
            response.usage,
        )
    with tempfile.TemporaryDirectory(prefix="forge-protocol-foundation-") as name:
        workspace = copy_repository(repository, Path(name).resolve() / "workspace")
        apply_task_setup(workspace, task.setup)
        if run_oracle(workspace, task.setup_commands) is EvaluationOutcome.FAIL:
            raise RuntimeError("Foundation diagnostic setup failed")
        path = workspace / fixture.path
        candidate = MutationCandidate(
            fixture.path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            0,
            "evaluator-source",
        )
        validation = validate_structured_edit(proposal, (candidate,), workspace, 0)
        oracle = EvaluationOutcome.NOT_RUN
        if validation.valid:
            path.write_text(
                fixture.source.replace(proposal.old_text, proposal.new_text, 1)
            )
            oracle = run_oracle(workspace, task.oracle_commands)
        return _result(
            profile,
            "E04",
            layer.value,
            _validation_status(
                validation.failure.value if validation.failure else None
            ),
            validation.valid,
            validation.valid,
            oracle,
            generation_latency,
            response.usage,
        )


def _l4(profile: str, result: RealWorldTaskResult) -> ProtocolLayerResult:
    metrics = result.metrics
    return ProtocolLayerResult(
        profile,
        result.task_id,
        "L4",
        result.seed,
        GenerationStatus.GENERATED.value,
        RepresentationStatus.FULL_FORGE.value,
        metrics.structured_mutation_valid > 0,
        metrics.actual_delta_proposed,
        result.oracle.value,
        result.elapsed_seconds,
        result.usage.input_tokens,
        result.usage.output_tokens,
    )


def _base_prompt(fixture: DiagnosticFixture) -> str:
    return (
        f"Coding task:\n{fixture.task}\n\nCurrent path:\n{fixture.path}"
        f"\n\nCurrent source:\n{fixture.source}"
    )


def _line_containing(source: str, text: str) -> int:
    offset = source.find(text)
    if offset < 0:
        raise ValueError("expected diagnostic source region was not found")
    return source.count("\n", 0, offset) + 1


def _validation_status(failure: str | None) -> RepresentationStatus:
    if failure is None:
        return RepresentationStatus.VALID_DELTA
    if failure in {"no_op_edit", "materialized_no_delta"}:
        return RepresentationStatus.NO_OP
    if failure == "path_not_eligible":
        return RepresentationStatus.PATH_MISMATCH
    if failure == "old_text_not_found":
        return RepresentationStatus.OLD_TEXT_MISMATCH
    return RepresentationStatus.INVALID_DELTA


def _generation_failure(
    profile: str, task: str, layer: str, started: float
) -> ProtocolLayerResult:
    return ProtocolLayerResult(
        profile,
        task,
        layer,
        42,
        GenerationStatus.FAILED.value,
        RepresentationStatus.NOT_APPLICABLE.value,
        False,
        False,
        EvaluationOutcome.NOT_RUN.value,
        time.perf_counter() - started,
        0,
        0,
    )


def _result(
    profile: str,
    task: str,
    layer: str,
    status: RepresentationStatus,
    valid: bool,
    delta: bool,
    oracle: EvaluationOutcome,
    latency_seconds: float,
    usage: ModelUsage,
) -> ProtocolLayerResult:
    return ProtocolLayerResult(
        profile,
        task,
        layer,
        42,
        GenerationStatus.GENERATED.value,
        status.value,
        valid,
        delta,
        oracle.value,
        latency_seconds,
        usage.input_tokens,
        usage.output_tokens,
    )

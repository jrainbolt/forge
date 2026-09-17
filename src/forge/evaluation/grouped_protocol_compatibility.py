"""A43 evaluation of multi-file reasoning and grouped protocol compatibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.conversation import ConservativeTokenEstimator
from forge.evaluation.edit_representation import materialize_line_range
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldTask,
    RealWorldTaskResult,
    copy_repository,
    run_oracle,
)
from forge.models import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    ModelResponse,
    MutationRepresentationPolicy,
    OutputSpecification,
    ResponseFormat,
)
from forge.orchestration import (
    LineRangeEditProposal,
    MultiFileLineRangeEditProposal,
    MutationCandidate,
    ToolCallOutcome,
    parse_model_output,
    validate_line_range_edit,
    validate_multi_file_line_range_edit,
)
from forge.orchestration.protocol import build_mutation_ready_output

GROUPED_PROTOCOL_COMPATIBILITY_V1 = "grouped-protocol-compatibility-v1"
GROUPED_PROTOCOL_PROMPT_VERSION = "grouped-protocol-diagnostic-v1"
GROUPED_PROTOCOL_SUITE_VERSION = 1
GROUPED_PROTOCOL_SCHEMA_VERSION = 1
GROUPED_GENERATION = GenerationConfig(max_tokens=512, temperature=0.0, seed=42)

# Intentionally the production function object, not a copied schema.
PRODUCTION_GROUPED_OUTPUT_BUILDER = build_mutation_ready_output


class GroupedProtocolLayer(Enum):
    G1_MULTI_FILE_REASONING = "G1"
    G2_INDEPENDENT_EDITS = "G2"
    G3_MINIMAL_GROUPED_EDIT = "G3"
    G4_PRODUCTION_GROUPED_SCHEMA = "G4"
    G5_FULL_FORGE = "G5"


class GroupedResponseClass(Enum):
    MULTI_FILE_LINE_RANGE_EDIT = "MULTI_FILE_LINE_RANGE_EDIT"
    FINAL = "FINAL"
    SINGLE_FILE_EDIT = "SINGLE_FILE_EDIT"
    TOOL_CALL = "TOOL_CALL"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    EMPTY = "EMPTY"
    GENERATION_ERROR = "GENERATION_ERROR"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    OTHER = "OTHER"


class GroupedFailure(Enum):
    CONCEPT_MISSING = "CONCEPT_MISSING"
    EDIT_CONSTRUCTION_FAILED = "EDIT_CONSTRUCTION_FAILED"
    GROUP_CONSTRUCTION_FAILED = "GROUP_CONSTRUCTION_FAILED"
    PRODUCTION_SCHEMA_INVALID = "PRODUCTION_SCHEMA_INVALID"
    PREMATURE_FINAL = "PREMATURE_FINAL"
    INCOMPLETE_GROUPED_MUTATION = "INCOMPLETE_GROUPED_MUTATION"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    SEMANTIC_FAILURE = "SEMANTIC_FAILURE"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class GroupedFixture:
    task_id: str
    language: str
    task: str
    implementation_path: str
    implementation_source: str
    test_path: str
    test_source: str
    implementation_concepts: tuple[tuple[str, ...], ...]
    test_concepts: tuple[tuple[str, ...], ...]
    contradictions: tuple[tuple[str, ...], ...]
    implementation_semantics: tuple[tuple[str, ...], ...]
    test_semantics: tuple[tuple[str, ...], ...]

    @property
    def paths(self) -> tuple[str, str]:
        return tuple(sorted((self.implementation_path, self.test_path)))  # type: ignore[return-value]

    def source(self, path: str) -> str:
        if path == self.implementation_path:
            return self.implementation_source
        if path == self.test_path:
            return self.test_source
        raise KeyError(path)


@dataclass(frozen=True, slots=True)
class GroupedProtocolResult:
    model_profile: str
    task_id: str
    language: str
    layer: str
    seed: int
    response_class: str
    failure: str
    schema_valid: bool
    concept_implementation: bool | None = None
    concept_test: bool | None = None
    concept_overall: bool | None = None
    implementation_target_valid: bool | None = None
    implementation_delta: bool | None = None
    implementation_semantic: bool | None = None
    test_target_valid: bool | None = None
    test_delta: bool | None = None
    test_semantic: bool | None = None
    path_set_valid: bool | None = None
    duplicate_path: bool | None = None
    edit_count: int = 0
    range_validity: tuple[bool, ...] = ()
    material_deltas: tuple[bool, ...] = ()
    replacement_chars: tuple[int, ...] = ()
    oracle: str = "NOT_RUN"
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = 0.0
    output_chars: int = 0
    generation_status: str = "generated"
    group_validation_valid: bool | None = None
    validation_failure: str | None = None
    grouped_readiness: bool | None = None
    first_response_class: str | None = None
    correction_category: str | None = None
    second_response_class: str | None = None
    preview_created: bool | None = None
    transaction_executed: bool | None = None
    verification_executed: bool | None = None
    verification_result: str | None = None
    forge_status: str | None = None
    tool_count: int | None = None
    model_calls: int | None = None
    context_peak: int | None = None
    total_elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class GroupedProtocolAggregate:
    model_profile: str
    g1_total: int
    g1_concepts: int
    g2_total: int
    g2_implementation_valid: int
    g2_test_valid: int
    g2_oracle_passes: int
    g3_total: int
    g3_valid: int
    g3_oracle_passes: int
    g4_total: int
    g4_valid: int
    g4_oracle_passes: int
    g5_total: int
    g5_ready: int
    g5_proposals: int
    g5_previews: int
    g5_oracle_passes: int


@dataclass(frozen=True, slots=True)
class GroupedProtocolRun:
    suite: str
    suite_version: int
    schema_version: int
    prompt_version: str
    generation: GenerationConfig
    context_capacity: int
    results: tuple[GroupedProtocolResult, ...]
    aggregates: tuple[GroupedProtocolAggregate, ...]
    canonical_unchanged: bool = True


@dataclass(frozen=True, slots=True)
class ModelExchange:
    request: ModelRequest
    response: ModelResponse
    latency_seconds: float


class RecordingModel(Model):
    """Record diagnostic requests without altering the wrapped model."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self.exchanges: list[ModelExchange] = []

    @property
    def identity(self):  # type: ignore[no-untyped-def]
        return self._model.identity

    @property
    def capabilities(self):  # type: ignore[no-untyped-def]
        return self._model.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self._model.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        response = self._model.generate(request)
        self.exchanges.append(
            ModelExchange(request, response, time.perf_counter() - started)
        )
        return response

    def close(self) -> None:
        self._model.close()


SYNTHETIC_GROUPED_FIXTURES = (
    GroupedFixture(
        "P01",
        "C17",
        "Allow a value equal to the limit and add a regression assertion for the "
        "equality boundary.",
        "boundary.c",
        "#include <stdbool.h>\n\nbool allowed(int value, int limit)\n{\n"
        "    return value > limit;\n}\n",
        "test_boundary.c",
        "#include <assert.h>\n#include <stdbool.h>\n\n"
        "bool allowed(int value, int limit);\n\nint main(void)\n{\n"
        "    assert(allowed(6, 5));\n    assert(!allowed(4, 5));\n"
        "    return 0;\n}\n",
        (("equal", "equality", "boundary"), (">=", "inclusive", "allow")),
        (("assert", "check", "test"), ("equal", "equality", "boundary")),
        (("exclude equality", "keep >"),),
        ((">=",),),
        (("assert",), ("allowed(5, 5)", "allowed (5, 5)")),
    ),
    GroupedFixture(
        "P02",
        "Python",
        "Cap an exact-limit sum at the limit and add a regression assertion for "
        "that exact boundary.",
        "calculator.py",
        "def capped_add(a: int, b: int, limit: int) -> int:\n"
        "    return min(a + b, limit - 1)\n",
        "test_calculator.py",
        "from calculator import capped_add\n\n"
        "assert capped_add(2, 3, 10) == 5\n"
        "assert capped_add(8, 7, 10) < 10\n",
        (("limit", "boundary", "exact"), ("min", "cap", "limit")),
        (("assert", "test"), ("exact", "equal", "boundary", "10")),
        (("limit - 1", "below the limit"),),
        (("min",), ("limit",)),
        (("assert",), ("== 10", "==10")),
    ),
    GroupedFixture(
        "P03",
        "Python",
        "Treat the minimum age itself as eligible and add a regression assertion "
        "for the equality case.",
        "access.py",
        "def eligible(age: int, minimum: int) -> bool:\n    return age > minimum\n",
        "test_access.py",
        "from access import eligible\n\n"
        "assert eligible(19, 18)\nassert not eligible(17, 18)\n",
        (("minimum", "boundary", "equal"), (">=", "inclusive", "eligible")),
        (("assert", "test"), ("equal", "equality", "boundary")),
        (("strictly greater", "keep >"),),
        ((">=",),),
        (("assert",), ("eligible(18, 18)", "eligible (18, 18)")),
    ),
)


G2_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            key: {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "new_text": {"type": "string"},
                },
                "required": ["start_line", "end_line", "new_text"],
                "additionalProperties": False,
            }
            for key in ("implementation", "test")
        },
        "required": ["implementation", "test"],
        "additionalProperties": False,
    },
)

G3_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "start_line", "end_line", "new_text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["edits"],
        "additionalProperties": False,
    },
)


def production_grouped_output(paths: tuple[str, ...]) -> OutputSpecification:
    """Return the exact live A41 grouped LINE_RANGE output specification."""
    return PRODUCTION_GROUPED_OUTPUT_BUILDER(
        paths, representation=MutationRepresentationPolicy.LINE_RANGE
    )


def foundation_e08_fixture(repository: Path, task: RealWorldTask) -> GroupedFixture:
    if task.task_id != "E08" or len(task.required_candidate_paths) != 2:
        raise ValueError("grouped Foundation diagnostic requires unchanged E08")
    paths = tuple(sorted(task.required_candidate_paths))
    sources = {path: (repository / path).read_text(encoding="utf-8") for path in paths}
    return GroupedFixture(
        "E08",
        "C17",
        task.prompt,
        "src/clock.c",
        sources["src/clock.c"],
        "tests/test_solar.c",
        sources["tests/test_solar.c"],
        (
            ("boundary", "maximum", "UINT64_MAX", "overflow"),
            ("advance", "increment", "wrap"),
        ),
        (
            ("test", "assert", "check", "regression"),
            ("boundary", "maximum", "UINT64_MAX", "overflow"),
        ),
        (("allow overflow", "wrap at maximum"),),
        (("UINT64_MAX",), ("advance", "tick")),
        (("UINT64_MAX",), ("CHECK", "assert")),
    )


def run_grouped_diagnostics(
    profile: str,
    model: Model,
    *,
    fixtures: tuple[GroupedFixture, ...] = SYNTHETIC_GROUPED_FIXTURES,
    repository: Path | None = None,
    foundation_task: RealWorldTask | None = None,
) -> tuple[GroupedProtocolResult, ...]:
    results: list[GroupedProtocolResult] = []
    for fixture in fixtures:
        results.extend(run_grouped_fixture(profile, model, fixture))
    if (repository is None) != (foundation_task is None):
        raise ValueError("Foundation repository and task must be supplied together")
    if repository is not None and foundation_task is not None:
        fixture = foundation_e08_fixture(repository, foundation_task)
        results.extend(
            run_grouped_fixture(
                profile,
                model,
                fixture,
                repository=repository,
                task=foundation_task,
            )
        )
    return tuple(results)


def run_grouped_fixture(
    profile: str,
    model: Model,
    fixture: GroupedFixture,
    *,
    repository: Path | None = None,
    task: RealWorldTask | None = None,
) -> tuple[GroupedProtocolResult, ...]:
    return (
        _run_g1(profile, model, fixture),
        _run_edit_layer(
            profile,
            model,
            fixture,
            GroupedProtocolLayer.G2_INDEPENDENT_EDITS,
            repository,
            task,
        ),
        _run_edit_layer(
            profile,
            model,
            fixture,
            GroupedProtocolLayer.G3_MINIMAL_GROUPED_EDIT,
            repository,
            task,
        ),
        _run_edit_layer(
            profile,
            model,
            fixture,
            GroupedProtocolLayer.G4_PRODUCTION_GROUPED_SCHEMA,
            repository,
            task,
        ),
    )


def _run_g1(
    profile: str, model: Model, fixture: GroupedFixture
) -> GroupedProtocolResult:
    request = ModelRequest(
        (
            Message(
                MessageRole.USER,
                _source_prompt(fixture)
                + "\nDescribe briefly what behavior must change in the implementation "
                "and what regression behavior the test should assert. Do not return "
                "a patch, JSON, tool call, or hidden reasoning steps.",
            ),
        ),
        GROUPED_GENERATION,
        OutputSpecification(ResponseFormat.TEXT),
    )
    response, latency = _generate(model, request)
    if response is None:
        return _generation_failure(profile, fixture, "G1", latency)
    lowered = response.text.lower()
    implementation = _concepts_present(lowered, fixture.implementation_concepts)
    test = _concepts_present(lowered, fixture.test_concepts)
    contradiction = any(
        all(term.lower() in lowered for term in group)
        for group in fixture.contradictions
    )
    overall = implementation and test and not contradiction
    input_tokens, output_tokens = _usage(request, response)
    truncated = response.finish_reason is FinishReason.MAX_TOKENS
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        "G1",
        42,
        (
            GroupedResponseClass.OUTPUT_TRUNCATED.value
            if truncated
            else GroupedResponseClass.OTHER.value
        ),
        (
            GroupedFailure.OUTPUT_TRUNCATED.value
            if truncated
            else GroupedFailure.PASS.value
            if overall
            else GroupedFailure.CONCEPT_MISSING.value
        ),
        bool(response.text.strip()),
        implementation,
        test,
        overall,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        output_chars=len(response.text),
    )


def _run_edit_layer(
    profile: str,
    model: Model,
    fixture: GroupedFixture,
    layer: GroupedProtocolLayer,
    repository: Path | None,
    task: RealWorldTask | None,
) -> GroupedProtocolResult:
    if layer is GroupedProtocolLayer.G2_INDEPENDENT_EDITS:
        instruction = (
            "Return two independent minimal line-range edits. The implementation "
            "object applies only to the implementation file and the test object "
            "applies only to the test file."
        )
        output = G2_OUTPUT
    elif layer is GroupedProtocolLayer.G3_MINIMAL_GROUPED_EDIT:
        instruction = (
            "Return one minimal grouped object containing exactly one contiguous "
            "changed line-range edit for each listed path."
        )
        output = G3_OUTPUT
    else:
        instruction = (
            "Return the action required by the supplied production response schema."
        )
        output = production_grouped_output(fixture.paths)
    request = ModelRequest(
        (Message(MessageRole.USER, _source_prompt(fixture) + "\n" + instruction),),
        GROUPED_GENERATION,
        output,
    )
    response, latency = _generate(model, request)
    if response is None:
        return _generation_failure(profile, fixture, layer.value, latency)
    if response.finish_reason is FinishReason.MAX_TOKENS:
        return _truncated_result(
            profile, fixture, layer.value, request, response, latency
        )
    if layer is GroupedProtocolLayer.G2_INDEPENDENT_EDITS:
        return _evaluate_g2(
            profile, fixture, response, request, latency, repository, task
        )
    return _evaluate_group(
        profile,
        fixture,
        layer,
        response,
        request,
        latency,
        repository,
        task,
    )


def _evaluate_g2(
    profile: str,
    fixture: GroupedFixture,
    response: ModelResponse,
    request: ModelRequest,
    latency: float,
    repository: Path | None,
    task: RealWorldTask | None,
) -> GroupedProtocolResult:
    try:
        payload = json.loads(response.text)
        if not isinstance(payload, dict) or set(payload) != {"implementation", "test"}:
            raise ValueError
        raw_implementation = _parse_independent_edit(payload["implementation"])
        raw_test = _parse_independent_edit(payload["test"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return _invalid_edit_result(profile, fixture, "G2", response, request, latency)
    implementation = _materialize_independent(
        fixture.implementation_source, raw_implementation
    )
    test = _materialize_independent(fixture.test_source, raw_test)
    implementation_semantic = implementation[0] and _concepts_present(
        raw_implementation[2], fixture.implementation_semantics
    )
    test_semantic = test[0] and _concepts_present(raw_test[2], fixture.test_semantics)
    oracle = EvaluationOutcome.NOT_RUN
    if implementation[0] and test[0]:
        oracle = _run_combined_oracle(
            fixture,
            implementation[2],
            test[2],
            repository,
            task,
        )
    passed = bool(
        implementation_semantic and test_semantic and oracle is EvaluationOutcome.PASS
    )
    input_tokens, output_tokens = _usage(request, response)
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        "G2",
        42,
        GroupedResponseClass.OTHER.value,
        GroupedFailure.PASS.value if passed else GroupedFailure.SEMANTIC_FAILURE.value,
        True,
        implementation_target_valid=implementation[0],
        implementation_delta=implementation[1],
        implementation_semantic=bool(implementation_semantic),
        test_target_valid=test[0],
        test_delta=test[1],
        test_semantic=bool(test_semantic),
        edit_count=2,
        range_validity=(implementation[0], test[0]),
        material_deltas=(implementation[1], test[1]),
        replacement_chars=(len(raw_implementation[2]), len(raw_test[2])),
        oracle=oracle.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        output_chars=len(response.text),
    )


def _evaluate_group(
    profile: str,
    fixture: GroupedFixture,
    layer: GroupedProtocolLayer,
    response: ModelResponse,
    request: ModelRequest,
    latency: float,
    repository: Path | None,
    task: RealWorldTask | None,
) -> GroupedProtocolResult:
    response_class = classify_grouped_response(response.text)
    if layer is GroupedProtocolLayer.G4_PRODUCTION_GROUPED_SCHEMA:
        try:
            parsed = parse_model_output(response.text)
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.outcome is not ToolCallOutcome.MULTI_FILE_LINE_RANGE_EDIT
        ):
            return _classified_invalid_group(
                profile,
                fixture,
                layer.value,
                response_class,
                response,
                request,
                latency,
            )
        assert parsed.multi_file_line_range_edit is not None
        raw_edits = tuple(dict(item) for item in parsed.multi_file_line_range_edit)
    else:
        try:
            payload = json.loads(response.text)
            if not isinstance(payload, dict) or set(payload) != {"edits"}:
                raise ValueError
            raw = payload["edits"]
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError
            raw_edits = tuple(_parse_group_edit(item) for item in raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return _classified_invalid_group(
                profile,
                fixture,
                layer.value,
                GroupedResponseClass.SCHEMA_INVALID,
                response,
                request,
                latency,
            )
        response_class = GroupedResponseClass.MULTI_FILE_LINE_RANGE_EDIT
    paths = tuple(str(item["path"]) for item in raw_edits)
    duplicate = len(set(paths)) != len(paths)
    path_set_valid = set(paths) == set(fixture.paths) and len(paths) == 2
    with tempfile.TemporaryDirectory(prefix="forge-grouped-validation-") as name:
        workspace = Path(name).resolve()
        _write_fixture(workspace, fixture)
        candidates = _candidates(workspace, fixture)
        proposal = MultiFileLineRangeEditProposal(
            tuple(LineRangeEditProposal(**item) for item in raw_edits)  # type: ignore[arg-type]
        )
        validation = validate_multi_file_line_range_edit(
            proposal, candidates, workspace, 0
        )
        updated: dict[str, str] = {}
        range_validity = []
        deltas = []
        for item in raw_edits:
            path = str(item["path"])
            try:
                source = fixture.source(path)
                child = LineRangeEditProposal(**item)  # type: ignore[arg-type]
                child_validation = validate_line_range_edit(
                    child, candidates, workspace, 0
                )
                if not child_validation.valid:
                    raise ValueError
                changed = materialize_line_range(
                    source,
                    int(item["start_line"]),
                    int(item["end_line"]),
                    str(item["new_text"]),
                )
            except (KeyError, TypeError, ValueError):
                range_validity.append(False)
                deltas.append(False)
            else:
                range_validity.append(True)
                deltas.append(changed != source)
                updated[path] = changed
    oracle = EvaluationOutcome.NOT_RUN
    if validation.valid and set(updated) == set(fixture.paths):
        oracle = _run_combined_oracle(
            fixture,
            updated[fixture.implementation_path],
            updated[fixture.test_path],
            repository,
            task,
        )
    passed = validation.valid and oracle is EvaluationOutcome.PASS
    failure = (
        GroupedFailure.PASS
        if passed
        else GroupedFailure.EDIT_CONSTRUCTION_FAILED
        if not validation.valid
        else GroupedFailure.SEMANTIC_FAILURE
    )
    input_tokens, output_tokens = _usage(request, response)
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        layer.value,
        42,
        response_class.value,
        failure.value,
        True,
        path_set_valid=path_set_valid,
        duplicate_path=duplicate,
        edit_count=len(raw_edits),
        range_validity=tuple(range_validity),
        material_deltas=tuple(deltas),
        replacement_chars=tuple(len(str(item["new_text"])) for item in raw_edits),
        oracle=oracle.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        output_chars=len(response.text),
        group_validation_valid=validation.valid,
        validation_failure=(
            validation.failure.value if validation.failure is not None else None
        ),
    )


def classify_grouped_response(text: str) -> GroupedResponseClass:
    if not isinstance(text, str) or not text.strip():
        return GroupedResponseClass.EMPTY
    try:
        parsed = parse_model_output(text)
    except ValueError:
        return GroupedResponseClass.SCHEMA_INVALID
    mapping = {
        ToolCallOutcome.MULTI_FILE_LINE_RANGE_EDIT: (
            GroupedResponseClass.MULTI_FILE_LINE_RANGE_EDIT
        ),
        ToolCallOutcome.FINAL: GroupedResponseClass.FINAL,
        ToolCallOutcome.LINE_RANGE_EDIT: GroupedResponseClass.SINGLE_FILE_EDIT,
        ToolCallOutcome.STRUCTURED_EDIT: GroupedResponseClass.SINGLE_FILE_EDIT,
        ToolCallOutcome.TOOL_CALL: GroupedResponseClass.TOOL_CALL,
    }
    return mapping.get(parsed.outcome, GroupedResponseClass.OTHER)


def full_forge_grouped_result(
    profile: str,
    result: RealWorldTaskResult,
    exchanges: tuple[ModelExchange, ...],
) -> GroupedProtocolResult:
    """Summarize the actual grouped mutation-ready calls made by full Forge."""
    grouped = tuple(
        exchange
        for exchange in exchanges
        if exchange.request.output.schema is not None
        and "multi_file_line_range_edit" in str(exchange.request.output.schema)
    )
    classes = tuple(
        classify_grouped_response(exchange.response.text) for exchange in grouped
    )
    first = classes[0] if classes else GroupedResponseClass.GENERATION_ERROR
    second = classes[1] if len(classes) > 1 else None
    correction = _correction_category(first) if len(classes) > 1 else None
    chosen = second or first
    chosen_exchange = grouped[-1] if grouped else None
    input_tokens = None
    output_tokens = None
    output_chars = 0
    latency = result.elapsed_seconds
    edit_count = 0
    replacement_chars: tuple[int, ...] = ()
    if chosen_exchange is not None:
        input_tokens, output_tokens = _usage(
            chosen_exchange.request, chosen_exchange.response
        )
        output_chars = len(chosen_exchange.response.text)
        latency = chosen_exchange.latency_seconds
        try:
            parsed = parse_model_output(chosen_exchange.response.text)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.multi_file_line_range_edit is not None:
            edit_count = len(parsed.multi_file_line_range_edit)
            replacement_chars = tuple(
                len(str(item["new_text"])) for item in parsed.multi_file_line_range_edit
            )
    metrics = result.metrics
    proposal = bool(metrics.mutation_proposed)
    preview = bool(metrics.mutation_group_preview_created or metrics.preview_created)
    transaction = metrics.mutations > 0
    verification = bool(metrics.verification_executed)
    if chosen is GroupedResponseClass.FINAL:
        failure = GroupedFailure.PREMATURE_FINAL
    elif chosen is GroupedResponseClass.SINGLE_FILE_EDIT:
        failure = GroupedFailure.INCOMPLETE_GROUPED_MUTATION
    elif chosen is GroupedResponseClass.OUTPUT_TRUNCATED:
        failure = GroupedFailure.OUTPUT_TRUNCATED
    elif proposal and result.oracle is EvaluationOutcome.PASS:
        failure = GroupedFailure.PASS
    elif chosen is GroupedResponseClass.MULTI_FILE_LINE_RANGE_EDIT:
        failure = GroupedFailure.SEMANTIC_FAILURE
    else:
        failure = GroupedFailure.PRODUCTION_SCHEMA_INVALID
    return GroupedProtocolResult(
        profile,
        result.task_id,
        "C17",
        "G5",
        result.seed,
        chosen.value,
        failure.value,
        chosen is GroupedResponseClass.MULTI_FILE_LINE_RANGE_EDIT,
        concept_overall=bool(metrics.mutation_ready_reached),
        edit_count=edit_count,
        replacement_chars=replacement_chars,
        oracle=result.oracle.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        output_chars=output_chars,
        grouped_readiness=bool(metrics.mutation_ready_reached),
        first_response_class=first.value,
        correction_category=correction,
        second_response_class=second.value if second is not None else None,
        preview_created=preview,
        transaction_executed=transaction,
        verification_executed=verification,
        verification_result=metrics.verification_result,
        forge_status=result.final_status,
        tool_count=metrics.tool_executions,
        model_calls=metrics.model_calls,
        context_peak=(
            input_tokens
            if metrics.context_peak_estimate == 0 and input_tokens is not None
            else metrics.context_peak_estimate
        ),
        total_elapsed_seconds=result.elapsed_seconds,
    )


def _correction_category(response_class: GroupedResponseClass) -> str:
    if response_class is GroupedResponseClass.FINAL:
        return "GROUPED_PREMATURE_FINAL"
    if response_class is GroupedResponseClass.SINGLE_FILE_EDIT:
        return "INCOMPLETE_GROUPED_MUTATION"
    if response_class is GroupedResponseClass.SCHEMA_INVALID:
        return "PROTOCOL_SCHEMA"
    return "GROUP_VALIDATION"


def build_grouped_protocol_run(
    results: tuple[GroupedProtocolResult, ...],
    *,
    context_capacity: int = 8192,
    canonical_unchanged: bool = True,
) -> GroupedProtocolRun:
    profiles = tuple(dict.fromkeys(item.model_profile for item in results))
    return GroupedProtocolRun(
        GROUPED_PROTOCOL_COMPATIBILITY_V1,
        GROUPED_PROTOCOL_SUITE_VERSION,
        GROUPED_PROTOCOL_SCHEMA_VERSION,
        GROUPED_PROTOCOL_PROMPT_VERSION,
        GROUPED_GENERATION,
        context_capacity,
        results,
        tuple(aggregate_grouped_results(profile, results) for profile in profiles),
        canonical_unchanged,
    )


def aggregate_grouped_results(
    profile: str, results: tuple[GroupedProtocolResult, ...]
) -> GroupedProtocolAggregate:
    selected = tuple(item for item in results if item.model_profile == profile)
    layers = {
        layer: tuple(item for item in selected if item.layer == layer)
        for layer in ("G1", "G2", "G3", "G4", "G5")
    }
    return GroupedProtocolAggregate(
        profile,
        len(layers["G1"]),
        sum(item.concept_overall is True for item in layers["G1"]),
        len(layers["G2"]),
        sum(item.implementation_target_valid is True for item in layers["G2"]),
        sum(item.test_target_valid is True for item in layers["G2"]),
        sum(item.oracle == "PASS" for item in layers["G2"]),
        len(layers["G3"]),
        sum(item.group_validation_valid is True for item in layers["G3"]),
        sum(item.oracle == "PASS" for item in layers["G3"]),
        len(layers["G4"]),
        sum(item.group_validation_valid is True for item in layers["G4"]),
        sum(item.oracle == "PASS" for item in layers["G4"]),
        len(layers["G5"]),
        sum(item.concept_overall is True for item in layers["G5"]),
        sum(item.edit_count > 0 for item in layers["G5"]),
        sum(item.schema_valid for item in layers["G5"]),
        sum(item.oracle == "PASS" for item in layers["G5"]),
    )


def grouped_protocol_to_dict(run: GroupedProtocolRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_grouped_protocol_json(run: GroupedProtocolRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(grouped_protocol_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_grouped_protocol(run: GroupedProtocolRun) -> str:
    lines = [
        "Model       Task Layer Class                           Valid Oracle Failure"
    ]
    for item in run.results:
        lines.append(
            f"{item.model_profile:<11} {item.task_id:<4} {item.layer:<5} "
            f"{item.response_class:<31} {'yes' if item.schema_valid else 'no':<5} "
            f"{item.oracle:<7} {item.failure}"
        )
    return "\n".join(lines)


def _source_prompt(fixture: GroupedFixture) -> str:
    chunks = [f"Coding task:\n{fixture.task}\n"]
    for path in fixture.paths:
        chunks.append(
            f"\nPATH: {path}\nCurrent source with 1-based line numbers:\n"
            f"{_numbered(fixture.source(path))}END FILE: {path}\n"
        )
    return "".join(chunks)


def _numbered(source: str) -> str:
    return "".join(
        f"{number:>4} | {line}"
        for number, line in enumerate(source.splitlines(keepends=True), 1)
    )


def _concepts_present(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    lowered = text.lower()
    return all(any(term.lower() in lowered for term in group) for group in groups)


def _generate(
    model: Model, request: ModelRequest
) -> tuple[ModelResponse | None, float]:
    started = time.perf_counter()
    try:
        return model.generate(request), time.perf_counter() - started
    except Exception:
        return None, time.perf_counter() - started


def _usage(request: ModelRequest, response: ModelResponse) -> tuple[int, int]:
    estimator = ConservativeTokenEstimator()
    estimated_input = sum(estimator.estimate(message) for message in request.messages)
    estimated_output = estimator.estimate(Message(MessageRole.ASSISTANT, response.text))
    return (
        response.usage.input_tokens
        if response.usage.input_tokens is not None
        else estimated_input,
        response.usage.output_tokens
        if response.usage.output_tokens is not None
        else estimated_output,
    )


def _parse_independent_edit(value: object) -> tuple[int, int, str]:
    if not isinstance(value, dict) or set(value) != {
        "start_line",
        "end_line",
        "new_text",
    }:
        raise TypeError
    start, end, text = value["start_line"], value["end_line"], value["new_text"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(text, str)
    ):
        raise TypeError
    return start, end, text


def _parse_group_edit(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "start_line",
        "end_line",
        "new_text",
    }:
        raise TypeError
    path = value["path"]
    start = value["start_line"]
    end = value["end_line"]
    text = value["new_text"]
    if (
        not isinstance(path, str)
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(text, str)
    ):
        raise TypeError
    return dict(value)


def _materialize_independent(
    source: str, edit: tuple[int, int, str]
) -> tuple[bool, bool, str]:
    try:
        updated = materialize_line_range(source, *edit)
    except ValueError:
        return False, False, source
    return True, updated != source, updated


def _write_fixture(workspace: Path, fixture: GroupedFixture) -> None:
    for path in fixture.paths:
        destination = workspace / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(fixture.source(path), encoding="utf-8")


def _candidates(
    workspace: Path, fixture: GroupedFixture
) -> tuple[MutationCandidate, ...]:
    values = []
    for path in fixture.paths:
        source = (workspace / path).read_bytes()
        values.append(
            MutationCandidate(
                path,
                hashlib.sha256(source).hexdigest(),
                0,
                f"evaluator-{path}",
                1,
                len(source.decode("utf-8").splitlines()),
            )
        )
    return tuple(values)


def _run_combined_oracle(
    fixture: GroupedFixture,
    implementation: str,
    test: str,
    repository: Path | None,
    task: RealWorldTask | None,
) -> EvaluationOutcome:
    with tempfile.TemporaryDirectory(prefix="forge-grouped-oracle-") as name:
        if repository is not None and task is not None:
            workspace = copy_repository(repository, Path(name).resolve() / "workspace")
        else:
            workspace = Path(name).resolve()
        for path, source in (
            (fixture.implementation_path, implementation),
            (fixture.test_path, test),
        ):
            destination = workspace / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source, encoding="utf-8")
        if repository is not None and task is not None:
            return run_oracle(workspace, task.oracle_commands)
        return _synthetic_oracle(fixture, workspace)


def _synthetic_oracle(fixture: GroupedFixture, workspace: Path) -> EvaluationOutcome:
    if fixture.language == "C17":
        executable = workspace / "fixture-test"
        commands = (
            (
                "/usr/bin/cc",
                "-std=c17",
                "-Wall",
                "-Werror",
                fixture.implementation_path,
                fixture.test_path,
                "-o",
                str(executable),
            ),
            (str(executable),),
        )
    else:
        commands = ((sys.executable, fixture.test_path),)
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            return EvaluationOutcome.FAIL
    return EvaluationOutcome.PASS


def _generation_failure(
    profile: str, fixture: GroupedFixture, layer: str, latency: float
) -> GroupedProtocolResult:
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        layer,
        42,
        GroupedResponseClass.GENERATION_ERROR.value,
        GroupedFailure.EDIT_CONSTRUCTION_FAILED.value,
        False,
        latency_seconds=latency,
        generation_status="generation_error",
    )


def _truncated_result(
    profile: str,
    fixture: GroupedFixture,
    layer: str,
    request: ModelRequest,
    response: ModelResponse,
    latency: float,
) -> GroupedProtocolResult:
    inputs, outputs = _usage(request, response)
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        layer,
        42,
        GroupedResponseClass.OUTPUT_TRUNCATED.value,
        GroupedFailure.OUTPUT_TRUNCATED.value,
        False,
        input_tokens=inputs,
        output_tokens=outputs,
        latency_seconds=latency,
        output_chars=len(response.text),
    )


def _invalid_edit_result(
    profile: str,
    fixture: GroupedFixture,
    layer: str,
    response: ModelResponse,
    request: ModelRequest,
    latency: float,
) -> GroupedProtocolResult:
    inputs, outputs = _usage(request, response)
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        layer,
        42,
        GroupedResponseClass.SCHEMA_INVALID.value,
        GroupedFailure.EDIT_CONSTRUCTION_FAILED.value,
        False,
        input_tokens=inputs,
        output_tokens=outputs,
        latency_seconds=latency,
        output_chars=len(response.text),
    )


def _classified_invalid_group(
    profile: str,
    fixture: GroupedFixture,
    layer: str,
    response_class: GroupedResponseClass,
    response: ModelResponse,
    request: ModelRequest,
    latency: float,
) -> GroupedProtocolResult:
    failure = (
        GroupedFailure.PREMATURE_FINAL
        if response_class is GroupedResponseClass.FINAL
        else GroupedFailure.INCOMPLETE_GROUPED_MUTATION
        if response_class is GroupedResponseClass.SINGLE_FILE_EDIT
        else GroupedFailure.PRODUCTION_SCHEMA_INVALID
        if layer == "G4"
        else GroupedFailure.GROUP_CONSTRUCTION_FAILED
    )
    inputs, outputs = _usage(request, response)
    return GroupedProtocolResult(
        profile,
        fixture.task_id,
        fixture.language,
        layer,
        42,
        response_class.value,
        failure.value,
        False,
        input_tokens=inputs,
        output_tokens=outputs,
        latency_seconds=latency,
        output_chars=len(response.text),
    )

"""A44 paired-edit semantic planning and benchmark-integrity evaluation."""

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
from forge.evaluation.grouped_protocol_compatibility import (
    SYNTHETIC_GROUPED_FIXTURES,
    GroupedFixture,
    production_grouped_output,
)
from forge.models import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    ModelResponse,
    OutputSpecification,
    ResponseFormat,
)
from forge.orchestration import (
    LineRangeEditProposal,
    MultiFileLineRangeEditProposal,
    MutationCandidate,
    ToolCallOutcome,
    parse_model_output,
    validate_multi_file_line_range_edit,
)

PAIRED_SEMANTIC_PLANNING_V1 = "paired-semantic-planning-v1"
PAIRED_SEMANTIC_PLANNING_SUITE_VERSION = 1
PAIRED_SEMANTIC_PLANNING_SCHEMA_VERSION = 1
PAIRED_SEMANTIC_PLANNING_PROMPT_VERSION = "paired-semantic-planning-v1"
PLAN_GENERATION = GenerationConfig(max_tokens=256, temperature=0.0, seed=42)
EDIT_GENERATION = GenerationConfig(max_tokens=512, temperature=0.0, seed=42)


class PlanningCondition(Enum):
    D0_DIRECT_GROUPED_EDIT = "D0"
    D1_PLAN_THEN_GROUPED_EDIT = "D1"
    D2_PLAN_EDIT_SELF_CHECK = "D2"


class PlanningFailure(Enum):
    PLAN_SCHEMA_INVALID = "PLAN_SCHEMA_INVALID"
    PLAN_PATH_SET_INVALID = "PLAN_PATH_SET_INVALID"
    PLAN_CONCEPT_MISSING = "PLAN_CONCEPT_MISSING"
    PLAN_CONTRADICTORY = "PLAN_CONTRADICTORY"
    EDIT_SCHEMA_INVALID = "EDIT_SCHEMA_INVALID"
    EDIT_TARGET_INVALID = "EDIT_TARGET_INVALID"
    EDIT_SEMANTIC_FAIL = "EDIT_SEMANTIC_FAIL"
    ORACLE_NON_DISCRIMINATING = "ORACLE_NON_DISCRIMINATING"
    PASS = "PASS"


class SemanticEligibility(Enum):
    ELIGIBLE = "ELIGIBLE"
    NON_DISCRIMINATING = "NON_DISCRIMINATING"
    SEMANTICALLY_UNDER_SPECIFIED = "SEMANTICALLY_UNDER_SPECIFIED"


@dataclass(frozen=True, slots=True)
class PairedSemanticTask:
    task_id: str
    language: str
    task: str
    implementation_path: str
    implementation_source: str
    test_path: str
    test_source: str
    reference_implementation: str
    reference_test: str
    plan_concepts: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]
    contradictions: tuple[tuple[str, ...], ...] = ()

    @property
    def paths(self) -> tuple[str, str]:
        return tuple(sorted((self.implementation_path, self.test_path)))  # type: ignore[return-value]

    def source(self, path: str) -> str:
        if path == self.implementation_path:
            return self.implementation_source
        if path == self.test_path:
            return self.test_source
        raise KeyError(path)

    def reference(self, path: str) -> str:
        if path == self.implementation_path:
            return self.reference_implementation
        if path == self.test_path:
            return self.reference_test
        raise KeyError(path)


@dataclass(frozen=True, slots=True)
class PlanEntry:
    path: str
    behavior_change: str
    preserve: tuple[str, ...]
    test_intent: str


@dataclass(frozen=True, slots=True)
class BoundSemanticPlan:
    task_id: str
    required_paths: tuple[str, ...]
    workspace_generation: int
    source_hashes: tuple[tuple[str, str], ...]
    files: tuple[PlanEntry, ...]


@dataclass(frozen=True, slots=True)
class OracleIntegrity:
    task_id: str
    baseline_oracle_pass: bool
    reference_oracle_pass: bool
    wrong_mutation_oracle_pass: bool
    semantic_oracle_discriminating: bool
    semantic_score_eligible: bool


@dataclass(frozen=True, slots=True)
class FoundationE08Audit:
    task_id: str
    legacy_definition_preserved: bool
    unchanged_baseline_oracle_pass: bool
    semantic_eligibility: str
    versioned_replacement_created: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PairedPlanningResult:
    model_profile: str
    task_id: str
    language: str
    condition: str
    seed: int
    plan_valid: bool | None
    plan_file_count: int
    plan_failure: str | None
    edit_schema_valid: bool
    target_valid_per_file: tuple[bool, ...]
    material_delta_per_file: tuple[bool, ...]
    group_valid: bool
    build_test: str
    oracle: str
    semantic_success: bool
    failure: str
    model_calls: int
    input_tokens: int
    plan_output_tokens: int
    edit_output_tokens: int
    planning_latency_seconds: float
    editing_latency_seconds: float
    total_latency_seconds: float


@dataclass(frozen=True, slots=True)
class PairedPlanningAggregate:
    model_profile: str
    condition: str
    total: int
    plan_valid: int
    structural_success: int
    semantic_success: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class PairedPlanningRun:
    suite: str
    suite_version: int
    schema_version: int
    prompt_version: str
    plan_generation: GenerationConfig
    edit_generation: GenerationConfig
    context_capacity: int
    integrity: tuple[OracleIntegrity, ...]
    results: tuple[PairedPlanningResult, ...]
    aggregates: tuple[PairedPlanningAggregate, ...]


PLAN_OUTPUT = OutputSpecification(
    ResponseFormat.JSON,
    {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "maxLength": 240},
                        "behavior_change": {"type": "string", "maxLength": 240},
                        "preserve": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "string", "maxLength": 160},
                        },
                        "test_intent": {"type": "string", "maxLength": 240},
                    },
                    "required": [
                        "path",
                        "behavior_change",
                        "preserve",
                        "test_intent",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["files"],
        "additionalProperties": False,
    },
)


def _from_a43(fixture: GroupedFixture, reference_impl: str, reference_test: str):  # type: ignore[no-untyped-def]
    return PairedSemanticTask(
        fixture.task_id,
        fixture.language,
        fixture.task,
        fixture.implementation_path,
        fixture.implementation_source,
        fixture.test_path,
        fixture.test_source,
        reference_impl,
        reference_test,
        (
            (fixture.implementation_path, fixture.implementation_concepts),
            (fixture.test_path, fixture.test_concepts),
        ),
        fixture.contradictions,
    )


PAIRED_SEMANTIC_TASKS = (
    _from_a43(
        SYNTHETIC_GROUPED_FIXTURES[0],
        "#include <stdbool.h>\n\nbool allowed(int value, int limit)\n{\n"
        "    return value >= limit;\n}\n",
        "#include <assert.h>\n#include <stdbool.h>\n\n"
        "bool allowed(int value, int limit);\n\n"
        "int main(void)\n{\n    assert(allowed(6, 5));\n"
        "    assert(allowed(5, 5));\n    assert(!allowed(4, 5));\n    return 0;\n}\n",
    ),
    _from_a43(
        SYNTHETIC_GROUPED_FIXTURES[1],
        "def capped_add(a: int, b: int, limit: int) -> int:\n"
        "    return min(a + b, limit)\n",
        "from calculator import capped_add\n\nassert capped_add(2, 3, 10) == 5\n"
        "assert capped_add(8, 7, 10) == 10\n"
        "assert capped_add(4, 6, 10) == 10\n",
    ),
    _from_a43(
        SYNTHETIC_GROUPED_FIXTURES[2],
        "def eligible(age: int, minimum: int) -> bool:\n    return age >= minimum\n",
        "from access import eligible\n\nassert eligible(19, 18)\n"
        "assert eligible(18, 18)\nassert not eligible(17, 18)\n",
    ),
    PairedSemanticTask(
        "P04",
        "Python",
        "Permit the final retry when the failure count equals max_retries, while "
        "still rejecting later attempts, and add an exact-boundary regression test.",
        "retry_policy.py",
        "def should_retry(failures: int, max_retries: int) -> bool:\n"
        '    if failures < 0 or max_retries < 0:\n        raise ValueError("counts")\n'
        "    return failures < max_retries\n",
        "test_retry_policy.py",
        "from retry_policy import should_retry\n\nassert should_retry(1, 3)\n"
        "assert not should_retry(4, 3)\n",
        "def should_retry(failures: int, max_retries: int) -> bool:\n"
        '    if failures < 0 or max_retries < 0:\n        raise ValueError("counts")\n'
        "    return failures <= max_retries\n",
        "from retry_policy import should_retry\n\nassert should_retry(1, 3)\n"
        "assert should_retry(3, 3)\nassert not should_retry(4, 3)\n",
        (
            ("retry_policy.py", (("equal", "boundary", "final"), ("<=", "inclusive"))),
            ("test_retry_policy.py", (("test", "assert"), ("equal", "boundary", "3"))),
        ),
        (("keep <", "exclude equality"),),
    ),
    PairedSemanticTask(
        "P05",
        "C17",
        "Accept parser tokens at the documented maximum length of eight characters "
        "and add a regression assertion at exactly that boundary.",
        "token.c",
        "#include <stdbool.h>\n#include <stddef.h>\n\n"
        "bool token_length_valid(size_t length)\n"
        "{\n    return length >= 3 && length < 8;\n}\n",
        "test_token.c",
        "#include <assert.h>\n#include <stdbool.h>\n#include <stddef.h>\n\n"
        "bool token_length_valid(size_t length);\n\nint main(void)\n{\n"
        "    assert(token_length_valid(3));\n    assert(!token_length_valid(9));\n"
        "    return 0;\n}\n",
        "#include <stdbool.h>\n#include <stddef.h>\n\n"
        "bool token_length_valid(size_t length)\n"
        "{\n    return length >= 3 && length <= 8;\n}\n",
        "#include <assert.h>\n#include <stdbool.h>\n#include <stddef.h>\n\n"
        "bool token_length_valid(size_t length);\n\nint main(void)\n{\n"
        "    assert(token_length_valid(3));\n    assert(token_length_valid(8));\n"
        "    assert(!token_length_valid(9));\n    return 0;\n}\n",
        (
            ("token.c", (("maximum", "boundary", "eight", "8"), ("<=", "inclusive"))),
            ("test_token.c", (("test", "assert"), ("eight", "8", "boundary"))),
        ),
        (("keep < 8", "exclude eight"),),
    ),
    PairedSemanticTask(
        "P06",
        "Python",
        "Accept a batch size at the documented minimum of one, preserve the maximum "
        "of 100, and add a regression assertion for the minimum boundary.",
        "validation.py",
        "def valid_batch_size(size: int) -> bool:\n"
        "    return size > 1 and size <= 100\n",
        "test_validation.py",
        "from validation import valid_batch_size\n\nassert valid_batch_size(2)\n"
        "assert valid_batch_size(100)\nassert not valid_batch_size(101)\n",
        "def valid_batch_size(size: int) -> bool:\n"
        "    return size >= 1 and size <= 100\n",
        "from validation import valid_batch_size\n\nassert valid_batch_size(1)\n"
        "assert valid_batch_size(2)\nassert valid_batch_size(100)\n"
        "assert not valid_batch_size(101)\n",
        (
            (
                "validation.py",
                (("minimum", "boundary", "one", "1"), (">=", "inclusive")),
            ),
            ("test_validation.py", (("test", "assert"), ("minimum", "one", "1"))),
        ),
        (("keep > 1", "exclude one"),),
    ),
)


def source_hashes(task: PairedSemanticTask) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, hashlib.sha256(task.source(path).encode()).hexdigest())
        for path in task.paths
    )


def validate_plan_response(
    text: str,
    task: PairedSemanticTask,
    *,
    workspace_generation: int = 0,
) -> tuple[BoundSemanticPlan | None, PlanningFailure | None]:
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict) or set(payload) != {"files"}:
            raise TypeError
        raw_files = payload["files"]
        if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= 4:
            raise TypeError
        entries = tuple(_parse_plan_entry(value) for value in raw_files)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, PlanningFailure.PLAN_SCHEMA_INVALID
    paths = tuple(entry.path for entry in entries)
    if len(set(paths)) != len(paths) or set(paths) != set(task.paths):
        return None, PlanningFailure.PLAN_PATH_SET_INVALID
    by_path = {entry.path: entry for entry in entries}
    for path, groups in task.plan_concepts:
        entry = by_path[path]
        text_value = " ".join(
            (entry.behavior_change, *entry.preserve, entry.test_intent)
        ).lower()
        if not _concepts_present(text_value, groups):
            return None, PlanningFailure.PLAN_CONCEPT_MISSING
        if any(
            all(term.lower() in text_value for term in group)
            for group in task.contradictions
        ):
            return None, PlanningFailure.PLAN_CONTRADICTORY
    return (
        BoundSemanticPlan(
            task.task_id,
            task.paths,
            workspace_generation,
            source_hashes(task),
            tuple(sorted(entries, key=lambda entry: entry.path)),
        ),
        None,
    )


def plan_is_current(
    plan: BoundSemanticPlan,
    task: PairedSemanticTask,
    *,
    workspace_generation: int,
) -> bool:
    return (
        plan.task_id == task.task_id
        and plan.required_paths == task.paths
        and plan.workspace_generation == workspace_generation
        and plan.source_hashes == source_hashes(task)
    )


def audit_oracle_integrity(task: PairedSemanticTask) -> OracleIntegrity:
    baseline = run_semantic_oracle(task, task.implementation_source, task.test_source)
    reference = run_semantic_oracle(
        task, task.reference_implementation, task.reference_test
    )
    wrong = run_semantic_oracle(task, task.implementation_source, task.reference_test)
    discriminating = not baseline and reference
    return OracleIntegrity(
        task.task_id,
        baseline,
        reference,
        wrong,
        discriminating,
        discriminating,
    )


def classify_foundation_e08(
    *,
    unchanged_baseline_oracle_pass: bool,
) -> FoundationE08Audit:
    """Classify legacy E08 without inventing an evaluator-only intended change."""
    return FoundationE08Audit(
        "E08",
        True,
        unchanged_baseline_oracle_pass,
        SemanticEligibility.SEMANTICALLY_UNDER_SPECIFIED.value,
        False,
        "The task names neither the boundary nor its intended behavior, and the "
        "legacy configure/build/test oracle passes the unchanged repository.",
    )


def run_semantic_oracle(
    task: PairedSemanticTask, implementation: str, test: str
) -> bool:
    with tempfile.TemporaryDirectory(prefix="forge-paired-semantic-") as name:
        workspace = Path(name).resolve()
        (workspace / task.implementation_path).write_text(
            implementation, encoding="utf-8"
        )
        (workspace / task.test_path).write_text(test, encoding="utf-8")
        if not _test_intent_present(task, test):
            return False
        if not _submitted_build_test(task, workspace):
            return False
        if task.language == "C17":
            hidden = workspace / "hidden.c"
            hidden.write_text(_hidden_c_oracle(task.task_id), encoding="utf-8")
            hidden_executable = workspace / "hidden-test"
            return _command_ok(
                (
                    "/usr/bin/cc",
                    "-std=c17",
                    "-Wall",
                    "-Werror",
                    task.implementation_path,
                    hidden.name,
                    "-o",
                    str(hidden_executable),
                ),
                workspace,
            ) and _command_ok((str(hidden_executable),), workspace)
        return _command_ok(
            (sys.executable, "-c", _hidden_python_oracle(task.task_id)), workspace
        )


def run_submitted_build_test(
    task: PairedSemanticTask, implementation: str, test: str
) -> bool:
    """Run the submitted implementation/test pair without the hidden oracle."""
    with tempfile.TemporaryDirectory(prefix="forge-paired-build-test-") as name:
        workspace = Path(name).resolve()
        (workspace / task.implementation_path).write_text(
            implementation, encoding="utf-8"
        )
        (workspace / task.test_path).write_text(test, encoding="utf-8")
        return _submitted_build_test(task, workspace)


def _submitted_build_test(task: PairedSemanticTask, workspace: Path) -> bool:
    if task.language != "C17":
        return _command_ok((sys.executable, task.test_path), workspace)
    executable = workspace / "submitted-test"
    return _command_ok(
        (
            "/usr/bin/cc",
            "-std=c17",
            "-Wall",
            "-Werror",
            task.implementation_path,
            task.test_path,
            "-o",
            str(executable),
        ),
        workspace,
    ) and _command_ok((str(executable),), workspace)


def run_planning_condition(
    profile: str,
    model: Model,
    task: PairedSemanticTask,
    condition: PlanningCondition,
) -> PairedPlanningResult:
    if condition is PlanningCondition.D2_PLAN_EDIT_SELF_CHECK:
        raise ValueError("D2 is diagnostic-only and not enabled in v1")
    integrity = audit_oracle_integrity(task)
    plan: BoundSemanticPlan | None = None
    plan_failure: PlanningFailure | None = None
    plan_response: ModelResponse | None = None
    plan_latency = 0.0
    input_tokens = 0
    if condition is PlanningCondition.D1_PLAN_THEN_GROUPED_EDIT:
        plan_request = ModelRequest(
            (Message(MessageRole.USER, _plan_prompt(task)),),
            PLAN_GENERATION,
            PLAN_OUTPUT,
        )
        plan_response, plan_latency = _generate(model, plan_request)
        if (
            plan_response is None
            or plan_response.finish_reason is FinishReason.MAX_TOKENS
        ):
            plan_failure = PlanningFailure.PLAN_SCHEMA_INVALID
        else:
            plan, plan_failure = validate_plan_response(plan_response.text, task)
        plan_input, plan_output = _usage(plan_request, plan_response)
        input_tokens += plan_input
        if plan is None:
            return PairedPlanningResult(
                profile,
                task.task_id,
                task.language,
                condition.value,
                42,
                False,
                _plan_file_count(plan_response),
                plan_failure.value if plan_failure else None,
                False,
                (),
                (),
                False,
                "NOT_RUN",
                "NOT_RUN",
                False,
                (plan_failure or PlanningFailure.PLAN_SCHEMA_INVALID).value,
                1,
                input_tokens,
                plan_output,
                0,
                plan_latency,
                0.0,
                plan_latency,
            )
    edit_request = ModelRequest(
        (Message(MessageRole.USER, _edit_prompt(task, plan)),),
        EDIT_GENERATION,
        production_grouped_output(task.paths),
    )
    edit_response, edit_latency = _generate(model, edit_request)
    edit_input, edit_output = _usage(edit_request, edit_response)
    input_tokens += edit_input
    evaluation = _evaluate_edit(task, edit_response)
    semantic = bool(
        evaluation[0] and integrity.semantic_score_eligible and evaluation[5]
    )
    failure = (
        PlanningFailure.ORACLE_NON_DISCRIMINATING
        if not integrity.semantic_score_eligible
        else PlanningFailure.EDIT_SCHEMA_INVALID
        if not evaluation[0]
        else PlanningFailure.EDIT_TARGET_INVALID
        if not evaluation[3]
        else PlanningFailure.PASS
        if semantic
        else PlanningFailure.EDIT_SEMANTIC_FAIL
    )
    plan_output_tokens = 0 if plan_response is None else _usage(None, plan_response)[1]
    return PairedPlanningResult(
        profile,
        task.task_id,
        task.language,
        condition.value,
        42,
        None if condition is PlanningCondition.D0_DIRECT_GROUPED_EDIT else True,
        0 if plan is None else len(plan.files),
        None,
        evaluation[0],
        evaluation[1],
        evaluation[2],
        evaluation[3],
        "PASS" if evaluation[4] else "FAIL" if evaluation[3] else "NOT_RUN",
        "PASS" if evaluation[5] else "FAIL" if evaluation[3] else "NOT_RUN",
        semantic,
        failure.value,
        1 if plan is None else 2,
        input_tokens,
        plan_output_tokens,
        edit_output,
        plan_latency,
        edit_latency,
        plan_latency + edit_latency,
    )


def run_paired_semantic_planning(
    profile: str,
    model: Model,
    *,
    tasks: tuple[PairedSemanticTask, ...] = PAIRED_SEMANTIC_TASKS,
) -> tuple[PairedPlanningResult, ...]:
    results = []
    for task in tasks:
        results.append(
            run_planning_condition(
                profile, model, task, PlanningCondition.D0_DIRECT_GROUPED_EDIT
            )
        )
        results.append(
            run_planning_condition(
                profile, model, task, PlanningCondition.D1_PLAN_THEN_GROUPED_EDIT
            )
        )
    return tuple(results)


def build_paired_planning_run(
    results: tuple[PairedPlanningResult, ...],
    *,
    tasks: tuple[PairedSemanticTask, ...] = PAIRED_SEMANTIC_TASKS,
    context_capacity: int = 8192,
) -> PairedPlanningRun:
    integrity = tuple(audit_oracle_integrity(task) for task in tasks)
    profiles = tuple(dict.fromkeys(result.model_profile for result in results))
    aggregates = tuple(
        aggregate_paired_results(profile, condition, results)
        for profile in profiles
        for condition in ("D0", "D1")
    )
    return PairedPlanningRun(
        PAIRED_SEMANTIC_PLANNING_V1,
        PAIRED_SEMANTIC_PLANNING_SUITE_VERSION,
        PAIRED_SEMANTIC_PLANNING_SCHEMA_VERSION,
        PAIRED_SEMANTIC_PLANNING_PROMPT_VERSION,
        PLAN_GENERATION,
        EDIT_GENERATION,
        context_capacity,
        integrity,
        results,
        aggregates,
    )


def aggregate_paired_results(
    profile: str,
    condition: str,
    results: tuple[PairedPlanningResult, ...],
) -> PairedPlanningAggregate:
    selected = tuple(
        result
        for result in results
        if result.model_profile == profile and result.condition == condition
    )
    return PairedPlanningAggregate(
        profile,
        condition,
        len(selected),
        sum(result.plan_valid is True for result in selected),
        sum(result.group_valid for result in selected),
        sum(result.semantic_success for result in selected),
        sum(result.model_calls for result in selected),
        sum(result.input_tokens for result in selected),
        sum(
            result.plan_output_tokens + result.edit_output_tokens for result in selected
        ),
        sum(result.total_latency_seconds for result in selected),
    )


def paired_planning_to_dict(run: PairedPlanningRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_paired_planning_json(run: PairedPlanningRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(paired_planning_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_paired_planning(run: PairedPlanningRun) -> str:
    lines = ["Model       Task Condition Plan  Group Oracle Failure"]
    for result in run.results:
        plan = (
            "n/a" if result.plan_valid is None else "yes" if result.plan_valid else "no"
        )
        lines.append(
            f"{result.model_profile:<11} {result.task_id:<4} {result.condition:<9} "
            f"{plan:<5} {'yes' if result.group_valid else 'no':<5} "
            f"{result.oracle:<6} {result.failure}"
        )
    return "\n".join(lines)


def _parse_plan_entry(value: object) -> PlanEntry:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "behavior_change",
        "preserve",
        "test_intent",
    }:
        raise TypeError
    path, behavior, preserve, test_intent = (
        value["path"],
        value["behavior_change"],
        value["preserve"],
        value["test_intent"],
    )
    if (
        not isinstance(path, str)
        or not isinstance(behavior, str)
        or not isinstance(test_intent, str)
    ):
        raise TypeError
    if (
        not isinstance(preserve, list)
        or len(preserve) > 3
        or not all(isinstance(item, str) for item in preserve)
    ):
        raise TypeError
    if (
        not path
        or not behavior
        or not test_intent
        or len(path) > 240
        or len(behavior) > 240
        or len(test_intent) > 240
        or any(len(item) > 160 for item in preserve)
    ):
        raise ValueError
    return PlanEntry(path, behavior, tuple(preserve), test_intent)


def _plan_prompt(task: PairedSemanticTask) -> str:
    return (
        _source_prompt(task)
        + "\nReturn only a concise externally visible change contract for each "
        "required "
        "file using the supplied JSON schema. Describe behavior, preservation "
        "constraints, and test intent. Do not include source replacement text, "
        "line numbers, patches, executable instructions, or internal deliberation. "
        "The plan grants no file or mutation authority.\n"
    )


def _edit_prompt(task: PairedSemanticTask, plan: BoundSemanticPlan | None) -> str:
    prompt = _source_prompt(task)
    if plan is not None:
        visible = {"files": [asdict(entry) for entry in plan.files]}
        prompt += (
            "\nAccepted non-authoritative change plan:\n"
            + json.dumps(visible, sort_keys=True)
            + "\n"
        )
    return (
        prompt
        + "\nReturn the complete grouped LINE_RANGE action required by the supplied "
        "production response schema. Include exactly one material edit for every "
        "authorized path.\n"
    )


def _source_prompt(task: PairedSemanticTask) -> str:
    chunks = [f"Coding task:\n{task.task}\n"]
    for path in task.paths:
        numbered = "".join(
            f"{number:>4} | {line}"
            for number, line in enumerate(
                task.source(path).splitlines(keepends=True), 1
            )
        )
        chunks.append(
            f"\nPATH: {path}\nCurrent source with 1-based line numbers:\n"
            f"{numbered}END FILE: {path}\n"
        )
    return "".join(chunks)


def _evaluate_edit(task: PairedSemanticTask, response: ModelResponse | None):  # type: ignore[no-untyped-def]
    if response is None or response.finish_reason is FinishReason.MAX_TOKENS:
        return False, (), (), False, False, False
    try:
        parsed = parse_model_output(response.text)
    except ValueError:
        return False, (), (), False, False, False
    if (
        parsed.outcome is not ToolCallOutcome.MULTI_FILE_LINE_RANGE_EDIT
        or parsed.multi_file_line_range_edit is None
    ):
        return False, (), (), False, False, False
    raw_edits = tuple(dict(item) for item in parsed.multi_file_line_range_edit)
    with tempfile.TemporaryDirectory(prefix="forge-paired-validation-") as name:
        workspace = Path(name).resolve()
        for path in task.paths:
            (workspace / path).write_text(task.source(path), encoding="utf-8")
        candidates = tuple(
            MutationCandidate(
                path,
                hashlib.sha256(task.source(path).encode()).hexdigest(),
                0,
                f"a44-{path}",
                1,
                len(task.source(path).splitlines()),
            )
            for path in task.paths
        )
        try:
            proposal = MultiFileLineRangeEditProposal(
                tuple(LineRangeEditProposal(**item) for item in raw_edits)
            )  # type: ignore[arg-type]
            validation = validate_multi_file_line_range_edit(
                proposal, candidates, workspace, 0
            )
        except (TypeError, ValueError):
            return True, (), (), False, False, False
    updated: dict[str, str] = {}
    target_valid = []
    deltas = []
    for item in raw_edits:
        path = item.get("path")
        try:
            source = task.source(str(path))
            changed = materialize_line_range(
                source,
                int(item["start_line"]),
                int(item["end_line"]),
                str(item["new_text"]),
            )
        except (KeyError, TypeError, ValueError):
            target_valid.append(False)
            deltas.append(False)
        else:
            target_valid.append(True)
            deltas.append(changed != source)
            updated[str(path)] = changed
    group_valid = validation.valid and set(updated) == set(task.paths) and all(deltas)
    build_test = group_valid and run_submitted_build_test(
        task, updated[task.implementation_path], updated[task.test_path]
    )
    oracle = build_test and run_semantic_oracle(
        task, updated[task.implementation_path], updated[task.test_path]
    )
    return True, tuple(target_valid), tuple(deltas), group_valid, build_test, oracle


def _test_intent_present(task: PairedSemanticTask, test: str) -> bool:
    markers = {
        "P01": ("assert", "allowed(5, 5)"),
        "P02": ("assert", "== 10"),
        "P03": ("assert", "eligible(18, 18)"),
        "P04": ("assert", "should_retry(3, 3)"),
        "P05": ("assert", "token_length_valid(8)"),
        "P06": ("assert", "valid_batch_size(1)"),
    }
    return all(marker in test for marker in markers[task.task_id])


def _concepts_present(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    lowered = text.lower()
    return all(any(term.lower() in lowered for term in group) for group in groups)


def _hidden_c_oracle(task_id: str) -> str:
    if task_id == "P01":
        return (
            "#include <assert.h>\n#include <stdbool.h>\n"
            "bool allowed(int,int);\n"
            "int main(void){assert(allowed(5,5));"
            "assert(!allowed(4,5));return 0;}\n"
        )
    if task_id == "P05":
        return (
            "#include <assert.h>\n#include <stdbool.h>\n#include <stddef.h>\n"
            "bool token_length_valid(size_t);\n"
            "int main(void){assert(token_length_valid(8));"
            "assert(!token_length_valid(9));return 0;}\n"
        )
    raise ValueError(task_id)


def _hidden_python_oracle(task_id: str) -> str:
    values = {
        "P02": (
            "from calculator import capped_add; assert capped_add(8,7,10)==10; "
            "assert capped_add(2,3,10)==5"
        ),
        "P03": (
            "from access import eligible; assert eligible(18,18); "
            "assert not eligible(17,18)"
        ),
        "P04": (
            "from retry_policy import should_retry; assert should_retry(3,3); "
            "assert not should_retry(4,3)"
        ),
        "P06": (
            "from validation import valid_batch_size; assert valid_batch_size(1); "
            "assert valid_batch_size(100); assert not valid_batch_size(101)"
        ),
    }
    return values[task_id]


def _command_ok(command: tuple[str, ...], cwd: Path) -> bool:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        shell=False,
    )
    return completed.returncode == 0


def _generate(
    model: Model, request: ModelRequest
) -> tuple[ModelResponse | None, float]:
    started = time.perf_counter()
    try:
        return model.generate(request), time.perf_counter() - started
    except Exception:
        return None, time.perf_counter() - started


def _usage(
    request: ModelRequest | None, response: ModelResponse | None
) -> tuple[int, int]:
    if response is None:
        return 0, 0
    estimator = ConservativeTokenEstimator()
    estimated_input = (
        0
        if request is None
        else sum(estimator.estimate(message) for message in request.messages)
    )
    estimated_output = estimator.estimate(Message(MessageRole.ASSISTANT, response.text))
    return (
        response.usage.input_tokens
        if response.usage.input_tokens is not None
        else estimated_input,
        response.usage.output_tokens
        if response.usage.output_tokens is not None
        else estimated_output,
    )


def _plan_file_count(response: ModelResponse | None) -> int:
    if response is None:
        return 0
    try:
        files = json.loads(response.text).get("files", [])
    except (AttributeError, json.JSONDecodeError):
        return 0
    return len(files) if isinstance(files, list) else 0

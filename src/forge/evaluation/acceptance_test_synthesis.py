"""Evaluator-only task-derived acceptance-test synthesis and qualification."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from forge.evaluation.realistic_semantic import RealisticSemanticTask
from forge.evaluation.realworld import (
    SetupReplacement,
    apply_task_setup,
    copy_repository,
    hash_workspace,
)
from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    OutputSpecification,
    ResponseFormat,
)

ACCEPTANCE_TEST_SYNTHESIS_V1 = "acceptance-test-synthesis-v1"
ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION = 1
ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION = 1
MAX_TEST_SOURCE_CHARS = 6000
TEST_TIMEOUT_SECONDS = 10.0


class SynthesisCondition(Enum):
    C0_TASK_ONLY = "C0_TASK_ONLY"
    C1_GROUNDED = "C1_TASK_RELEVANT_SOURCE_EXISTING_TESTS"


class AcceptanceQualification(Enum):
    QUALIFIED = "QUALIFIED"
    BASELINE_NOT_DETECTED = "BASELINE_NOT_DETECTED"
    REFERENCE_REJECTED = "REFERENCE_REJECTED"
    WRONG_ACCEPTED = "WRONG_ACCEPTED"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    STRUCTURAL_INVALID = "STRUCTURAL_INVALID"
    TEST_EXECUTION_FAILED = "TEST_EXECUTION_FAILED"
    UNSAFE_TEST = "UNSAFE_TEST"
    FLAKY = "FLAKY"


@dataclass(frozen=True, slots=True)
class GeneratedAcceptanceTest:
    test_name: str
    test_source: str


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    schema_valid: bool
    structurally_valid: bool
    safe: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceTestGenerationResult:
    task_id: str
    model_profile: str
    condition: SynthesisCondition
    seed: int
    schema_valid: bool
    structurally_valid: bool
    safe: bool
    baseline_result: str
    reference_result: str
    reference_repeat_result: str
    wrong_result: str
    qualification: AcceptanceQualification
    candidate_sha256: str | None
    candidate_size: int
    input_tokens: int | None
    output_tokens: int | None
    generation_latency_seconds: float
    execution_latency_seconds: float
    alternative_correct_result: str
    combined_baseline_result: str
    combined_reference_result: str
    combined_wrong_result: str
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceTestSynthesisRun:
    suite: str
    suite_version: int
    schema_version: int
    repository_identity: str
    model_profile: str
    context_capacity: int
    seed: int
    output_budget: int
    results: tuple[AcceptanceTestGenerationResult, ...]
    canonical_unchanged: bool


@dataclass(frozen=True, slots=True)
class QualificationResults:
    baseline: str
    reference: str
    reference_repeat: str
    wrong: str
    qualification: AcceptanceQualification
    duration_seconds: float


class AcceptanceTaskDefinition(Protocol):
    metadata: RealisticSemanticTask
    wrong: tuple[SetupReplacement, ...]


_TEST_PATHS = {
    "R01": "tests/test_retry.py",
    "R02": "tests/test_config.py",
    "R03": "cengine/tests/test_parser.c",
    "R04": "cengine/tests/test_quota.c",
    "R05": "tests/test_state.py",
    "R06": "tests/test_headers.py",
    "R07": "cengine/tests/test_window.c",
    "R08": "cengine/tests/test_status.c",
}
_LEAKAGE_MARKERS = (
    "benchmarks/realistic_semantic_v1/oracle.py",
    "hidden oracle",
    "reference mutation",
    "wrong mutation",
    "expected repaired source",
    "baseline_accepting",
)
_PYTHON_FORBIDDEN_IMPORTS = {
    "http",
    "os",
    "pathlib",
    "random",
    "requests",
    "socket",
    "subprocess",
    "time",
    "urllib",
}
_PYTHON_FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "open",
    "__import__",
}
_C_FORBIDDEN = re.compile(
    r"\b(?:system|popen|fork|execl|execv|fopen|freopen|open|socket|connect)\s*\("
)


def build_acceptance_test_request(
    task: RealisticSemanticTask,
    condition: SynthesisCondition,
    repository: Path,
) -> ModelRequest:
    """Build a source-bounded prompt from production-visible information only."""
    production = task.production_task
    language = task.language
    if language == "Python":
        convention = (
            "Return one standalone Python assertion script. It runs from repository "
            "root, imports public behavior from pyservice, and uses plain assert."
        )
    else:
        source_name = Path(production.expected_files[0]).stem
        convention = (
            "Return one standalone C17 file with main and assert checks. It is "
            f'materialized at repository root, so include "cengine/{source_name}.h".'
        )
    parts = [
        "Generate one focused behavioral acceptance test for the requested task.",
        "Return the artifact directly; do not explain reasoning.",
        "Do not read implementation files as text or invoke commands/network.",
        f"Language: {language}",
        f"Task: {production.prompt}",
        f"Visible test convention: {convention}",
    ]
    if condition is SynthesisCondition.C1_GROUNDED:
        for path in (*production.expected_files, _TEST_PATHS[task.task_id]):
            source = (repository / path).read_text(encoding="utf-8")
            parts.extend((f"BEGIN VISIBLE FILE {path}", source, "END VISIBLE FILE"))
    content = "\n\n".join(parts)
    if contains_evaluator_leakage(content):
        raise ValueError("acceptance-test prompt contains evaluator-only material")
    schema = {
        "type": "object",
        "properties": {
            "test_name": {"type": "string"},
            "test_source": {"type": "string"},
        },
        "required": ["test_name", "test_source"],
        "additionalProperties": False,
    }
    return ModelRequest(
        (Message(MessageRole.USER, content),),
        GenerationConfig(max_tokens=512, temperature=0.0, seed=42),
        OutputSpecification(ResponseFormat.JSON, schema),
    )


def parse_generated_acceptance_test(text: str) -> GeneratedAcceptanceTest | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"test_name", "test_source"}:
        return None
    name = value["test_name"]
    source = value["test_source"]
    if not isinstance(name, str) or not name.strip() or not isinstance(source, str):
        return None
    if not source.strip():
        return None
    return GeneratedAcceptanceTest(name.strip(), source)


def validate_generated_acceptance_test(
    candidate: GeneratedAcceptanceTest,
    language: str,
) -> CandidateValidation:
    source = candidate.test_source
    if len(source) > MAX_TEST_SOURCE_CHARS:
        return CandidateValidation(True, False, False, "candidate exceeds size limit")
    if contains_evaluator_leakage(source):
        return CandidateValidation(True, True, False, "evaluator reference leakage")
    lowered = source.casefold()
    if any(marker in lowered for marker in ("../", "/users/", "eval-results")):
        return CandidateValidation(
            True, True, False, "path escape or local data access"
        )
    if language == "Python":
        return _validate_python(source)
    if language == "C17":
        return _validate_c(source)
    return CandidateValidation(True, False, False, "unsupported language")


def classify_qualification(
    baseline: str,
    reference: str,
    wrong: str,
    *,
    reference_repeat: str | None = None,
) -> AcceptanceQualification:
    if "unavailable" in {baseline, reference, wrong, reference_repeat}:
        return AcceptanceQualification.TEST_EXECUTION_FAILED
    if baseline == "pass":
        return AcceptanceQualification.BASELINE_NOT_DETECTED
    if reference != "pass":
        return AcceptanceQualification.REFERENCE_REJECTED
    if wrong == "pass":
        return AcceptanceQualification.WRONG_ACCEPTED
    if reference_repeat is not None and reference_repeat != reference:
        return AcceptanceQualification.FLAKY
    return AcceptanceQualification.QUALIFIED


def combine_verification_signals(
    generated: tuple[str, str, str],
    full: tuple[str, str, str],
) -> tuple[str, str, str]:
    """Compose evaluation signals with AND semantics; never replace the full plan."""
    combined = []
    for generated_result, full_result in zip(generated, full, strict=True):
        if "unavailable" in {generated_result, full_result}:
            combined.append("unavailable")
        elif generated_result == full_result == "pass":
            combined.append("pass")
        else:
            combined.append("fail")
    return tuple(combined)  # type: ignore[return-value]


def qualify_acceptance_test(
    root: Path,
    canonical_repository: Path,
    task: RealisticSemanticTask,
    wrong_mutation: tuple[SetupReplacement, ...],
    candidate: GeneratedAcceptanceTest,
) -> QualificationResults:
    """Qualify one safe candidate on independent B/R/W states and repeat R once."""
    started = time.perf_counter()
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="forge-a49-qualify-", dir=root) as name:
        temporary = Path(name)
        workspaces: dict[str, Path] = {}
        for state in ("baseline", "reference", "wrong"):
            workspace = copy_repository(canonical_repository, temporary / state)
            if state != "reference":
                apply_task_setup(workspace, task.production_task.setup)
            if state == "wrong":
                if wrong_mutation:
                    apply_task_setup(workspace, wrong_mutation)
                else:
                    test_path = task.production_task.expected_changed_paths[1]
                    (workspace / test_path).write_bytes(
                        (canonical_repository / test_path).read_bytes()
                    )
            workspaces[state] = workspace
            results[state] = execute_candidate(candidate, task.language, workspace)
        results["reference_repeat"] = execute_candidate(
            candidate, task.language, workspaces["reference"]
        )
    qualification = classify_qualification(
        results["baseline"],
        results["reference"],
        results["wrong"],
        reference_repeat=results["reference_repeat"],
    )
    return QualificationResults(
        results["baseline"],
        results["reference"],
        results["reference_repeat"],
        results["wrong"],
        qualification,
        time.perf_counter() - started,
    )


def execute_candidate(
    candidate: GeneratedAcceptanceTest,
    language: str,
    workspace: Path,
) -> str:
    """Materialize and execute with evaluator-owned argument arrays and timeout."""
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(workspace),
        "HOME": str(workspace / ".a49-home"),
        "TMPDIR": str(workspace / ".a49-tmp"),
        "LANG": "C",
    }
    (workspace / ".a49-home").mkdir(exist_ok=True)
    (workspace / ".a49-tmp").mkdir(exist_ok=True)
    try:
        if language == "Python":
            test = workspace / "a49_generated_acceptance.py"
            test.write_text(candidate.test_source, encoding="utf-8")
            completed = subprocess.run(
                (sys.executable, str(test)),
                cwd=workspace,
                env=environment,
                capture_output=True,
                check=False,
                shell=False,
                timeout=TEST_TIMEOUT_SECONDS,
            )
            return "pass" if completed.returncode == 0 else "fail"
        test = workspace / "a49_generated_acceptance.c"
        binary = workspace / "a49_generated_acceptance"
        test.write_text(candidate.test_source, encoding="utf-8")
        implementation = _c_implementation(candidate.test_source)
        if implementation is None:
            return "structural_invalid"
        compiled = subprocess.run(
            (
                "/usr/bin/cc",
                "-std=c17",
                "-Wall",
                "-Werror",
                implementation,
                str(test),
                "-o",
                str(binary),
            ),
            cwd=workspace,
            env=environment,
            capture_output=True,
            check=False,
            shell=False,
            timeout=TEST_TIMEOUT_SECONDS,
        )
        if compiled.returncode != 0:
            return "structural_invalid"
        completed = subprocess.run(
            (str(binary),),
            cwd=workspace,
            env=environment,
            capture_output=True,
            check=False,
            shell=False,
            timeout=TEST_TIMEOUT_SECONDS,
        )
        return "pass" if completed.returncode == 0 else "fail"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def run_acceptance_test_synthesis_v1(
    root: Path,
    canonical_repository: Path,
    definitions: tuple[AcceptanceTaskDefinition, ...],
    model_profile: str,
    model: Model,
    *,
    full_verification_signals: dict[str, tuple[str, str, str]] | None = None,
    conditions: tuple[SynthesisCondition, ...] = tuple(SynthesisCondition),
) -> AcceptanceTestSynthesisRun:
    """Run one generation per task/condition with no success-seeking retries."""
    root.mkdir(parents=True, exist_ok=True)
    before = hash_workspace(canonical_repository)
    results: list[AcceptanceTestGenerationResult] = []
    for definition in definitions:
        metadata = definition.metadata
        wrong = definition.wrong
        for condition in conditions:
            results.append(
                _run_generation_cell(
                    root,
                    canonical_repository,
                    metadata,
                    wrong,
                    model_profile,
                    model,
                    condition,
                    (full_verification_signals or {}).get(
                        metadata.task_id, ("unavailable", "unavailable", "unavailable")
                    ),
                )
            )
    unchanged = before == hash_workspace(canonical_repository)
    if not unchanged:
        raise RuntimeError("canonical realistic semantic repository changed")
    return AcceptanceTestSynthesisRun(
        ACCEPTANCE_TEST_SYNTHESIS_V1,
        ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION,
        ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION,
        hashlib.sha256(repr(before).encode()).hexdigest(),
        model_profile,
        model.context_capacity or 0,
        42,
        512,
        tuple(results),
        unchanged,
    )


def acceptance_test_synthesis_to_dict(
    run: AcceptanceTestSynthesisRun,
) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    payload = convert(asdict(run))
    assert isinstance(payload, dict)
    return payload


def write_acceptance_test_synthesis_json(
    run: AcceptanceTestSynthesisRun, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(acceptance_test_synthesis_to_dict(run), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_acceptance_test_synthesis_json(path: Path) -> AcceptanceTestSynthesisRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = tuple(
        AcceptanceTestGenerationResult(
            **{
                **item,
                "condition": SynthesisCondition(item["condition"]),
                "qualification": AcceptanceQualification(item["qualification"]),
            }
        )
        for item in payload["results"]
    )
    return AcceptanceTestSynthesisRun(
        payload["suite"],
        payload["suite_version"],
        payload["schema_version"],
        payload["repository_identity"],
        payload["model_profile"],
        payload["context_capacity"],
        payload["seed"],
        payload["output_budget"],
        results,
        payload["canonical_unchanged"],
    )


def contains_evaluator_leakage(value: str) -> bool:
    lowered = value.casefold()
    return any(marker in lowered for marker in _LEAKAGE_MARKERS)


def _run_generation_cell(
    root: Path,
    canonical: Path,
    task: RealisticSemanticTask,
    wrong: tuple[SetupReplacement, ...],
    profile: str,
    model: Model,
    condition: SynthesisCondition,
    full_verification: tuple[str, str, str],
) -> AcceptanceTestGenerationResult:
    with tempfile.TemporaryDirectory(prefix="forge-a49-context-", dir=root) as name:
        visible_repository = copy_repository(canonical, Path(name) / "baseline")
        apply_task_setup(visible_repository, task.production_task.setup)
        request = build_acceptance_test_request(task, condition, visible_repository)
    started = time.perf_counter()
    response = model.generate(request)
    generation_latency = time.perf_counter() - started
    candidate = parse_generated_acceptance_test(response.text)
    if candidate is None:
        return _failed_result(
            task.task_id,
            profile,
            condition,
            AcceptanceQualification.SCHEMA_INVALID,
            response.usage.input_tokens,
            response.usage.output_tokens,
            generation_latency,
            "response did not match the constrained schema",
        )
    validation = validate_generated_acceptance_test(candidate, task.language)
    if not validation.safe:
        return _candidate_result(
            task.task_id,
            profile,
            condition,
            candidate,
            validation,
            AcceptanceQualification.UNSAFE_TEST,
            response.usage.input_tokens,
            response.usage.output_tokens,
            generation_latency,
            validation.reason,
        )
    if not validation.structurally_valid:
        return _candidate_result(
            task.task_id,
            profile,
            condition,
            candidate,
            validation,
            AcceptanceQualification.STRUCTURAL_INVALID,
            response.usage.input_tokens,
            response.usage.output_tokens,
            generation_latency,
            validation.reason,
        )
    qualified = qualify_acceptance_test(root, canonical, task, wrong, candidate)
    qualification = qualified.qualification
    if "structural_invalid" in {
        qualified.baseline,
        qualified.reference,
        qualified.wrong,
    }:
        qualification = AcceptanceQualification.STRUCTURAL_INVALID
    alternative = (
        execute_alternative_correct(root, canonical, task, candidate)
        if qualification is AcceptanceQualification.QUALIFIED
        else "not_run"
    )
    combined = (
        combine_verification_signals(
            (qualified.baseline, qualified.reference, qualified.wrong),
            full_verification,
        )
        if qualification is AcceptanceQualification.QUALIFIED
        else ("not_run", "not_run", "not_run")
    )
    return AcceptanceTestGenerationResult(
        task.task_id,
        profile,
        condition,
        42,
        True,
        qualification is not AcceptanceQualification.STRUCTURAL_INVALID,
        True,
        qualified.baseline,
        qualified.reference,
        qualified.reference_repeat,
        qualified.wrong,
        qualification,
        hashlib.sha256(candidate.test_source.encode()).hexdigest(),
        len(candidate.test_source),
        response.usage.input_tokens,
        response.usage.output_tokens,
        generation_latency,
        qualified.duration_seconds,
        alternative,
        *combined,
        None,
    )


def execute_alternative_correct(
    root: Path,
    canonical: Path,
    task: RealisticSemanticTask,
    candidate: GeneratedAcceptanceTest,
) -> str:
    """Exercise two evaluator-owned behaviorally equivalent implementations."""
    replacements: dict[str, SetupReplacement] = {
        "R05": SetupReplacement(
            "pyservice/state.py",
            "def can_transition(current: JobState, target: JobState) -> bool:\n"
            "    return target in _TRANSITIONS[current]",
            "def can_transition(current: JobState, target: JobState) -> bool:\n"
            "    if current in {JobState.SUCCEEDED, JobState.FAILED, "
            "JobState.CANCELLED}:\n"
            "        return False\n"
            "    return target in _TRANSITIONS[current]",
        ),
        "R06": SetupReplacement(
            "pyservice/headers.py",
            "if not candidate or _HEADER_NAME.fullmatch(candidate) is None:\n"
            '        raise ValueError("invalid header name")',
            "allowed = candidate.isascii() and all(\n"
            '        character.isalnum() or character == "-"\n'
            "        for character in candidate\n"
            "    )\n"
            "    if not candidate or not allowed:\n"
            '        raise ValueError("invalid header name")',
        ),
    }
    replacement = replacements.get(task.task_id)
    if replacement is None:
        return "not_run"
    with tempfile.TemporaryDirectory(prefix="forge-a49-alternative-", dir=root) as name:
        workspace = copy_repository(canonical, Path(name) / "workspace")
        apply_task_setup(workspace, (replacement,))
        return execute_candidate(candidate, task.language, workspace)


def _validate_python(source: str) -> CandidateValidation:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return CandidateValidation(True, False, True, f"Python syntax: {error.msg}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [item.name.split(".")[0] for item in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            if any(name in _PYTHON_FORBIDDEN_IMPORTS for name in names):
                return CandidateValidation(True, True, False, "forbidden import")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in _PYTHON_FORBIDDEN_CALLS:
                return CandidateValidation(True, True, False, "forbidden call")
    return CandidateValidation(True, True, True)


def _validate_c(source: str) -> CandidateValidation:
    if _C_FORBIDDEN.search(source):
        return CandidateValidation(True, True, False, "forbidden C operation")
    includes = re.findall(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", source, re.M)
    allowed = {"assert.h", "stdbool.h", "stddef.h"}
    if any(
        item not in allowed and not item.startswith("cengine/") for item in includes
    ):
        return CandidateValidation(True, True, False, "forbidden C include")
    if "main(" not in source.replace(" ", ""):
        return CandidateValidation(True, False, True, "C candidate has no main")
    return CandidateValidation(True, True, True)


def _c_implementation(source: str) -> str | None:
    match = re.search(r"cengine/(parser|quota|window|status)\.h", source)
    return f"cengine/{match.group(1)}.c" if match else None


def _failed_result(
    task_id: str,
    profile: str,
    condition: SynthesisCondition,
    qualification: AcceptanceQualification,
    input_tokens: int | None,
    output_tokens: int | None,
    latency: float,
    reason: str,
) -> AcceptanceTestGenerationResult:
    return AcceptanceTestGenerationResult(
        task_id,
        profile,
        condition,
        42,
        False,
        False,
        False,
        "not_run",
        "not_run",
        "not_run",
        "not_run",
        qualification,
        None,
        0,
        input_tokens,
        output_tokens,
        latency,
        0.0,
        "not_run",
        "not_run",
        "not_run",
        "not_run",
        reason,
    )


def _candidate_result(
    task_id: str,
    profile: str,
    condition: SynthesisCondition,
    candidate: GeneratedAcceptanceTest,
    validation: CandidateValidation,
    qualification: AcceptanceQualification,
    input_tokens: int | None,
    output_tokens: int | None,
    latency: float,
    reason: str | None,
) -> AcceptanceTestGenerationResult:
    return AcceptanceTestGenerationResult(
        task_id,
        profile,
        condition,
        42,
        validation.schema_valid,
        validation.structurally_valid,
        validation.safe,
        "not_run",
        "not_run",
        "not_run",
        "not_run",
        qualification,
        hashlib.sha256(candidate.test_source.encode()).hexdigest(),
        len(candidate.test_source),
        input_tokens,
        output_tokens,
        latency,
        0.0,
        "not_run",
        "not_run",
        "not_run",
        "not_run",
        reason,
    )

"""Frozen R01-v1 through R08-v1 definitions and integrity validation."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forge.evaluation import (
    EvaluationOutcome,
    RealWorldLevel,
    RealWorldTask,
    SemanticIntegrityResult,
    SetupReplacement,
    apply_task_setup,
    copy_repository,
    hash_workspace,
    run_oracle,
)
from forge.evaluation.realistic_semantic import RealisticSemanticTask
from forge.interaction import AutonomyMode
from forge.project_config import VerificationPlan

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT / "repository"
ORACLE = ROOT / "oracle.py"

CONFIGURE = ("python3", "tools/configure.py")
BUILD = ("python3", "tools/build.py")
TEST = ("python3", "tools/test.py")
VERIFICATION = VerificationPlan(
    "realistic-semantic-configure-build-test",
    ("project.configure", "project.build", "project.test"),
)


@dataclass(frozen=True, slots=True)
class FrozenTaskDefinition:
    metadata: RealisticSemanticTask
    wrong: tuple[SetupReplacement, ...]


def _production_task(
    task_id: str,
    prompt: str,
    expected_files: tuple[str, ...],
    changed_paths: tuple[str, ...],
    setup: tuple[SetupReplacement, ...],
    *,
    seeds: tuple[int, ...],
    required: tuple[str, ...] = (),
) -> RealWorldTask:
    return RealWorldTask(
        task_id,
        RealWorldLevel.BOUNDED_REPAIR,
        AutonomyMode.REPAIR,
        prompt,
        expected_files,
        allowed_paths=changed_paths,
        expected_changed_paths=changed_paths,
        required_candidate_paths=required,
        setup=setup,
        setup_commands=(CONFIGURE,),
        configure_command=CONFIGURE,
        build_command=BUILD,
        test_command=TEST,
        verification_plan=VERIFICATION,
        oracle_commands=(("python3", str(ORACLE), task_id),),
        seeds=seeds,
        max_mutations=2,
    )


def realistic_semantic_tasks(
    seeds: tuple[int, ...],
) -> tuple[FrozenTaskDefinition, ...]:
    """Return the immutable A45 task set for the requested model seeds."""
    definitions = (
        (
            "R01",
            "Python",
            "L2",
            "single_file",
            "discovery_required",
            "The retry policy incorrectly permits a retry after the final allowed "
            "zero-based attempt. Fix should_retry so a three-attempt policy retries "
            "attempts 0 and 1, never attempt 2, while preserving retryable-status "
            "handling. Verify the repository.",
            ("pyservice/retry.py",),
            (
                SetupReplacement(
                    "pyservice/retry.py",
                    "return retryable and attempt + 1 < policy.max_attempts",
                    "return retryable and attempt < policy.max_attempts",
                ),
            ),
            (
                SetupReplacement(
                    "pyservice/retry.py",
                    "return retryable and attempt < policy.max_attempts",
                    "return retryable and attempt <= policy.max_attempts",
                ),
            ),
            (),
        ),
        (
            "R02",
            "Python",
            "L2",
            "single_file",
            "discovery_required",
            "Boolean configuration parsing now treats every non-empty string as "
            "enabled. Restore documented case-insensitive true/false values, retain "
            "the default for a missing key, reject unknown values, and verify.",
            ("pyservice/config.py",),
            (
                SetupReplacement(
                    "pyservice/config.py",
                    "normalized = value.strip().casefold()\n"
                    "    if normalized in _TRUE_VALUES:\n"
                    "        return True\n"
                    "    if normalized in _FALSE_VALUES:\n"
                    "        return False\n"
                    '    raise ValueError(f"invalid boolean setting: {key}")',
                    "return bool(value.strip())",
                ),
            ),
            (
                SetupReplacement(
                    "pyservice/config.py",
                    "return bool(value.strip())",
                    'return value.strip().casefold() == "true"',
                ),
            ),
            (),
        ),
        (
            "R03",
            "C17",
            "L2",
            "single_file",
            "discovery_required",
            "The port parser accepts a numeric prefix followed by trailing junk. "
            "Make parse_port require the entire input to be a valid decimal port, "
            "while retaining null, empty, range, and errno checks. Verify it.",
            ("cengine/parser.c",),
            (
                SetupReplacement(
                    "cengine/parser.c",
                    "errno != 0 || *end != '\\0' || value < 1 || value > 65535",
                    "errno != 0 || value < 1 || value > 65535",
                ),
            ),
            (
                SetupReplacement(
                    "cengine/parser.c",
                    "errno != 0 || value < 1 || value > 65535",
                    "errno != 0 || *end == ' ' || value < 1 || value > 65535",
                ),
            ),
            (),
        ),
        (
            "R04",
            "C17",
            "L2",
            "single_file",
            "discovery_required",
            "Releasing more quota than is currently used underflows the accounting "
            "counter. Make quota_release saturate at zero without changing reserve "
            "behavior or null handling, then verify.",
            ("cengine/quota.c",),
            (
                SetupReplacement(
                    "cengine/quota.c",
                    "value->used = amount > value->used ? 0 : value->used - amount;",
                    "value->used -= amount;",
                ),
            ),
            (
                SetupReplacement(
                    "cengine/quota.c",
                    "value->used -= amount;",
                    "value->used = amount >= value->used ? 1 : value->used - amount;",
                ),
            ),
            (),
        ),
        (
            "R05",
            "Python",
            "L3",
            "multi_file",
            "path_known",
            "In pyservice/state.py, terminal FAILED jobs can incorrectly transition "
            "back to RUNNING. Restore terminal-state behavior and add the coordinated "
            "regression in tests/test_state.py, then verify both files.",
            ("pyservice/state.py", "tests/test_state.py"),
            (
                SetupReplacement(
                    "pyservice/state.py",
                    "JobState.FAILED: frozenset(),",
                    "JobState.FAILED: frozenset({JobState.RUNNING}),",
                ),
                SetupReplacement(
                    "tests/test_state.py",
                    "assert not can_transition(JobState.FAILED, JobState.RUNNING)",
                    "assert not can_transition(JobState.FAILED, JobState.SUCCEEDED)",
                ),
            ),
            (),
            ("pyservice/state.py", "tests/test_state.py"),
        ),
        (
            "R06",
            "Python",
            "L3",
            "multi_file",
            "path_known",
            "In pyservice/headers.py, header names with underscores are being "
            "accepted even though only letters, digits, and hyphens are valid. Fix "
            "the validator and add the coordinated regression in "
            "tests/test_headers.py, then verify both files.",
            ("pyservice/headers.py", "tests/test_headers.py"),
            (
                SetupReplacement(
                    "pyservice/headers.py",
                    're.compile(r"^[A-Za-z0-9-]+$")',
                    're.compile(r"^[A-Za-z0-9_-]+$")',
                ),
                SetupReplacement(
                    "tests/test_headers.py",
                    'assert rejects("x_header")',
                    'assert normalize_header_name("X-Header") == "x-header"',
                ),
            ),
            (),
            ("pyservice/headers.py", "tests/test_headers.py"),
        ),
        (
            "R07",
            "C17",
            "L3",
            "multi_file",
            "path_known",
            "In cengine/window.c, a window at its event limit is incorrectly "
            "accepted. Restore the exclusive limit and add the boundary regression "
            "to cengine/tests/test_window.c, then verify both files.",
            ("cengine/window.c", "cengine/tests/test_window.c"),
            (
                SetupReplacement(
                    "cengine/window.c",
                    "return limit > 0 && events < limit;",
                    "return limit > 0 && events <= limit;",
                ),
                SetupReplacement(
                    "cengine/tests/test_window.c",
                    "assert(!window_accepts(3, 3));",
                    "assert(window_accepts(1, 3));",
                ),
            ),
            (),
            ("cengine/window.c", "cengine/tests/test_window.c"),
        ),
        (
            "R08",
            "C17",
            "L3",
            "multi_file",
            "path_known",
            "In cengine/status.c, dependency I/O errors are incorrectly normalized "
            "to retry. Preserve distinct I/O error propagation and add the "
            "coordinated regression in cengine/tests/test_status.c, then verify.",
            ("cengine/status.c", "cengine/tests/test_status.c"),
            (
                SetupReplacement(
                    "cengine/status.c",
                    "return status;\n    }\n    return ENGINE_INVALID;",
                    "return status == ENGINE_IO_ERROR ? ENGINE_RETRY : status;\n"
                    "    }\n    return ENGINE_INVALID;",
                ),
                SetupReplacement(
                    "cengine/tests/test_status.c",
                    "assert(normalize_dependency_status(ENGINE_IO_ERROR) == "
                    "ENGINE_IO_ERROR);",
                    "assert(normalize_dependency_status(ENGINE_INVALID) == "
                    "ENGINE_INVALID);",
                ),
            ),
            (),
            ("cengine/status.c", "cengine/tests/test_status.c"),
        ),
    )
    values = []
    for (
        task_id,
        language,
        difficulty,
        mutation_kind,
        path_mode,
        prompt,
        paths,
        setup,
        wrong,
        required,
    ) in definitions:
        production = _production_task(
            task_id,
            prompt,
            paths,
            paths,
            setup,
            seeds=seeds,
            required=required,
        )
        values.append(
            FrozenTaskDefinition(
                RealisticSemanticTask(
                    task_id,
                    1,
                    language,
                    difficulty,
                    mutation_kind,
                    path_mode,
                    production,
                ),
                wrong,
            )
        )
    return tuple(values)


def validate_integrity(
    definitions: tuple[FrozenTaskDefinition, ...],
) -> tuple[SemanticIntegrityResult, ...]:
    """Prove baseline FAIL, canonical reference PASS, and wrong mutation FAIL."""
    before = hash_workspace(REPOSITORY)
    results = []
    for definition in definitions:
        task = definition.metadata.production_task
        with tempfile.TemporaryDirectory(prefix=f"forge-a45-{task.task_id}-") as name:
            workspace = copy_repository(REPOSITORY, Path(name).resolve() / "baseline")
            apply_task_setup(workspace, task.setup)
            baseline = run_oracle(workspace, task.oracle_commands)

            reference = Path(name).resolve() / "reference"
            shutil.copytree(workspace, reference)
            for path in task.expected_changed_paths:
                destination = reference / path
                destination.write_bytes((REPOSITORY / path).read_bytes())
            reference_result = run_oracle(reference, task.oracle_commands)

            wrong = Path(name).resolve() / "wrong"
            shutil.copytree(workspace, wrong)
            if definition.wrong:
                apply_task_setup(wrong, definition.wrong)
            else:
                # A paired plausible-but-wrong change restores only the test.
                test_path = task.expected_changed_paths[1]
                (wrong / test_path).write_bytes((REPOSITORY / test_path).read_bytes())
            wrong_result = run_oracle(wrong, task.oracle_commands)
        eligible = (
            baseline is EvaluationOutcome.FAIL
            and reference_result is EvaluationOutcome.PASS
        )
        results.append(
            SemanticIntegrityResult(
                task.task_id,
                definition.metadata.task_version,
                baseline is EvaluationOutcome.PASS,
                reference_result is EvaluationOutcome.PASS,
                wrong_result is EvaluationOutcome.PASS,
                eligible,
            )
        )
    if before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical realistic semantic repository changed")
    return tuple(results)

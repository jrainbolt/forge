"""Frozen predeclared task and primary-cell corpus for A59."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import (
    BUILD,
    CONFIGURE,
    REPOSITORY,
    TEST,
    AuthorityMode,
    FrozenTask,
    OperationClass,
    _task,
    manifest_identity,
)
from benchmarks.realistic_coding_v2.suite import (
    tasks as a56_tasks,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    SetupReplacement,
    apply_task_absences,
    apply_task_setup,
    copy_repository,
    run_oracle,
)

SUITE = "cross-model-repair-confirmation-v1"
VERSION = 1
ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / "oracle.py"
PRIMARY_PROFILES = ("qwen-small", "codestral-22b")
HISTORICAL_EXCLUSION = "K01"
HYPOTHESIS = "qwen-large repair produces more semantic recoveries than same-model"
THRESHOLD = {
    "qwen_large_recoveries": 3,
    "minimum_advantage": 2,
    "minimum_source_models_or_operation_classes": 2,
}


def _base(
    task_id: str,
    prompt: str,
    expected: tuple[str, ...],
    edits: tuple[str, ...],
    creates: tuple[str, ...],
    *,
    setup: tuple[SetupReplacement, ...] = (),
):
    return replace(
        _task(task_id, prompt, expected, edits, creates, setup=setup),
        oracle_commands=(("python3", str(ORACLE), task_id),),
    )


def tasks() -> tuple[FrozenTask, ...]:
    return (
        FrozenTask(
            "Q01",
            1,
            "Python",
            OperationClass.EDIT_SINGLE,
            AuthorityMode.DISCOVERY_REQUIRED,
            _base(
                "Q01",
                "Fix pyservice/retry.py so HTTP 500 is retryable, attempts remain "
                "zero-based, the last allowed attempt is not retried, and "
                "exponential delay behavior remains correct. Modify only that "
                "file and verify.",
                ("pyservice/retry.py",),
                ("pyservice/retry.py",),
                (),
                setup=(
                    SetupReplacement(
                        "pyservice/retry.py",
                        "500 <= status < 600",
                        "500 < status < 600",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "Q02",
            1,
            "C17",
            OperationClass.EDIT_SINGLE,
            AuthorityMode.DISCOVERY_REQUIRED,
            _base(
                "Q02",
                "Fix cengine/quota.c so a reservation may exactly fill remaining "
                "capacity while overflow remains rejected and release still "
                "saturates at zero. Modify only that file. Configure, build, and test.",
                ("cengine/quota.c", "cengine/quota.h"),
                ("cengine/quota.c",),
                (),
                setup=(
                    SetupReplacement(
                        "cengine/quota.c",
                        "amount > value->limit - value->used",
                        "amount >= value->limit - value->used",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "Q03",
            1,
            "Python",
            OperationClass.EDIT_MULTI,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "Q03",
                "Repair both pyservice/headers.py and pyservice/service.py: header "
                "names must be trimmed, validated, and case-folded, while requests "
                "with missing IDs return 400 before empty payloads return 204. Both "
                "files must change. Verify.",
                ("pyservice/headers.py", "pyservice/service.py"),
                ("pyservice/headers.py", "pyservice/service.py"),
                (),
                setup=(
                    SetupReplacement(
                        "pyservice/headers.py",
                        "return candidate.casefold()",
                        "return candidate",
                    ),
                    SetupReplacement(
                        "pyservice/service.py",
                        "if not request.request_id:\n        return Response(400, "
                        'b"missing request id")\n    if not request.payload:',
                        'if not request.payload:\n        return Response(204, b"")\n'
                        "    if not request.request_id:",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "Q04",
            1,
            "C17",
            OperationClass.EDIT_MULTI,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "Q04",
                "Repair cengine/window.c and cengine/status.c: windows accept "
                "strictly fewer events than a positive limit, and dependency retry "
                "status must remain retry rather than becoming invalid. Both files "
                "must change. Configure, build, and test.",
                ("cengine/window.c", "cengine/status.c"),
                ("cengine/window.c", "cengine/status.c"),
                (),
                setup=(
                    SetupReplacement(
                        "cengine/window.c", "events < limit", "events <= limit"
                    ),
                    SetupReplacement(
                        "cengine/status.c",
                        "return ENGINE_RETRY;",
                        "return ENGINE_INVALID;",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "Q05",
            1,
            "Python",
            OperationClass.CREATE,
            AuthorityMode.PATH_KNOWN,
            _base(
                "Q05",
                "pyservice/billing.py imports missing pyservice/currency.py. Create "
                "that module with format_cents producing exact two-decimal dollar "
                "strings using integer arithmetic. Do not modify existing files. "
                "Verify.",
                ("pyservice/billing.py",),
                (),
                ("pyservice/currency.py",),
            ),
        ),
        FrozenTask(
            "Q06",
            1,
            "C17",
            OperationClass.CREATE,
            AuthorityMode.PATH_KNOWN,
            _base(
                "Q06",
                "cengine/health.c requires missing cengine/health_policy.c. Create "
                "its implementation of health_is_healthy: failures strictly below "
                "limit are healthy and failures at the limit are not. Do not "
                "modify existing files. Configure, build, and test.",
                ("cengine/health.c", "cengine/health_policy.h"),
                (),
                ("cengine/health_policy.c",),
            ),
        ),
        FrozenTask(
            "Q07",
            1,
            "Python",
            OperationClass.MIXED_EDIT_CREATE,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "Q07",
                "Update pyservice/dispatch.py so dispatch is trimmed and "
                "case-insensitive, and create missing pyservice/dispatch_rules.py "
                "mapping read, write, delete to reader, writer, remover with "
                "ValueError for unknown actions. Both paths are required. Verify.",
                ("pyservice/dispatch.py",),
                ("pyservice/dispatch.py",),
                ("pyservice/dispatch_rules.py",),
                setup=(
                    SetupReplacement(
                        "pyservice/dispatch.py",
                        "action.strip().casefold()",
                        "action.strip()",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "Q08",
            1,
            "C17",
            OperationClass.MIXED_EDIT_CREATE,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "Q08",
                "Fix cengine/retry_policy.c so retry_wait starts at two and uses "
                "the intended capped exponential schedule, and create missing "
                "cengine/backoff.c with overflow-safe saturation. Both paths are "
                "required. Configure, build, and test.",
                ("cengine/retry_policy.c", "cengine/backoff.h"),
                ("cengine/retry_policy.c",),
                ("cengine/backoff.c",),
                setup=(
                    SetupReplacement(
                        "cengine/retry_policy.c", "attempt, 2, 30", "attempt, 3, 30"
                    ),
                ),
            ),
        ),
    )


def definition_identity() -> str:
    values = []
    for definition in tasks():
        task = definition.production_task
        values.append(
            {
                "id": definition.versioned_id,
                "language": definition.language,
                "operation": definition.operation_class.value,
                "prompt": task.prompt,
                "expected": task.expected_changed_paths,
                "creates": task.create_candidate_paths,
                "setup": [(x.path, x.expected, x.replacement) for x in task.setup],
            }
        )
    payload = {
        "tasks": values,
        "oracle": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "repository": manifest_identity(a56_tasks()),
        "hypothesis": HYPOTHESIS,
        "threshold": THRESHOLD,
        "primary_profiles": PRIMARY_PROFILES,
        "historical_exclusion": HISTORICAL_EXCLUSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def integrity() -> tuple[dict[str, object], ...]:
    results = []
    for definition in tasks():
        task = definition.production_task
        with tempfile.TemporaryDirectory(
            prefix=f"forge-a59-{definition.task_id}-"
        ) as name:
            baseline = copy_repository(REPOSITORY, Path(name) / "baseline")
            apply_task_setup(baseline, task.setup)
            apply_task_absences(baseline, task.setup_absent_paths)
            baseline_pass = (
                run_oracle(baseline, task.oracle_commands) is EvaluationOutcome.PASS
            )
            baseline_verification = (
                run_oracle(baseline, (CONFIGURE, BUILD, TEST)) is EvaluationOutcome.PASS
            )
            reference = Path(name) / "reference"
            shutil.copytree(
                baseline,
                reference,
                ignore=shutil.ignore_patterns("build", "__pycache__"),
            )
            for relative in task.expected_changed_paths:
                target = reference / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPOSITORY / relative).read_bytes())
            reference_pass = (
                run_oracle(reference, task.oracle_commands) is EvaluationOutcome.PASS
            )
            reference_verification = (
                run_oracle(reference, (CONFIGURE, BUILD, TEST))
                is EvaluationOutcome.PASS
            )
            partial_pass = False
            if len(task.expected_changed_paths) > 1:
                for relative in task.expected_changed_paths:
                    partial = Path(name) / ("partial-" + Path(relative).name)
                    shutil.copytree(
                        baseline,
                        partial,
                        ignore=shutil.ignore_patterns("build", "__pycache__"),
                    )
                    target = partial / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((REPOSITORY / relative).read_bytes())
                    partial_pass |= (
                        run_oracle(partial, task.oracle_commands)
                        is EvaluationOutcome.PASS
                    )
            results.append(
                {
                    "task_id": definition.task_id,
                    "baseline_pass": baseline_pass,
                    "baseline_verification_pass": baseline_verification,
                    "reference_pass": reference_pass,
                    "reference_verification_pass": reference_verification,
                    "partial_pass": partial_pass,
                    "eligible": not baseline_pass
                    and reference_pass
                    and reference_verification
                    and not partial_pass,
                }
            )
    return tuple(results)

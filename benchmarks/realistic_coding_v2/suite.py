"""Frozen C01-v1 through C12-v1 integrated coding benchmark definitions."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import realistic_semantic_tasks
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldLevel,
    RealWorldTask,
    SetupReplacement,
    apply_task_absences,
    apply_task_setup,
    copy_repository,
    hash_workspace,
    run_oracle,
)
from forge.interaction import AutonomyMode
from forge.project_config import VerificationPlan

SUITE = "realistic-coding-v2"
VERSION = 1
ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT / "repository"
ORACLE = ROOT / "oracle.py"
CONFIGURE = ("python3", "tools/configure.py")
BUILD = ("python3", "tools/build.py")
TEST = ("python3", "tools/test.py")
VERIFICATION = VerificationPlan(
    "realistic-coding-v2-configure-build-test",
    ("project.configure", "project.build", "project.test"),
)


class OperationClass(StrEnum):
    EDIT_SINGLE = "EDIT_SINGLE"
    EDIT_MULTI = "EDIT_MULTI"
    CREATE = "CREATE"
    MIXED_EDIT_CREATE = "MIXED_EDIT_CREATE"


class AuthorityMode(StrEnum):
    DISCOVERY_REQUIRED = "DISCOVERY_REQUIRED"
    PATH_KNOWN = "PATH_KNOWN"
    TRUSTED_REQUIRED_CANDIDATES = "TRUSTED_REQUIRED_CANDIDATES"


@dataclass(frozen=True, slots=True)
class FrozenTask:
    task_id: str
    version: int
    language: str
    operation_class: OperationClass
    authority_mode: AuthorityMode
    production_task: RealWorldTask
    wrong_replacements: tuple[SetupReplacement, ...] = ()
    wrong_create_source: str | None = None

    @property
    def versioned_id(self) -> str:
        return f"{self.task_id}-v{self.version}"

    @property
    def edit_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in self.production_task.expected_changed_paths
            if path not in self.production_task.create_candidate_paths
        )

    @property
    def create_paths(self) -> tuple[str, ...]:
        return self.production_task.create_candidate_paths


@dataclass(frozen=True, slots=True)
class Integrity:
    task_id: str
    baseline_pass: bool
    reference_pass: bool
    wrong_pass: bool
    partial_pass: bool
    verification_baseline_pass: bool
    verification_reference_pass: bool
    verification_wrong_pass: bool
    eligible: bool


def _task(
    task_id: str,
    prompt: str,
    expected_files: tuple[str, ...],
    edit_paths: tuple[str, ...],
    create_paths: tuple[str, ...],
    *,
    setup: tuple[SetupReplacement, ...] = (),
) -> RealWorldTask:
    changed = (*edit_paths, *create_paths)
    return RealWorldTask(
        task_id,
        RealWorldLevel.BOUNDED_REPAIR,
        AutonomyMode.REPAIR,
        prompt,
        expected_files,
        allowed_paths=changed,
        expected_changed_paths=changed,
        required_candidate_paths=edit_paths or expected_files,
        setup=setup,
        setup_absent_paths=create_paths,
        create_candidate_paths=create_paths,
        mixed_file_operations=bool(edit_paths and create_paths),
        setup_commands=(CONFIGURE,),
        configure_command=CONFIGURE,
        build_command=BUILD,
        test_command=TEST,
        verification_plan=VERIFICATION,
        oracle_commands=(("python3", str(ORACLE), task_id),),
        seeds=(42,),
        max_mutations=2,
    )


def tasks() -> tuple[FrozenTask, ...]:
    """Return stable definitions; references/wrong states remain evaluator-only."""
    old = {item.metadata.task_id: item for item in realistic_semantic_tasks((42,))}
    adopted = (
        ("C01", "R01", "Python", OperationClass.EDIT_SINGLE),
        ("C02", "R03", "C17", OperationClass.EDIT_SINGLE),
        ("C03", "R04", "C17", OperationClass.EDIT_SINGLE),
        ("C04", "R05", "Python", OperationClass.EDIT_MULTI),
        ("C05", "R07", "C17", OperationClass.EDIT_MULTI),
        ("C06", "R08", "C17", OperationClass.EDIT_MULTI),
    )
    values = []
    for task_id, old_id, language, operation in adopted:
        definition = old[old_id]
        original = definition.metadata.production_task
        converted = replace(
            original,
            task_id=task_id,
            oracle_commands=(("python3", str(ORACLE), task_id),),
            verification_plan=VERIFICATION,
        )
        authority = (
            AuthorityMode.DISCOVERY_REQUIRED
            if operation is OperationClass.EDIT_SINGLE
            else AuthorityMode.TRUSTED_REQUIRED_CANDIDATES
        )
        values.append(
            FrozenTask(
                task_id,
                1,
                language,
                operation,
                authority,
                converted,
                definition.wrong,
            )
        )

    values.extend(
        (
            FrozenTask(
                "C07",
                1,
                "Python",
                OperationClass.CREATE,
                AuthorityMode.PATH_KNOWN,
                _task(
                    "C07",
                    "The scheduler in pyservice/scheduler.py imports a missing "
                    "pyservice/priority.py. Create that module with priority_rank "
                    "for high, normal, and low labels (case-insensitive, trimmed), "
                    "stable high-first ordering, and ValueError for unknown labels. "
                    "Do not modify existing files. Verify the project.",
                    ("pyservice/scheduler.py",),
                    (),
                    ("pyservice/priority.py",),
                ),
                wrong_create_source=(
                    "def priority_rank(value):\n"
                    "    ranks = {'high': 2, 'normal': 1, 'low': 0}\n"
                    "    return ranks[value.strip().casefold()]\n"
                ),
            ),
            FrozenTask(
                "C08",
                1,
                "Python",
                OperationClass.CREATE,
                AuthorityMode.PATH_KNOWN,
                _task(
                    "C08",
                    "pyservice/response_wire.py imports a missing "
                    "pyservice/serialization.py. Create encode_response(status, body) "
                    "there: reject statuses outside 100..599 and frame binary-safe "
                    "bytes as three-digit status, colon, decimal byte length, colon, "
                    "then unchanged body. Do not modify existing files. Verify.",
                    ("pyservice/response_wire.py",),
                    (),
                    ("pyservice/serialization.py",),
                ),
                wrong_create_source=(
                    "def encode_response(status, body):\n"
                    "    return f'{status}:{len(body) + 1}:'.encode() + body\n"
                ),
            ),
            FrozenTask(
                "C09",
                1,
                "C17",
                OperationClass.CREATE,
                AuthorityMode.PATH_KNOWN,
                _task(
                    "C09",
                    "cengine/retry_policy.c calls backoff_delay declared in "
                    "cengine/backoff.h, but cengine/backoff.c is missing. Create "
                    "that source file: start at min(base, cap), double per attempt, "
                    "saturate at cap without unsigned overflow. Do not modify "
                    "existing files. Configure, build, and test.",
                    ("cengine/retry_policy.c", "cengine/backoff.h"),
                    (),
                    ("cengine/backoff.c",),
                ),
                wrong_create_source=(
                    '#include "backoff.h"\n'
                    "unsigned backoff_delay(unsigned attempt,unsigned base,"
                    "unsigned cap)"
                    "{unsigned value=base*(attempt+1);return value>cap?cap:value;}\n"
                ),
            ),
            FrozenTask(
                "C10",
                1,
                "Python",
                OperationClass.MIXED_EDIT_CREATE,
                AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
                _task(
                    "C10",
                    "Update pyservice/dispatch.py so action dispatch remains "
                    "case-insensitive after trimming and rejects missing or unknown "
                    "actions. Create its missing pyservice/dispatch_rules.py module "
                    "with choose_handler mapping read, write, delete to reader, "
                    "writer, remover. Both paths are required. Verify.",
                    ("pyservice/dispatch.py",),
                    ("pyservice/dispatch.py",),
                    ("pyservice/dispatch_rules.py",),
                    setup=(
                        SetupReplacement(
                            "pyservice/dispatch.py",
                            "return choose_handler(action.strip().casefold())",
                            "return choose_handler(action.strip())",
                        ),
                    ),
                ),
                wrong_create_source=(
                    "def choose_handler(action):\n"
                    "    handlers = {'read':'reader','write':'reader',"
                    "'delete':'remover'}\n"
                    "    return handlers[action]\n"
                ),
            ),
            FrozenTask(
                "C11",
                1,
                "Python",
                OperationClass.MIXED_EDIT_CREATE,
                AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
                _task(
                    "C11",
                    "Fix pyservice/billing.py so negative cents always raise "
                    "ValueError. Create missing pyservice/currency.py with "
                    "format_cents returning exact two-decimal dollar strings "
                    "without floating point. Both paths must change. Verify.",
                    ("pyservice/billing.py",),
                    ("pyservice/billing.py",),
                    ("pyservice/currency.py",),
                    setup=(
                        SetupReplacement(
                            "pyservice/billing.py",
                            "if amount_cents < 0:",
                            "if amount_cents < -1:",
                        ),
                    ),
                ),
                wrong_create_source=(
                    "def format_cents(amount_cents):\n"
                    "    return f'${amount_cents//100}.{amount_cents%100}'\n"
                ),
            ),
            FrozenTask(
                "C12",
                1,
                "C17",
                OperationClass.MIXED_EDIT_CREATE,
                AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
                _task(
                    "C12",
                    "Fix cengine/health.c so nonpositive limits return -1 and "
                    "failures at the limit are unhealthy. Create its missing "
                    "cengine/health_policy.c implementation of health_is_healthy "
                    "declared in health_policy.h. Both paths are required. "
                    "Configure, build, and test.",
                    ("cengine/health.c", "cengine/health_policy.h"),
                    ("cengine/health.c",),
                    ("cengine/health_policy.c",),
                    setup=(
                        SetupReplacement(
                            "cengine/health.c",
                            "if (limit <= 0) {",
                            "if (limit < 0) {",
                        ),
                    ),
                ),
                wrong_create_source=(
                    '#include "health_policy.h"\n'
                    "int health_is_healthy(unsigned failures,unsigned limit)"
                    "{return failures<=limit;}\n"
                ),
            ),
        )
    )
    return tuple(values)


def manifest_identity(definitions: tuple[FrozenTask, ...]) -> str:
    """Bind task texts, setup, wrong states, and canonical source bytes."""
    manifest = []
    for definition in definitions:
        task = definition.production_task
        manifest.append(
            {
                "id": definition.versioned_id,
                "language": definition.language,
                "class": definition.operation_class.value,
                "authority": definition.authority_mode.value,
                "prompt": task.prompt,
                "edits": definition.edit_paths,
                "creates": definition.create_paths,
                "setup": [
                    (item.path, item.expected, item.replacement) for item in task.setup
                ],
                "wrong": [
                    (item.path, item.expected, item.replacement)
                    for item in definition.wrong_replacements
                ],
                "wrong_create": definition.wrong_create_source,
            }
        )
    payload = {
        "repository": hash_workspace(REPOSITORY),
        "oracle_sha256": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "legacy_oracle_sha256": hashlib.sha256(
            (ROOT.parent / "realistic_semantic_v1" / "oracle.py").read_bytes()
        ).hexdigest(),
        "tasks": manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def baseline_workspace(definition: FrozenTask, destination: Path) -> Path:
    workspace = copy_repository(REPOSITORY, destination)
    apply_task_setup(workspace, definition.production_task.setup)
    apply_task_absences(workspace, definition.create_paths)
    return workspace


def _restore(workspace: Path, paths: tuple[str, ...]) -> None:
    for path in paths:
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / path).read_bytes())


def _oracle(workspace: Path, task_id: str) -> bool:
    return (
        run_oracle(workspace, (("python3", str(ORACLE), task_id),))
        is EvaluationOutcome.PASS
    )


def _verification(workspace: Path) -> bool:
    return run_oracle(workspace, (CONFIGURE, BUILD, TEST)) is EvaluationOutcome.PASS


def validate_integrity(definitions: tuple[FrozenTask, ...]) -> tuple[Integrity, ...]:
    """Prove each scored task's baseline, reference, wrong and partial states."""
    canonical_before = hash_workspace(REPOSITORY)
    results = []
    for definition in definitions:
        with tempfile.TemporaryDirectory(
            prefix=f"forge-a56-{definition.task_id}-"
        ) as name:
            root = Path(name).resolve()
            baseline = baseline_workspace(definition, root / "baseline")
            baseline_pass = _oracle(baseline, definition.task_id)
            baseline_verification = _verification(baseline)

            reference = root / "reference"
            shutil.copytree(
                baseline,
                reference,
                ignore=shutil.ignore_patterns("build", "__pycache__"),
            )
            _restore(reference, definition.production_task.expected_changed_paths)
            reference_pass = _oracle(reference, definition.task_id)
            reference_verification = _verification(reference)

            wrong = root / "wrong"
            shutil.copytree(
                baseline, wrong, ignore=shutil.ignore_patterns("build", "__pycache__")
            )
            if definition.wrong_create_source is not None:
                _restore(wrong, definition.edit_paths)
                (wrong / definition.create_paths[0]).write_text(
                    definition.wrong_create_source, encoding="utf-8"
                )
            elif definition.wrong_replacements:
                apply_task_setup(wrong, definition.wrong_replacements)
            else:
                _restore(wrong, definition.edit_paths[1:])
            wrong_pass = _oracle(wrong, definition.task_id)
            wrong_verification = _verification(wrong)

            partial_results = []
            if definition.operation_class is OperationClass.EDIT_MULTI:
                for path in definition.edit_paths:
                    partial = root / ("partial-" + Path(path).name)
                    shutil.copytree(
                        baseline,
                        partial,
                        ignore=shutil.ignore_patterns("build", "__pycache__"),
                    )
                    _restore(partial, (path,))
                    partial_results.append(_oracle(partial, definition.task_id))
            elif definition.operation_class is OperationClass.MIXED_EDIT_CREATE:
                for index, paths in enumerate(
                    (definition.edit_paths, definition.create_paths)
                ):
                    partial = root / f"partial-{index}"
                    shutil.copytree(
                        baseline,
                        partial,
                        ignore=shutil.ignore_patterns("build", "__pycache__"),
                    )
                    _restore(partial, paths)
                    partial_results.append(_oracle(partial, definition.task_id))
            partial_pass = any(partial_results)
            eligible = (
                not baseline_pass
                and reference_pass
                and not wrong_pass
                and not partial_pass
            )
            results.append(
                Integrity(
                    definition.task_id,
                    baseline_pass,
                    reference_pass,
                    wrong_pass,
                    partial_pass,
                    baseline_verification,
                    reference_verification,
                    wrong_verification,
                    eligible,
                )
            )
    if canonical_before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical realistic-coding-v2 repository changed")
    return tuple(results)

"""Frozen Foundation task corpus and source snapshot support for A60."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldLevel,
    RealWorldTask,
    RepositorySnapshot,
    SetupReplacement,
    apply_task_setup,
    run_oracle,
)
from forge.interaction import AutonomyMode
from forge.project_config import VerificationPlan

SUITE = "real-repository-pilot-v1"
VERSION = 1
SEED = 42
ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / "oracle.py"
CONFIGURE = ("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON")
BUILD = ("cmake", "--build", "build")
TEST = (
    "ctest",
    "--test-dir",
    "build",
    "--output-on-failure",
    "--timeout",
    "10",
)
VERIFICATION = VerificationPlan(
    "foundation-configure-build-test",
    ("project.configure", "project.build", "project.test"),
)
SOURCE_SUFFIXES = frozenset({".c", ".h"})
EXCLUDED_PARTS = frozenset({".git", ".godot", "build", "dist", "__pycache__"})


class OperationClass(StrEnum):
    EDIT_SINGLE = "EDIT_SINGLE"
    EDIT_MULTI = "EDIT_MULTI"


class AuthorityMode(StrEnum):
    DISCOVERY_REQUIRED = "DISCOVERY_REQUIRED"
    PATH_KNOWN = "PATH_KNOWN"


@dataclass(frozen=True, slots=True)
class PilotTask:
    task_id: str
    version: int
    operation_class: OperationClass
    authority_mode: AuthorityMode
    production_task: RealWorldTask
    oracle_pattern: str
    wrong_available: bool = True

    @property
    def versioned_id(self) -> str:
        return f"{self.task_id}-v{self.version}"

    @property
    def edit_paths(self) -> tuple[str, ...]:
        return self.production_task.expected_changed_paths

    @property
    def create_paths(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Integrity:
    task_id: str
    baseline_oracle: bool
    reference_oracle: bool
    wrong_oracle: bool | None
    baseline_verification: bool
    reference_verification: bool
    wrong_verification: bool | None
    alignment: str
    eligible: bool


def _task(
    task_id: str,
    prompt: str,
    paths: tuple[str, ...],
    setup: tuple[SetupReplacement, ...],
    oracle_pattern: str,
    *,
    authority: AuthorityMode = AuthorityMode.DISCOVERY_REQUIRED,
) -> PilotTask:
    operation = (
        OperationClass.EDIT_SINGLE if len(paths) == 1 else OperationClass.EDIT_MULTI
    )
    task = RealWorldTask(
        task_id,
        RealWorldLevel.BOUNDED_REPAIR,
        AutonomyMode.REPAIR,
        prompt,
        paths,
        allowed_paths=paths,
        expected_changed_paths=paths,
        required_candidate_paths=paths,
        setup=setup,
        configure_command=CONFIGURE,
        build_command=BUILD,
        test_command=TEST,
        verification_plan=VERIFICATION,
        oracle_commands=(("python3", str(ORACLE), task_id),),
        seeds=(SEED,),
        max_mutations=2,
    )
    return PilotTask(task_id, 1, operation, authority, task, oracle_pattern)


def tasks() -> tuple[PilotTask, ...]:
    """Return all definitions at once so they can be frozen before model loading."""
    return (
        _task(
            "F01",
            "Fix the simulation clock so ordinary ticks advance while the maximum "
            "tick remains saturated, then run the full project verification.",
            ("src/clock.c",),
            (
                SetupReplacement(
                    "src/clock.c",
                    "clock->tick != UINT64_MAX",
                    "clock->tick == UINT64_MAX",
                ),
            ),
            "factory_solar_tests",
        ),
        _task(
            "F02",
            "Correct the solar boundary so the deterministic dusk ramp continues "
            "until sunset and sunset itself has zero intensity, then verify.",
            ("src/clock.c",),
            (
                SetupReplacement(
                    "src/clock.c",
                    "time_of_day >= FACTORY_CLOCK_SUNSET",
                    "time_of_day >= FACTORY_CLOCK_FULL_DAYLIGHT_END",
                ),
            ),
            "factory_solar_tests",
        ),
        _task(
            "F03",
            "Repair entity liveness checks so only allocated, nonzero identifiers are "
            "reported live after arbitrary create and destroy operations. Verify it.",
            ("src/entity.c",),
            (
                SetupReplacement(
                    "src/entity.c",
                    "if (manager->live_ids[index] == id) {\n            return true;",
                    "if (manager->live_ids[index] != id) {\n            return true;",
                ),
            ),
            "factory_entity_tests",
        ),
        _task(
            "F04",
            "Reject refinery placement commands whose input and output directions are "
            "the same while retaining all valid distinct orientations. Verify it.",
            ("src/command.c",),
            (
                SetupReplacement(
                    "src/command.c",
                    "command->data.place_refinery.input_direction != direction",
                    "command->data.place_refinery.input_direction == direction",
                ),
            ),
            "factory_refinery_tests",
        ),
        _task(
            "F05",
            "Fix removal from both transport-belt and storage component stores so "
            "removing one entity preserves dense lookup of every survivor. Verify "
            "all tests.",
            ("src/belt.c", "src/storage.c"),
            (
                SetupReplacement(
                    "src/belt.c",
                    "store->items[index] = store->items[store->count];",
                    "store->items[index] = store->items[0];",
                ),
                SetupReplacement(
                    "src/storage.c",
                    "store->items[index] = store->items[store->count];",
                    "store->items[index] = store->items[0];",
                ),
            ),
            "factory_(belt|storage|demolition)_tests",
        ),
        _task(
            "F06",
            "Restore initial routing state for splitters and pickup geometry for "
            "inserters so first transfers use the documented output and opposite "
            "pickup side.",
            ("src/splitter.c", "src/inserter.c"),
            (
                SetupReplacement(
                    "src/splitter.c",
                    "splitter->next_output = FACTORY_SPLITTER_OUTPUT_LEFT;",
                    "splitter->next_output = FACTORY_SPLITTER_OUTPUT_RIGHT;",
                ),
                SetupReplacement(
                    "src/inserter.c",
                    "(FactoryDirection)(((int)facing + 2) % 4)",
                    "(FactoryDirection)(((int)facing + 1) % 4)",
                ),
            ),
            "factory_(splitter|inserter)_tests",
        ),
        _task(
            "F07",
            "Fix advanced-science handling in src/storage.c and src/telemetry.c so the "
            "last valid item remains readable and reportable without widening the "
            "enum range.",
            ("src/storage.c", "src/telemetry.c"),
            (
                SetupReplacement(
                    "src/storage.c",
                    "*out_amount = storage->advanced_science_amount;",
                    "*out_amount = storage->basic_science_amount;",
                ),
                SetupReplacement(
                    "src/telemetry.c",
                    "item<=FACTORY_ITEM_ADVANCED_SCIENCE",
                    "item<FACTORY_ITEM_ADVANCED_SCIENCE",
                ),
            ),
            "factory_(storage|telemetry|advanced_science)_tests",
            authority=AuthorityMode.PATH_KNOWN,
        ),
        _task(
            "F08",
            "Repair src/snapshot.c and src/presentation.c so valid current snapshots "
            "load and rebuilt presentation snapshots expose every live entity.",
            ("src/snapshot.c", "src/presentation.c"),
            (
                SetupReplacement(
                    "src/snapshot.c",
                    "if (version != FACTORY_SNAPSHOT_VERSION) {",
                    "if (version == FACTORY_SNAPSHOT_VERSION) {",
                ),
                SetupReplacement(
                    "src/presentation.c",
                    "return snapshot == NULL ? 0U : snapshot->entity_count;",
                    "return snapshot == NULL || snapshot->entity_count == 0U ? "
                    "0U : snapshot->entity_count - 1U;",
                ),
            ),
            "factory_(snapshot|presentation)_tests",
            authority=AuthorityMode.PATH_KNOWN,
        ),
    )


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(
        part in EXCLUDED_PARTS or part.startswith("build") for part in relative.parts
    ):
        return False
    if path.suffix in {".o", ".os", ".dylib", ".so", ".a"} or path.name == ".DS_Store":
        return False
    return path.is_file()


def source_manifest(root: Path) -> tuple[tuple[str, str], ...]:
    values = []
    for path in sorted(root.rglob("*")):
        if _included(path, root):
            values.append(
                (
                    path.relative_to(root).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(values)


def repository_identity(root: Path) -> str:
    return hashlib.sha256(
        json.dumps(source_manifest(root), separators=(",", ":")).encode()
    ).hexdigest()


def create_snapshot(canonical: Path, destination: Path) -> Path:
    if destination.resolve() == canonical.resolve():
        raise ValueError("canonical Foundation root cannot be a pilot target")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, _digest in source_manifest(canonical):
        source = canonical / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def definition_identity(definitions: tuple[PilotTask, ...], snapshot: Path) -> str:
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "snapshot": repository_identity(snapshot),
        "oracle": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "tasks": [
            {
                "id": item.versioned_id,
                "prompt": item.production_task.prompt,
                "operation": item.operation_class.value,
                "authority": item.authority_mode.value,
                "paths": item.edit_paths,
                "setup": [
                    (change.path, change.expected, change.replacement)
                    for change in item.production_task.setup
                ],
                "oracle_pattern": item.oracle_pattern,
            }
            for item in definitions
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def task_relevant_hashes(
    snapshot: Path, definitions: tuple[PilotTask, ...]
) -> dict[str, dict[str, str]]:
    return {
        item.task_id: {
            path: hashlib.sha256((snapshot / path).read_bytes()).hexdigest()
            for path in item.edit_paths
        }
        for item in definitions
    }


def snapshot_description(
    root: Path, baseline_seconds: float = 0.0
) -> RepositorySnapshot:
    sources = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and "tests" not in path.parts
    ]
    tests = [path for path in root.rglob("tests/*.c") if path.is_file()]
    return RepositorySnapshot(
        "Foundation",
        repository_identity(root),
        "C17",
        len(sources),
        len(tests),
        sum(
            len(path.read_text(errors="replace").splitlines())
            for path in (*sources, *tests)
        ),
        BUILD,
        TEST,
        EvaluationOutcome.PASS,
        baseline_seconds,
    )


def _run(command: tuple[str, ...], cwd: Path) -> bool:
    try:
        return (
            subprocess.run(
                command,
                cwd=cwd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=120,
            ).returncode
            == 0
        )
    except subprocess.TimeoutExpired:
        return False


def _copy(snapshot: Path, destination: Path) -> Path:
    shutil.copytree(
        snapshot, destination, ignore=shutil.ignore_patterns("build", "__pycache__")
    )
    return destination


def validate_integrity(
    snapshot: Path, definitions: tuple[PilotTask, ...]
) -> tuple[Integrity, ...]:
    """Build baseline/reference/wrong states and classify verification alignment."""
    before = repository_identity(snapshot)
    with tempfile.TemporaryDirectory(prefix="forge-a60-reference-") as name:
        reference = _copy(snapshot, Path(name) / "reference")
        reference_verification = (
            _run(CONFIGURE, reference)
            and _run(BUILD, reference)
            and _run(TEST, reference)
        )
        reference_oracles = {
            item.task_id: run_oracle(reference, item.production_task.oracle_commands)
            is EvaluationOutcome.PASS
            for item in definitions
        }

    def audit(item: PilotTask) -> Integrity:
        with tempfile.TemporaryDirectory(prefix=f"forge-a60-{item.task_id}-") as name:
            baseline = _copy(snapshot, Path(name) / "baseline")
            apply_task_setup(baseline, item.production_task.setup)
            baseline_verification = (
                _run(CONFIGURE, baseline)
                and _run(BUILD, baseline)
                and _run(TEST, baseline)
            )
            baseline_oracle = (
                run_oracle(baseline, item.production_task.oracle_commands)
                is EvaluationOutcome.PASS
            )
        reference_oracle = reference_oracles[item.task_id]
        wrong_oracle = baseline_oracle if item.wrong_available else None
        wrong_verification = baseline_verification if item.wrong_available else None
        if not reference_verification:
            alignment = "REFERENCE_REJECTING"
        elif baseline_verification:
            alignment = "BASELINE_ACCEPTING"
        elif wrong_verification:
            alignment = "WRONG_ACCEPTING"
        else:
            alignment = "FULLY_DISCRIMINATING"
        return Integrity(
            item.task_id,
            baseline_oracle,
            reference_oracle,
            wrong_oracle,
            baseline_verification,
            reference_verification,
            wrong_verification,
            alignment,
            not baseline_oracle and reference_oracle,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(audit, definitions))
    if repository_identity(snapshot) != before:
        raise RuntimeError("frozen Foundation snapshot changed during integrity audit")
    return tuple(values)

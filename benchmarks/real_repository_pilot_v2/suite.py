"""Frozen A62 task definitions with the A61 discovery boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from benchmarks.real_repository_discovery_boundary_v1.suite import (
    AuthorityMode,
    ExpectedImplementationPath,
)
from benchmarks.real_repository_pilot_v1.suite import (
    Integrity,
    OperationClass,
    PilotTask,
    repository_identity,
)
from benchmarks.real_repository_pilot_v1.suite import (
    tasks as a60_tasks,
)
from forge.evaluation.realworld import RealWorldTask

SUITE = "real-repository-pilot-v2"
VERSION = 2
SEED = 42


@dataclass(frozen=True, slots=True)
class PilotV2Task:
    task_id: str
    version: int
    operation_class: OperationClass
    authority_mode: AuthorityMode
    production_task: RealWorldTask
    expected_implementation_paths: tuple[ExpectedImplementationPath, ...]
    oracle_pattern: str
    wrong_available: bool = True

    @property
    def versioned_id(self) -> str:
        return f"{self.task_id}-v{self.version}"

    @property
    def edit_paths(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.expected_implementation_paths)

    @property
    def create_paths(self) -> tuple[str, ...]:
        return ()


def _convert(source: PilotTask) -> PilotV2Task:
    original = source.production_task
    expected = tuple(source.edit_paths)
    if source.authority_mode is AuthorityMode.DISCOVERY_REQUIRED:
        production = replace(
            original,
            expected_files=(),
            allowed_paths=(),
            expected_changed_paths=(),
            required_candidate_paths=(),
        )
    else:
        explicit = tuple(path for path in expected if path in original.prompt)
        if set(explicit) != set(expected):
            raise ValueError("path-known paths must be literal task text")
        production = replace(
            original,
            required_candidate_paths=explicit,
        )
    return PilotV2Task(
        source.task_id,
        VERSION,
        source.operation_class,
        source.authority_mode,
        production,
        tuple(ExpectedImplementationPath(path) for path in expected),
        source.oracle_pattern,
        source.wrong_available,
    )


def tasks() -> tuple[PilotV2Task, ...]:
    return tuple(_convert(item) for item in a60_tasks())


def validate_boundary(task: PilotV2Task) -> None:
    if task.authority_mode is not AuthorityMode.DISCOVERY_REQUIRED:
        return
    production = task.production_task
    supplied = {
        *production.expected_files,
        *production.allowed_paths,
        *production.expected_changed_paths,
        *production.required_candidate_paths,
        *production.create_candidate_paths,
    }
    leaked = supplied.intersection(task.edit_paths)
    if leaked:
        raise ValueError("DISCOVERY_EXPECTATION_LEAK: " + ", ".join(sorted(leaked)))


def definition_identity(definitions: tuple[PilotV2Task, ...], snapshot) -> str:  # type: ignore[no-untyped-def]
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "snapshot": repository_identity(snapshot),
        "tasks": [
            {
                "id": item.versioned_id,
                "prompt": item.production_task.prompt,
                "operation": item.operation_class.value,
                "authority": item.authority_mode.value,
                "expected_paths": item.edit_paths,
                "production_required_paths": (
                    item.production_task.required_candidate_paths
                ),
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


def integrity_payload(values: tuple[Integrity, ...]) -> list[dict[str, object]]:
    return [
        {field: getattr(item, field) for field in item.__dataclass_fields__}
        for item in values
    ]

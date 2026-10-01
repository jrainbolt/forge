"""Evaluation-only reconciliation of independently generated child edits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from forge.tools.repository_write import MultiFilePatchTool
from forge.tools.types import ExecutionContext, StructuredValue


class ChildCompositionFailure(StrEnum):
    CHILD_SCHEMA_INVALID = "CHILD_SCHEMA_INVALID"
    CHILD_SEMANTIC_INCOMPLETE = "CHILD_SEMANTIC_INCOMPLETE"
    MISSING_CHILD = "MISSING_CHILD"
    COMPOSITION_CONFLICT = "COMPOSITION_CONFLICT"
    COMPOSITION_VALID_SEMANTIC_FAIL = "COMPOSITION_VALID_SEMANTIC_FAIL"
    PASS = "PASS"


class CrossFileDiagnosis(StrEnum):
    INDEPENDENT_CHILD_WRONG = "INDEPENDENT_CHILD_WRONG"
    CROSS_FILE_INCONSISTENCY = "CROSS_FILE_INCONSISTENCY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ChildEdit:
    path: str
    expected_sha256: str
    edits: tuple[Mapping[str, str], ...]
    schema_valid: bool = True


@dataclass(frozen=True, slots=True)
class CompositionResult:
    failure: ChildCompositionFailure
    child_validity: tuple[tuple[str, bool], ...]
    complete_child_set: bool
    composition_valid: bool
    arguments: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CompositionExecution:
    output: StructuredValue
    logical_mutations: int
    generation_before: int
    generation_after: int


def _current_hash(workspace: Path, relative: str) -> str | None:
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def compose_child_edits(
    workspace: Path,
    required_paths: tuple[str, ...],
    children: tuple[ChildEdit, ...],
    *,
    workspace_generation: int,
) -> CompositionResult:
    """Compose a complete valid set into the existing A41 transaction schema."""
    validity: list[tuple[str, bool]] = []
    if len({child.path for child in children}) != len(children):
        return CompositionResult(
            ChildCompositionFailure.COMPOSITION_CONFLICT,
            tuple((child.path, False) for child in children),
            False,
            False,
        )
    required = set(required_paths)
    for child in children:
        schema_valid = (
            child.schema_valid
            and child.path in required
            and len(child.expected_sha256) == 64
            and bool(child.edits)
            and all(
                set(edit) == {"old", "new"}
                and isinstance(edit["old"], str)
                and isinstance(edit["new"], str)
                and edit["old"] != edit["new"]
                for edit in child.edits
            )
        )
        current = _current_hash(workspace, child.path)
        valid = schema_valid and current == child.expected_sha256
        validity.append((child.path, valid))
        if not valid:
            return CompositionResult(
                ChildCompositionFailure.CHILD_SCHEMA_INVALID,
                tuple(validity),
                False,
                False,
            )
        old_fragments = [edit["old"] for edit in child.edits]
        if len(set(old_fragments)) != len(old_fragments):
            return CompositionResult(
                ChildCompositionFailure.COMPOSITION_CONFLICT,
                tuple(validity),
                False,
                False,
            )
    actual = {child.path for child in children}
    if actual != required or len(children) != len(required_paths):
        return CompositionResult(
            ChildCompositionFailure.MISSING_CHILD,
            tuple(validity),
            False,
            False,
        )
    patches = tuple(
        {
            "path": child.path,
            "expected_sha256": child.expected_sha256,
            "edits": tuple(dict(edit) for edit in child.edits),
        }
        for child in sorted(children, key=lambda item: item.path)
    )
    identity_patches = [
        {
            "path": patch["path"],
            "expected_sha256": patch["expected_sha256"],
            "edits": [dict(edit) for edit in patch["edits"]],
        }
        for patch in patches
    ]
    group_id = hashlib.sha256(
        json.dumps(
            {
                "workspace_generation": workspace_generation,
                "patches": identity_patches,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return CompositionResult(
        ChildCompositionFailure.PASS,
        tuple(validity),
        True,
        True,
        {
            "group_id": group_id,
            "workspace_generation": workspace_generation,
            "patches": patches,
        },
    )


def source_free_result(result: CompositionResult) -> dict[str, object]:
    """Serialize metrics without source, replacement text, or hidden references."""
    return {
        "failure": result.failure.value,
        "child_validity": result.child_validity,
        "complete_child_set": result.complete_child_set,
        "composition_valid": result.composition_valid,
    }


def execute_composition(
    result: CompositionResult,
    context: ExecutionContext,
    *,
    tool: MultiFilePatchTool | None = None,
) -> CompositionExecution:
    """Execute exactly once through the existing A41 multi-file transaction."""
    if not result.composition_valid or result.arguments is None:
        raise ValueError("only a valid complete composition may be executed")
    generation = result.arguments["workspace_generation"]
    assert isinstance(generation, int)
    output = (tool or MultiFilePatchTool()).execute(result.arguments, context)
    return CompositionExecution(output, 1, generation, generation + 1)

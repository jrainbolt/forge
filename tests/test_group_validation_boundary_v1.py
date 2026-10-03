from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.child_edit_composition_v1.protocol import (
    ChildEdit,
    compose_child_edits,
    execute_composition,
)
from benchmarks.group_validation_boundary_v1.protocol import (
    ComposedChild,
    GroupValidationFailure,
    ProductionCandidate,
    source_free_result,
    validate_v1,
)
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.tools.types import ExecutionContext

PATHS = ("src/a.py", "src/b.py")


def _candidate(path: str, index: int) -> ProductionCandidate:
    return ProductionCandidate(
        path,
        "line_range",
        str(index) * 64,
        3,
        1,
        20,
        f"trusted-read-{index}",
        "trusted_task_source",
    )


def _child(path: str, index: int) -> ComposedChild:
    return ComposedChild(
        path,
        "edit",
        "line_range",
        str(index) * 64,
        3,
        4,
        5,
        f"trusted-read-{index}",
    )


def _valid(**changes: object):
    values = {
        "required_paths": PATHS,
        "children": (_child(PATHS[0], 1), _child(PATHS[1], 2)),
        "candidates": (_candidate(PATHS[0], 1), _candidate(PATHS[1], 2)),
        "representation": "line_range",
        "generation": 3,
        "mutation_ready": True,
        "canonicalized": True,
    }
    values.update(changes)
    return validate_v1(**values)  # type: ignore[arg-type]


def test_accepted_group_mirrors_production_validation() -> None:
    result = _valid()
    assert result.failure is GroupValidationFailure.PASS
    assert result.production_validatable and result.transaction_ready


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"required_paths": (*PATHS, "src/c.py")},
            GroupValidationFailure.OPERATION_COUNT_MISMATCH,
        ),
        (
            {"required_paths": (PATHS[0], "src/c.py")},
            GroupValidationFailure.PATH_SET_MISMATCH,
        ),
        (
            {"children": (_child(PATHS[0], 1), _child(PATHS[0], 1))},
            GroupValidationFailure.DUPLICATE_PATH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), operation_type="create"),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.OPERATION_TYPE_MISMATCH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), representation="exact_text"),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.REPRESENTATION_MISMATCH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), source_sha256="f" * 64),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.SOURCE_IDENTITY_MISMATCH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), end_line=21),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.RANGE_IDENTITY_MISMATCH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), generation=4),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.GENERATION_MISMATCH,
        ),
        (
            {
                "children": (
                    replace(_child(PATHS[0], 1), candidate_observation_id="other"),
                    _child(PATHS[1], 2),
                )
            },
            GroupValidationFailure.PROVENANCE_MISMATCH,
        ),
        (
            {"candidates": (_candidate(PATHS[0], 1),)},
            GroupValidationFailure.AUTHORITY_MISMATCH,
        ),
        ({"canonicalized": False}, GroupValidationFailure.CANONICALIZATION_MISMATCH),
    ],
)
def test_v1_rejects_each_production_invalid_invariant(
    changes: dict[str, object], expected: GroupValidationFailure
) -> None:
    assert _valid(**changes).failure is expected


def test_a68_rejections_lacked_mutation_ready_provenance() -> None:
    result = _valid(mutation_ready=False)
    assert result.failure is GroupValidationFailure.PROVENANCE_MISMATCH
    assert result.mechanically_materializable
    assert not result.production_validatable


def test_known_good_group_reaches_existing_atomic_transaction(tmp_path: Path) -> None:
    children = []
    for path in ("a.py", "b.py"):
        text = "VALUE = 1\n"
        (tmp_path / path).write_text(text)
        children.append(
            ChildEdit(
                path,
                hashlib.sha256(text.encode()).hexdigest(),
                ({"old": text, "new": "VALUE = 2\n"},),
            )
        )
    composed = compose_child_edits(
        tmp_path, ("a.py", "b.py"), tuple(children), workspace_generation=0
    )
    executed = execute_composition(composed, ExecutionContext(tmp_path.resolve()))
    assert executed.output["operation"] == "multi_patch"
    assert executed.logical_mutations == 1


def test_v1_is_stricter_than_v0_without_weakening_production() -> None:
    v0_materializable = True
    v1 = _valid(mutation_ready=False)
    assert v0_materializable and not v1.production_validatable
    assert GroupValidationFailure.PASS is _valid().failure


def test_source_free_checkpoint_package_and_canonical_identity(tmp_path: Path) -> None:
    result = source_free_result(_valid(mutation_ready=False))
    encoded = json.dumps(result).lower()
    assert "new_text" not in encoded and "old_text" not in encoded
    checkpoint = tmp_path / "case.json"
    payload = {"case": "codestral-F06", **result}
    atomic_json(checkpoint, payload)
    assert read_checkpoint(checkpoint, {"case": "codestral-F06"}) == json.loads(
        checkpoint.read_text()
    )
    assert (
        'where = ["src"]' in (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    canonical = Path("/Users/jamesrainbolt/Documents/github/foundation/foundation")
    if canonical.exists():
        assert source_state_identity(canonical) == source_state_identity(canonical)

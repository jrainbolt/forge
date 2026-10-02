from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.child_edit_composition_v1.protocol import ChildEdit, compose_child_edits
from benchmarks.child_mutation_diagnosis_v1.protocol import (
    MaterializationFailure,
    audit_child,
    source_free_audit,
)
from benchmarks.child_mutation_diagnosis_v1.suite import (
    ChildCondition,
    validate_conditions,
)
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from forge.evaluation.replay import source_state_identity


def _source(root: Path, text: str = "one\ntwo\nthree\n") -> str:
    path = root / "src/a.py"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _range(start: int, end: int, new: str, path: str = "src/a.py") -> str:
    return json.dumps(
        {
            "type": "line_range_edit",
            "path": path,
            "start_line": start,
            "end_line": end,
            "new_text": new,
        }
    )


def test_valid_range_records_source_free_line_metadata(tmp_path: Path) -> None:
    digest = _source(tmp_path)
    result = audit_child(
        _range(2, 2, "changed\n"),
        tmp_path,
        "src/a.py",
        representation="line_range",
        trusted_sha256=digest,
    )
    assert result.failure is MaterializationFailure.VALID_CHILD
    assert result.materializable and result.bounds_exist
    assert result.selected_source_sha256 == hashlib.sha256(b"two\n").hexdigest()
    assert "changed" not in json.dumps(source_free_audit(result))


def test_out_of_bounds_range_is_precise(tmp_path: Path) -> None:
    digest = _source(tmp_path)
    result = audit_child(
        _range(2, 20, "changed\n"),
        tmp_path,
        "src/a.py",
        representation="line_range",
        trusted_sha256=digest,
    )
    assert result.failure is MaterializationFailure.INVALID_RANGE_BOUNDS
    assert result.bounds_exist is False


def test_stale_source_is_range_source_mismatch(tmp_path: Path) -> None:
    _source(tmp_path)
    result = audit_child(
        _range(1, 1, "changed\n"),
        tmp_path,
        "src/a.py",
        representation="line_range",
        trusted_sha256="0" * 64,
    )
    assert result.failure is MaterializationFailure.RANGE_SOURCE_MISMATCH


def test_no_op_replacement_is_precise(tmp_path: Path) -> None:
    digest = _source(tmp_path)
    result = audit_child(
        _range(2, 2, "two\n"),
        tmp_path,
        "src/a.py",
        representation="line_range",
        trusted_sha256=digest,
    )
    assert result.failure is MaterializationFailure.NO_OP
    assert result.replacement_material is False


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_range(1, 1, "x", "src/b.py"), MaterializationFailure.UNAUTHORIZED_PATH),
        (json.dumps({"type": "line_range_edit"}), MaterializationFailure.WRONG_PATH),
        ("not-json", MaterializationFailure.MALFORMED_SCHEMA),
        ("", MaterializationFailure.MISSING_CHILD),
    ],
)
def test_path_and_schema_failures_are_distinct(
    tmp_path: Path, payload: str, expected: MaterializationFailure
) -> None:
    digest = _source(tmp_path)
    assert (
        audit_child(
            payload,
            tmp_path,
            "src/a.py",
            representation="line_range",
            trusted_sha256=digest,
        ).failure
        is expected
    )


def test_exact_text_audit_records_found_unique_and_materiality(tmp_path: Path) -> None:
    digest = _source(tmp_path)
    valid = json.dumps(
        {
            "type": "structured_edit",
            "path": "src/a.py",
            "old_text": "two\n",
            "new_text": "changed\n",
        }
    )
    result = audit_child(
        valid,
        tmp_path,
        "src/a.py",
        representation="exact_text",
        trusted_sha256=digest,
    )
    assert result.failure is MaterializationFailure.VALID_CHILD
    assert result.old_text_found and result.old_text_unique
    missing = valid.replace("two\\n", "absent\\n")
    result = audit_child(
        missing,
        tmp_path,
        "src/a.py",
        representation="exact_text",
        trusted_sha256=digest,
    )
    assert result.failure is MaterializationFailure.REPLACEMENT_SOURCE_MISMATCH


def test_d0_d1_match_and_d2_only_adds_trusted_context() -> None:
    base = ChildCondition("task", "src/a.py", "a" * 64, ("src/a.py",), "line_range")
    d2 = ChildCondition(
        "task",
        "src/a.py",
        "a" * 64,
        ("src/a.py",),
        "line_range",
        ("src/b.py",),
    )
    validate_conditions(base, base, d2, ("src/a.py", "src/b.py"))
    with pytest.raises(ValueError, match="already be trusted"):
        validate_conditions(base, base, d2, ("src/a.py",))


def test_no_representation_switching() -> None:
    d0 = ChildCondition("task", "a", "a" * 64, ("a",), "line_range")
    d1 = ChildCondition("task", "a", "a" * 64, ("a",), "exact_text")
    with pytest.raises(ValueError, match="switching"):
        validate_conditions(d0, d1, d0, ("a",))


def test_complete_set_required_before_composition(tmp_path: Path) -> None:
    digest = _source(tmp_path, "VALUE = 1\n")
    child = ChildEdit(
        "src/a.py", digest, ({"old": "VALUE = 1\n", "new": "VALUE = 2\n"},)
    )
    result = compose_child_edits(
        tmp_path, ("src/a.py", "src/b.py"), (child,), workspace_generation=0
    )
    assert not result.composition_valid


def test_checkpoint_identity_hidden_reference_and_package_exclusion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "cell.json"
    payload = {
        "task_id": "F05",
        "condition": "D1",
        "failure": MaterializationFailure.INVALID_RANGE_BOUNDS.value,
    }
    atomic_json(checkpoint, payload)
    assert read_checkpoint(checkpoint, {"task_id": "F05", "condition": "D1"}) == payload
    assert "reference" not in json.dumps(payload).lower()
    project = Path(__file__).parents[1] / "pyproject.toml"
    assert 'where = ["src"]' in project.read_text()


def test_canonical_source_identity_is_deterministic() -> None:
    canonical = Path("/Users/jamesrainbolt/Documents/github/foundation/foundation")
    if canonical.exists():
        assert source_state_identity(canonical) == source_state_identity(canonical)

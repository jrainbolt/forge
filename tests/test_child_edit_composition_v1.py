from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.child_edit_composition_v1.protocol import (
    ChildCompositionFailure,
    ChildEdit,
    compose_child_edits,
    execute_composition,
    source_free_result,
)
from benchmarks.child_edit_composition_v1.suite import (
    MULTI_TASK_IDS,
    SINGLE_CONTROL_IDS,
)
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from benchmarks.real_repository_pilot_v2.suite import tasks
from forge.evaluation.replay import source_manifest, source_state_identity
from forge.tools.repository_write import MultiFilePatchTool
from forge.tools.types import ExecutionContext


def _write(root: Path, relative: str, text: str = "VALUE = 1\n") -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _child(root: Path, relative: str) -> ChildEdit:
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return ChildEdit(
        relative,
        digest,
        ({"old": "VALUE = 1\n", "new": "VALUE = 2\n"},),
    )


def test_source_identity_ignores_git_build_and_mutable_metadata(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.c", "int value = 1;\n")
    before = source_state_identity(tmp_path)
    _write(tmp_path, ".git/HEAD", "ref: one\n")
    _write(tmp_path, "build-stale/cache.txt", "generated\n")
    _write(tmp_path, ".pytest_cache/state.json", "{}\n")
    (tmp_path / ".DS_Store").write_bytes(b"metadata")
    assert source_state_identity(tmp_path) == before
    assert source_manifest(tmp_path) == (
        ("src/a.c", hashlib.sha256(b"int value = 1;\n").hexdigest()),
    )


def test_source_modification_changes_identity(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.c", "int value = 1;\n")
    before = source_state_identity(tmp_path)
    _write(tmp_path, "src/a.c", "int value = 2;\n")
    assert source_state_identity(tmp_path) != before


def test_complete_children_compose_and_use_existing_transaction(tmp_path: Path) -> None:
    paths = ("src/a.py", "src/b.py")
    for path in paths:
        _write(tmp_path, path)
    result = compose_child_edits(
        tmp_path,
        paths,
        tuple(_child(tmp_path, path) for path in paths),
        workspace_generation=7,
    )
    assert result.failure is ChildCompositionFailure.PASS
    assert result.complete_child_set and result.composition_valid
    execution = execute_composition(result, ExecutionContext(tmp_path.resolve()))
    assert execution.logical_mutations == 1
    assert (execution.generation_before, execution.generation_after) == (7, 8)
    assert execution.output["operation"] == "multi_patch"
    assert all((tmp_path / path).read_text() == "VALUE = 2\n" for path in paths)


def test_partial_child_set_is_rejected_without_writes(tmp_path: Path) -> None:
    paths = ("a.py", "b.py")
    for path in paths:
        _write(tmp_path, path)
    result = compose_child_edits(
        tmp_path, paths, (_child(tmp_path, "a.py"),), workspace_generation=0
    )
    assert result.failure is ChildCompositionFailure.MISSING_CHILD
    with pytest.raises(ValueError, match="valid complete"):
        execute_composition(result, ExecutionContext(tmp_path.resolve()))
    assert all((tmp_path / path).read_text() == "VALUE = 1\n" for path in paths)


@pytest.mark.parametrize("case", ["unauthorized", "stale"])
def test_unauthorized_or_stale_child_rejects_whole_set(
    tmp_path: Path, case: str
) -> None:
    paths = ("a.py", "b.py")
    for path in (*paths, "c.py"):
        _write(tmp_path, path)
    children = [_child(tmp_path, path) for path in paths]
    if case == "unauthorized":
        children[1] = _child(tmp_path, "c.py")
    else:
        children[1] = ChildEdit("b.py", "0" * 64, children[1].edits)
    result = compose_child_edits(
        tmp_path, paths, tuple(children), workspace_generation=0
    )
    assert result.failure is ChildCompositionFailure.CHILD_SCHEMA_INVALID
    assert all((tmp_path / path).read_text() == "VALUE = 1\n" for path in paths)


def test_conflicting_child_rejected_before_transaction(tmp_path: Path) -> None:
    paths = ("a.py", "b.py")
    for path in paths:
        _write(tmp_path, path)
    first = _child(tmp_path, "a.py")
    duplicate = ChildEdit("a.py", first.expected_sha256, first.edits)
    result = compose_child_edits(
        tmp_path, paths, (first, duplicate), workspace_generation=0
    )
    assert result.failure is ChildCompositionFailure.COMPOSITION_CONFLICT


def test_transaction_failure_has_no_partial_writes(tmp_path: Path) -> None:
    paths = ("a.py", "b.py")
    for path in paths:
        _write(tmp_path, path)
    result = compose_child_edits(
        tmp_path,
        paths,
        tuple(_child(tmp_path, path) for path in paths),
        workspace_generation=0,
    )

    def fail_second(source: Path, target: Path) -> None:
        if target.name == "b.py":
            raise OSError("injected")
        source.replace(target)

    with pytest.raises(Exception, match="original files restored"):
        execute_composition(
            result,
            ExecutionContext(tmp_path.resolve()),
            tool=MultiFilePatchTool(replace_operation=fail_second),
        )
    assert all((tmp_path / path).read_text() == "VALUE = 1\n" for path in paths)


def test_structural_result_is_separate_from_semantic_classification(
    tmp_path: Path,
) -> None:
    paths = ("a.py", "b.py")
    for path in paths:
        _write(tmp_path, path)
    result = compose_child_edits(
        tmp_path,
        paths,
        tuple(_child(tmp_path, path) for path in paths),
        workspace_generation=0,
    )
    assert result.failure is ChildCompositionFailure.PASS
    semantic_failure = ChildCompositionFailure.COMPOSITION_VALID_SEMANTIC_FAIL
    assert semantic_failure is not result.failure


def test_results_are_source_free_and_api_has_no_hidden_reference(
    tmp_path: Path,
) -> None:
    paths = ("a.py", "b.py")
    for path in paths:
        _write(tmp_path, path)
    result = compose_child_edits(
        tmp_path,
        paths,
        tuple(_child(tmp_path, path) for path in paths),
        workspace_generation=0,
    )
    serialized = json.dumps(source_free_result(result))
    assert "VALUE" not in serialized
    assert "reference" not in serialized.lower()


def test_checkpoint_resume_and_package_exclusion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cell.json"
    payload = {"task_id": "F05", "profile": "qwen-small", "failure": "MISSING_CHILD"}
    atomic_json(checkpoint, payload)
    assert (
        read_checkpoint(checkpoint, {"task_id": "F05", "profile": "qwen-small"})
        == payload
    )
    project = Path(__file__).parents[1] / "pyproject.toml"
    packaging = project.read_text()
    assert 'where = ["src"]' in packaging
    assert "benchmarks.child_edit_composition" not in packaging


def test_corpus_has_four_multi_tasks_and_two_single_controls() -> None:
    definitions = {item.task_id: item for item in tasks()}
    assert len(MULTI_TASK_IDS) == 4 and len(SINGLE_CONTROL_IDS) == 2
    assert all(len(definitions[item].edit_paths) > 1 for item in MULTI_TASK_IDS)
    assert all(len(definitions[item].edit_paths) == 1 for item in SINGLE_CONTROL_IDS)

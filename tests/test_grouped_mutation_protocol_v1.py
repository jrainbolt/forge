from __future__ import annotations

import json
from pathlib import Path

from benchmarks.grouped_mutation_protocol_v1.protocol import (
    GroupFailure,
    classify_group_response,
    source_free_classification,
)
from benchmarks.grouped_mutation_protocol_v1.suite import (
    MULTI_TASK_IDS,
    SINGLE_CONTROL_IDS,
    LayerInvariant,
    validate_layers,
)
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from benchmarks.real_repository_pilot_v2.suite import tasks

PATHS = ("src/a.py", "src/b.py")


def _response(edits: list[dict[str, object]]) -> str:
    return json.dumps({"type": "multi_file_structured_edit", "edits": edits})


def _edit(path: str, old: str = "old", new: str = "new") -> dict[str, object]:
    return {"path": path, "old_text": old, "new_text": new}


def test_exact_required_path_set_has_one_operation_per_path() -> None:
    result = classify_group_response(
        _response([_edit(PATHS[0]), _edit(PATHS[1])]),
        PATHS,
        representation="exact_text",
    )
    assert result.failure is GroupFailure.PASS
    assert set(result.paths) == set(PATHS)


def test_duplicate_missing_and_unauthorized_children_are_distinct() -> None:
    duplicate = classify_group_response(
        _response([_edit(PATHS[0]), _edit(PATHS[0])]),
        PATHS,
        representation="exact_text",
    )
    missing = classify_group_response(
        _response([_edit(PATHS[0])]), PATHS, representation="exact_text"
    )
    unauthorized = classify_group_response(
        _response([_edit(PATHS[0]), _edit("src/c.py")]),
        PATHS,
        representation="exact_text",
    )
    assert duplicate.failure is GroupFailure.DUPLICATE_FILE_OPERATION
    assert missing.failure is GroupFailure.MISSING_FILE_OPERATION
    assert unauthorized.failure is GroupFailure.UNAUTHORIZED_PATH


def test_no_op_range_and_semantic_failures_are_separate() -> None:
    no_op = classify_group_response(
        _response([_edit(PATHS[0], "same", "same"), _edit(PATHS[1])]),
        PATHS,
        representation="exact_text",
    )
    invalid_range = classify_group_response(
        json.dumps(
            {
                "type": "multi_file_line_range_edit",
                "edits": [
                    {"path": path, "start_line": 4, "end_line": 2, "new_text": "x"}
                    for path in PATHS
                ],
            }
        ),
        PATHS,
        representation="line_range",
    )
    semantic = classify_group_response(
        _response([_edit(PATHS[0]), _edit(PATHS[1])]),
        PATHS,
        representation="exact_text",
        semantic_pass=False,
    )
    assert no_op.failure is GroupFailure.NO_OP_CHILD
    assert invalid_range.failure is GroupFailure.INVALID_RANGE
    assert semantic.failure is GroupFailure.STRUCTURALLY_VALID_SEMANTIC_FAIL


def test_source_free_result_excludes_edit_content() -> None:
    result = source_free_classification(
        classify_group_response(
            _response([_edit(PATHS[0]), _edit(PATHS[1])]),
            PATHS,
            representation="exact_text",
        )
    )
    assert not {"old_text", "new_text", "content"}.intersection(result)


def test_corpus_freezes_four_multi_tasks_and_two_single_controls() -> None:
    definitions = {item.task_id: item for item in tasks()}
    assert len(MULTI_TASK_IDS) == 4
    assert len(SINGLE_CONTROL_IDS) == 2
    assert all(len(definitions[item].edit_paths) > 1 for item in MULTI_TASK_IDS)
    assert all(len(definitions[item].edit_paths) == 1 for item in SINGLE_CONTROL_IDS)


def test_p0_p1_p2_hold_task_source_authority_and_representation_constant() -> None:
    invariant = LayerInvariant(
        "same task", ("src/a.py", "src/b.py"), PATHS, "exact_text"
    )
    validate_layers(invariant, invariant, invariant)


def test_checkpoint_resume_and_source_free_payload(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    payload = {
        "task_id": "F05",
        "profile": "qwen-small",
        "failure": GroupFailure.MISSING_FILE_OPERATION.value,
        "paths": PATHS,
    }
    atomic_json(path, payload)
    assert read_checkpoint(
        path, {"task_id": "F05", "profile": "qwen-small"}
    ) == json.loads(path.read_text())
    assert not {"old_text", "new_text", "content"}.intersection(payload)

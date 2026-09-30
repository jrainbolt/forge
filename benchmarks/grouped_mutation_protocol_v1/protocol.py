"""Source-free grouped response classification for A65 diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class GroupFailure(StrEnum):
    GROUP_SCHEMA_INVALID = "GROUP_SCHEMA_INVALID"
    MISSING_FILE_OPERATION = "MISSING_FILE_OPERATION"
    DUPLICATE_FILE_OPERATION = "DUPLICATE_FILE_OPERATION"
    WRONG_PATH = "WRONG_PATH"
    UNAUTHORIZED_PATH = "UNAUTHORIZED_PATH"
    NO_OP_CHILD = "NO_OP_CHILD"
    INVALID_RANGE = "INVALID_RANGE"
    INCOMPLETE_CHANGE_SET = "INCOMPLETE_CHANGE_SET"
    STRUCTURALLY_VALID_SEMANTIC_FAIL = "STRUCTURALLY_VALID_SEMANTIC_FAIL"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class GroupClassification:
    failure: GroupFailure
    paths: tuple[str, ...]
    structurally_valid: bool


def classify_group_response(
    text: str,
    required_paths: tuple[str, ...],
    *,
    representation: str,
    semantic_pass: bool | None = None,
) -> GroupClassification:
    """Classify one response without retaining edit content in the result."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return GroupClassification(GroupFailure.GROUP_SCHEMA_INVALID, (), False)
    expected_type = (
        "multi_file_line_range_edit"
        if representation == "line_range"
        else "multi_file_structured_edit"
    )
    if not isinstance(payload, dict) or payload.get("type") != expected_type:
        return GroupClassification(GroupFailure.GROUP_SCHEMA_INVALID, (), False)
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        return GroupClassification(GroupFailure.GROUP_SCHEMA_INVALID, (), False)
    raw_paths = [item.get("path") for item in edits if isinstance(item, dict)]
    if len(raw_paths) != len(edits) or any(
        not isinstance(path, str) for path in raw_paths
    ):
        return GroupClassification(GroupFailure.WRONG_PATH, (), False)
    paths = tuple(str(path) for path in raw_paths)
    if len(set(paths)) != len(paths):
        return GroupClassification(GroupFailure.DUPLICATE_FILE_OPERATION, paths, False)
    unauthorized = set(paths).difference(required_paths)
    if unauthorized:
        return GroupClassification(GroupFailure.UNAUTHORIZED_PATH, paths, False)
    missing = set(required_paths).difference(paths)
    if missing:
        return GroupClassification(GroupFailure.MISSING_FILE_OPERATION, paths, False)
    if len(paths) != len(required_paths):
        return GroupClassification(GroupFailure.INCOMPLETE_CHANGE_SET, paths, False)
    for item in edits:
        assert isinstance(item, dict)
        if representation == "line_range":
            start, end = item.get("start_line"), item.get("end_line")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
            ):
                return GroupClassification(GroupFailure.INVALID_RANGE, paths, False)
            if not isinstance(item.get("new_text"), str):
                return GroupClassification(
                    GroupFailure.GROUP_SCHEMA_INVALID, paths, False
                )
        else:
            old, new = item.get("old_text"), item.get("new_text")
            if not isinstance(old, str) or not isinstance(new, str):
                return GroupClassification(
                    GroupFailure.GROUP_SCHEMA_INVALID, paths, False
                )
            if old == new:
                return GroupClassification(GroupFailure.NO_OP_CHILD, paths, False)
    if semantic_pass is False:
        return GroupClassification(
            GroupFailure.STRUCTURALLY_VALID_SEMANTIC_FAIL, paths, True
        )
    return GroupClassification(GroupFailure.PASS, paths, True)


def source_free_classification(value: GroupClassification) -> dict[str, object]:
    return {
        "failure": value.failure.value,
        "paths": value.paths,
        "structurally_valid": value.structurally_valid,
    }

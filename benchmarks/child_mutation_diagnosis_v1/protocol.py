"""Precise, source-free mechanical diagnosis for A67 child proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MaterializationFailure(StrEnum):
    VALID_CHILD = "VALID_CHILD"
    WRONG_PATH = "WRONG_PATH"
    UNAUTHORIZED_PATH = "UNAUTHORIZED_PATH"
    NO_OP = "NO_OP"
    INVALID_RANGE_BOUNDS = "INVALID_RANGE_BOUNDS"
    RANGE_SOURCE_MISMATCH = "RANGE_SOURCE_MISMATCH"
    REPLACEMENT_SOURCE_MISMATCH = "REPLACEMENT_SOURCE_MISMATCH"
    MALFORMED_SCHEMA = "MALFORMED_SCHEMA"
    MISSING_CHILD = "MISSING_CHILD"
    OTHER_MATERIALIZATION_FAILURE = "OTHER_MATERIALIZATION_FAILURE"


@dataclass(frozen=True, slots=True)
class ChildAudit:
    failure: MaterializationFailure
    path: str | None
    schema_valid: bool
    materializable: bool
    representation: str
    start_line: int | None = None
    end_line: int | None = None
    bounds_exist: bool | None = None
    selected_source_sha256: str | None = None
    replacement_material: bool | None = None
    old_text_found: bool | None = None
    old_text_unique: bool | None = None


def _failed(
    failure: MaterializationFailure,
    representation: str,
    path: str | None = None,
    **values: object,
) -> ChildAudit:
    return ChildAudit(
        failure,
        path,
        failure
        not in {
            MaterializationFailure.MALFORMED_SCHEMA,
            MaterializationFailure.MISSING_CHILD,
        },
        False,
        representation,
        **values,
    )  # type: ignore[arg-type]


def audit_child(
    text: str,
    workspace: Path,
    authorized_path: str,
    *,
    representation: str,
    trusted_sha256: str,
) -> ChildAudit:
    """Classify the first observable mechanical failure without retaining text."""
    if not text.strip():
        return _failed(MaterializationFailure.MISSING_CHILD, representation)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _failed(MaterializationFailure.MALFORMED_SCHEMA, representation)
    if not isinstance(payload, dict):
        return _failed(MaterializationFailure.MALFORMED_SCHEMA, representation)
    expected = (
        "line_range_edit" if representation == "line_range" else "structured_edit"
    )
    if payload.get("type") != expected:
        return _failed(MaterializationFailure.MISSING_CHILD, representation)
    path = payload.get("path")
    if not isinstance(path, str):
        return _failed(MaterializationFailure.WRONG_PATH, representation)
    if path != authorized_path:
        return _failed(MaterializationFailure.UNAUTHORIZED_PATH, representation, path)
    target = workspace / authorized_path
    try:
        source_bytes = target.read_bytes()
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return _failed(
            MaterializationFailure.OTHER_MATERIALIZATION_FAILURE, representation, path
        )
    if hashlib.sha256(source_bytes).hexdigest() != trusted_sha256:
        return _failed(
            MaterializationFailure.RANGE_SOURCE_MISMATCH, representation, path
        )
    if representation == "line_range":
        start, end, replacement = (
            payload.get("start_line"),
            payload.get("end_line"),
            payload.get("new_text"),
        )
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(replacement, str)
        ):
            return _failed(
                MaterializationFailure.MALFORMED_SCHEMA, representation, path
            )
        lines = source.splitlines(keepends=True)
        bounds = start >= 1 and end >= start and end <= len(lines)
        if not bounds:
            return _failed(
                MaterializationFailure.INVALID_RANGE_BOUNDS,
                representation,
                path,
                start_line=start,
                end_line=end,
                bounds_exist=False,
            )
        original = "".join(lines[start - 1 : end])
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        material = bool(replacement) and replacement != original
        if not material:
            return _failed(
                MaterializationFailure.NO_OP,
                representation,
                path,
                start_line=start,
                end_line=end,
                bounds_exist=True,
                selected_source_sha256=digest,
                replacement_material=False,
            )
        return ChildAudit(
            MaterializationFailure.VALID_CHILD,
            path,
            True,
            True,
            representation,
            start,
            end,
            True,
            digest,
            True,
        )
    old, new = payload.get("old_text"), payload.get("new_text")
    if not isinstance(old, str) or not isinstance(new, str):
        return _failed(MaterializationFailure.MALFORMED_SCHEMA, representation, path)
    if old == new or not new:
        return _failed(
            MaterializationFailure.NO_OP,
            representation,
            path,
            replacement_material=False,
        )
    occurrences = source.count(old) if old else 0
    if occurrences == 0:
        return _failed(
            MaterializationFailure.REPLACEMENT_SOURCE_MISMATCH,
            representation,
            path,
            replacement_material=True,
            old_text_found=False,
            old_text_unique=False,
        )
    if occurrences != 1:
        return _failed(
            MaterializationFailure.REPLACEMENT_SOURCE_MISMATCH,
            representation,
            path,
            replacement_material=True,
            old_text_found=True,
            old_text_unique=False,
        )
    return ChildAudit(
        MaterializationFailure.VALID_CHILD,
        path,
        True,
        True,
        representation,
        replacement_material=True,
        old_text_found=True,
        old_text_unique=True,
    )


def source_free_audit(value: ChildAudit) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}

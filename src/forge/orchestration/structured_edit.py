"""Bounded exact-replacement proposals for mutation-ready coding tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from forge.orchestration.coding_task import MutationCandidate
from forge.tools.paths import WorkspacePathError, resolve_workspace_write_path

MAX_EDIT_TEXT_BYTES = 16 * 1024
MAX_EDIT_TOTAL_BYTES = 24 * 1024
MAX_EDIT_LINES = 200
MAX_GROUPED_FILES = 4


@dataclass(frozen=True, slots=True)
class StructuredEditProposal:
    path: str
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class LineRangeEditProposal:
    """One model-selected 1-based inclusive current-source line range."""

    path: str
    start_line: int
    end_line: int
    new_text: str


@dataclass(frozen=True, slots=True)
class MultiFileStructuredEditProposal:
    edits: tuple[StructuredEditProposal, ...]


@dataclass(frozen=True, slots=True)
class MultiFileLineRangeEditProposal:
    edits: tuple[LineRangeEditProposal, ...]


class StructuredEditFailure(Enum):
    NO_OP_EDIT = "no_op_edit"
    MATERIALIZED_NO_DELTA = "materialized_no_delta"
    OLD_TEXT_NOT_FOUND = "old_text_not_found"
    OLD_TEXT_AMBIGUOUS = "old_text_ambiguous"
    PATH_NOT_ELIGIBLE = "path_not_eligible"
    STALE_SOURCE = "stale_source"
    OUT_OF_RANGE = "out_of_range"
    TOO_LARGE = "too_large"
    INVALID_ENCODING = "invalid_encoding"
    MATERIALIZATION_FAILED = "materialization_failed"
    DUPLICATE_PATH = "duplicate_path"
    GROUP_BOUNDS = "group_bounds"


@dataclass(frozen=True, slots=True)
class StructuredEditValidation:
    failure: StructuredEditFailure | None
    arguments: dict[str, object] | None = None
    start_line: int | None = None
    end_line: int | None = None

    @property
    def valid(self) -> bool:
        return self.failure is None and self.arguments is not None


@dataclass(frozen=True, slots=True)
class MultiFileEditValidation:
    failure: StructuredEditFailure | None
    arguments: dict[str, object] | None = None
    ranges: tuple[tuple[str, int, int], ...] = ()
    failed_path: str | None = None

    @property
    def valid(self) -> bool:
        return self.failure is None and self.arguments is not None


def validate_multi_file_structured_edit(
    proposal: MultiFileStructuredEditProposal,
    candidates: tuple[MutationCandidate, ...],
    workspace: Path,
    generation: int,
) -> MultiFileEditValidation:
    return _validate_group(proposal.edits, candidates, workspace, generation, False)


def validate_multi_file_line_range_edit(
    proposal: MultiFileLineRangeEditProposal,
    candidates: tuple[MutationCandidate, ...],
    workspace: Path,
    generation: int,
) -> MultiFileEditValidation:
    return _validate_group(proposal.edits, candidates, workspace, generation, True)


def _validate_group(
    edits: tuple[StructuredEditProposal, ...] | tuple[LineRangeEditProposal, ...],
    candidates: tuple[MutationCandidate, ...],
    workspace: Path,
    generation: int,
    line_range: bool,
) -> MultiFileEditValidation:
    """Validate and materialize every child before producing one tool invocation."""
    if not 2 <= len(edits) <= MAX_GROUPED_FILES:
        return MultiFileEditValidation(StructuredEditFailure.GROUP_BOUNDS)
    paths = [edit.path for edit in edits]
    if len(paths) != len(set(paths)):
        return MultiFileEditValidation(StructuredEditFailure.DUPLICATE_PATH)
    patches: list[dict[str, object]] = []
    ranges: list[tuple[str, int, int]] = []
    for edit in sorted(edits, key=lambda item: item.path):
        validation = (
            validate_line_range_edit(edit, candidates, workspace, generation)  # type: ignore[arg-type]
            if line_range
            else validate_structured_edit(edit, candidates, workspace, generation)  # type: ignore[arg-type]
        )
        if not validation.valid:
            return MultiFileEditValidation(validation.failure, failed_path=edit.path)
        assert validation.arguments is not None
        assert validation.start_line is not None and validation.end_line is not None
        patches.append(validation.arguments)
        ranges.append((edit.path, validation.start_line, validation.end_line))
    identity_payload = {
        "workspace_generation": generation,
        "patches": patches,
    }
    group_id = hashlib.sha256(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return MultiFileEditValidation(
        None,
        {
            "group_id": group_id,
            "workspace_generation": generation,
            "patches": patches,
        },
        tuple(ranges),
    )


def validate_structured_edit(
    proposal: StructuredEditProposal,
    candidates: tuple[MutationCandidate, ...],
    workspace: Path,
    generation: int,
) -> StructuredEditValidation:
    """Validate and materialize one exact edit as existing A9 patch arguments."""
    candidate = next((item for item in candidates if item.path == proposal.path), None)
    if candidate is None or candidate.generation != generation:
        return StructuredEditValidation(StructuredEditFailure.PATH_NOT_ELIGIBLE)
    if proposal.old_text == proposal.new_text:
        return StructuredEditValidation(StructuredEditFailure.NO_OP_EDIT)
    if not proposal.old_text:
        return StructuredEditValidation(StructuredEditFailure.MATERIALIZATION_FAILED)
    try:
        old_bytes = proposal.old_text.encode("utf-8")
        new_bytes = proposal.new_text.encode("utf-8")
    except UnicodeEncodeError:
        return StructuredEditValidation(StructuredEditFailure.INVALID_ENCODING)
    if (
        len(old_bytes) > MAX_EDIT_TEXT_BYTES
        or len(new_bytes) > MAX_EDIT_TEXT_BYTES
        or len(old_bytes) + len(new_bytes) > MAX_EDIT_TOTAL_BYTES
        or _line_count(proposal.old_text) > MAX_EDIT_LINES
        or _line_count(proposal.new_text) > MAX_EDIT_LINES
    ):
        return StructuredEditValidation(StructuredEditFailure.TOO_LARGE)
    try:
        path = resolve_workspace_write_path(workspace, proposal.path)
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8")
    except WorkspacePathError:
        return StructuredEditValidation(StructuredEditFailure.PATH_NOT_ELIGIBLE)
    except (OSError, UnicodeDecodeError):
        return StructuredEditValidation(StructuredEditFailure.INVALID_ENCODING)
    if hashlib.sha256(source_bytes).hexdigest() != candidate.sha256:
        return StructuredEditValidation(StructuredEditFailure.STALE_SOURCE)

    matches = _match_offsets(source, proposal.old_text)
    if not matches:
        return StructuredEditValidation(StructuredEditFailure.OLD_TEXT_NOT_FOUND)
    authorized = [
        offset
        for offset in matches
        if _inside_range(source, offset, len(proposal.old_text), candidate)
    ]
    if not authorized:
        return StructuredEditValidation(StructuredEditFailure.OUT_OF_RANGE)
    if len(authorized) > 1:
        return StructuredEditValidation(StructuredEditFailure.OLD_TEXT_AMBIGUOUS)
    if len(matches) != len(authorized):
        # A match outside the trusted range does not create ambiguity inside it.
        pass
    offset = authorized[0]
    updated = (
        source[:offset] + proposal.new_text + source[offset + len(proposal.old_text) :]
    )
    if updated == source:
        return StructuredEditValidation(StructuredEditFailure.MATERIALIZED_NO_DELTA)
    if (
        updated[:offset] != source[:offset]
        or updated[offset + len(proposal.new_text) :]
        != source[offset + len(proposal.old_text) :]
    ):
        return StructuredEditValidation(StructuredEditFailure.MATERIALIZATION_FAILED)
    return StructuredEditValidation(
        None,
        {
            "path": proposal.path,
            "expected_sha256": candidate.sha256,
            "edits": [{"old": proposal.old_text, "new": proposal.new_text}],
        },
        source.count("\n", 0, offset) + 1,
        source.count("\n", 0, offset + len(proposal.old_text)) + 1,
    )


def validate_line_range_edit(
    proposal: LineRangeEditProposal,
    candidates: tuple[MutationCandidate, ...],
    workspace: Path,
    generation: int,
) -> StructuredEditValidation:
    """Bind a line range to trusted bytes, then reuse exact-edit validation."""
    candidate = next((item for item in candidates if item.path == proposal.path), None)
    if candidate is None or candidate.generation != generation:
        return StructuredEditValidation(StructuredEditFailure.PATH_NOT_ELIGIBLE)
    try:
        path = resolve_workspace_write_path(workspace, proposal.path)
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8")
    except WorkspacePathError:
        return StructuredEditValidation(StructuredEditFailure.PATH_NOT_ELIGIBLE)
    except (OSError, UnicodeDecodeError):
        return StructuredEditValidation(StructuredEditFailure.INVALID_ENCODING)
    if hashlib.sha256(source_bytes).hexdigest() != candidate.sha256:
        return StructuredEditValidation(StructuredEditFailure.STALE_SOURCE)
    start = proposal.start_line
    end = proposal.end_line
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end < start
        or end - start + 1 > MAX_EDIT_LINES
    ):
        return StructuredEditValidation(StructuredEditFailure.OUT_OF_RANGE)
    if candidate.start_line is None or candidate.end_line is None:
        return StructuredEditValidation(StructuredEditFailure.OUT_OF_RANGE)
    if start < candidate.start_line or end > candidate.end_line:
        return StructuredEditValidation(StructuredEditFailure.OUT_OF_RANGE)
    lines = source.splitlines(keepends=True)
    if end > len(lines):
        return StructuredEditValidation(StructuredEditFailure.OUT_OF_RANGE)
    old_text = "".join(lines[start - 1 : end])
    new_text = _complete_replacement_lines(old_text, proposal.new_text)
    validation = validate_structured_edit(
        StructuredEditProposal(proposal.path, old_text, new_text),
        candidates,
        workspace,
        generation,
    )
    if validation.valid:
        return StructuredEditValidation(
            None, validation.arguments, proposal.start_line, proposal.end_line
        )
    return validation


def _complete_replacement_lines(old_text: str, new_text: str) -> str:
    """Preserve a selected range's terminal line boundary when replacing content."""
    if not new_text or new_text.endswith(("\n", "\r")):
        return new_text
    if old_text.endswith("\r\n"):
        return new_text + "\r\n"
    if old_text.endswith("\n"):
        return new_text + "\n"
    return new_text


def _match_offsets(source: str, old: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = source.find(old, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + len(old)


def _inside_range(
    source: str, offset: int, length: int, candidate: MutationCandidate
) -> bool:
    if candidate.start_line is None or candidate.end_line is None:
        return True
    start_line = source.count("\n", 0, offset) + 1
    end_line = source.count("\n", 0, offset + length) + 1
    if length and source[offset + length - 1 : offset + length] == "\n":
        end_line -= 1
    return candidate.start_line <= start_line and end_line <= candidate.end_line


def _line_count(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") and text else 1)

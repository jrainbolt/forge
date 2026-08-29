"""Bounded exact-replacement proposals for mutation-ready coding tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from forge.orchestration.coding_task import MutationCandidate
from forge.tools.paths import WorkspacePathError, resolve_workspace_write_path

MAX_EDIT_TEXT_BYTES = 16 * 1024
MAX_EDIT_TOTAL_BYTES = 24 * 1024
MAX_EDIT_LINES = 200


@dataclass(frozen=True, slots=True)
class StructuredEditProposal:
    path: str
    old_text: str
    new_text: str


class StructuredEditFailure(Enum):
    OLD_TEXT_NOT_FOUND = "old_text_not_found"
    OLD_TEXT_AMBIGUOUS = "old_text_ambiguous"
    PATH_NOT_ELIGIBLE = "path_not_eligible"
    STALE_SOURCE = "stale_source"
    OUT_OF_RANGE = "out_of_range"
    TOO_LARGE = "too_large"
    INVALID_ENCODING = "invalid_encoding"
    MATERIALIZATION_FAILED = "materialization_failed"


@dataclass(frozen=True, slots=True)
class StructuredEditValidation:
    failure: StructuredEditFailure | None
    arguments: dict[str, object] | None = None

    @property
    def valid(self) -> bool:
        return self.failure is None and self.arguments is not None


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
    if not proposal.old_text or proposal.old_text == proposal.new_text:
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
    )


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

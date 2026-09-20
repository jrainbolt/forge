"""A53 bounded, candidate-authorized creation of new regular text files."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from forge.tools.repository_write import MultiFileMutationPreview, MutationPreview
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    CreateCandidate,
    ExecutionContext,
    StructuredValue,
    ToolCapability,
    ToolEvidence,
    ToolMetadata,
    ToolRisk,
)

MAX_CREATED_FILES = 2
MAX_CREATE_BYTES = 64 * 1024
CREATE_MODE = 0o644
PROTECTED_CREATE_ROOTS = frozenset(
    {
        ".git",
        ".forge-exec",
        ".venv",
        "eval-results",
        "benchmarks",
        "oracle",
        "build",
        "dist",
        "models",
        "__pycache__",
    }
)


@dataclass(frozen=True, slots=True)
class _PreparedCreate:
    candidate: CreateCandidate
    target: Path
    data: bytes
    sha256: str
    staging_seconds: float


def _target(workspace: Path, requested: str) -> tuple[str, Path, os.stat_result]:
    """Reject aliases, symlink parents, and generated/internal destinations."""
    if not isinstance(requested, str) or not requested or "\x00" in requested:
        raise ToolError("create path must be nonempty text without NUL")
    try:
        requested.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ToolError("create path is not valid UTF-8") from error
    if "\\" in requested or "//" in requested or requested.startswith("/"):
        raise ToolError("create path must be normalized workspace-relative POSIX text")
    path = PurePosixPath(requested)
    if (
        path.as_posix() != requested
        or any(part in {".", "..", ""} for part in requested.split("/"))
        or path.parts[0] in PROTECTED_CREATE_ROOTS
        or any(part in PROTECTED_CREATE_ROOTS for part in path.parts)
        or any(part.startswith(".") for part in path.parts)
        or any(":" in part or part.endswith((".", " ")) for part in path.parts)
    ):
        raise ToolError("create path is not an eligible normalized source path")
    parent = workspace
    for part in path.parts[:-1]:
        parent = parent / part
        try:
            info = parent.lstat()
        except OSError as error:
            raise ToolError("create parent directory does not exist") from error
        if not stat.S_ISDIR(info.st_mode):
            raise ToolError("create parent must be a real directory")
    try:
        parent_info = parent.lstat()
    except OSError as error:
        raise ToolError("create parent directory does not exist") from error
    if not stat.S_ISDIR(parent_info.st_mode):
        raise ToolError("create parent must be a real directory")
    target = parent / path.name
    if os.path.lexists(target):
        raise ToolError("create target already exists")
    return requested, target, parent_info


def authorize_create_candidate(
    workspace: Path, path: str, generation: int, provenance: str
) -> CreateCandidate:
    """Called only from trusted task setup with an exact caller-supplied path."""
    if not provenance or generation < 0:
        raise ValueError("create authority requires provenance and generation")
    normalized, _, parent_info = _target(workspace, path)
    return CreateCandidate(
        normalized,
        parent_info.st_dev,
        parent_info.st_ino,
        generation,
        provenance,
    )


def create_group_id(
    creates: tuple[Mapping[str, str], ...],
    candidates: tuple[CreateCandidate, ...],
    generation: int,
) -> str:
    """Bind bytes, path, trusted parent identities, and workspace generation."""
    if len(creates) != len(candidates):
        raise ToolError("create group must match exact authorized candidates")
    try:
        encoded = tuple(item["content"].encode("utf-8") for item in creates)
    except UnicodeEncodeError as error:
        raise ToolError("created content is not valid UTF-8") from error
    identity = {
        "workspace_generation": generation,
        "creates": [
            {
                "path": item["path"],
                "content_sha256": hashlib.sha256(data).hexdigest(),
                "parent_device": candidate.parent_device,
                "parent_inode": candidate.parent_inode,
                "provenance": candidate.provenance,
            }
            for item, candidate, data in zip(creates, candidates, encoded, strict=True)
        ],
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepare(
    arguments: Mapping[str, object], context: ExecutionContext
) -> tuple[tuple[_PreparedCreate, ...], str, int]:
    group_id = arguments.get("group_id")
    generation = arguments.get("workspace_generation")
    creates = arguments.get("creates")
    if not isinstance(group_id, str) or len(group_id) != 64:
        raise ToolError("create group_id must be a SHA-256 identity")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ToolError("create workspace_generation must be nonnegative")
    if not isinstance(creates, (tuple, list)) or not 1 <= len(creates) <= 2:
        raise ToolError("create group requires one or two files")
    if any(
        not isinstance(item, Mapping) or set(item) != {"path", "content"}
        for item in creates
    ):
        raise ToolError("each create requires exactly path and content")
    paths = [item["path"] for item in creates]
    if any(not isinstance(path, str) for path in paths) or paths != sorted(set(paths)):
        raise ToolError("create paths must be unique and canonically ordered")
    authority = {candidate.path: candidate for candidate in context.create_candidates}
    if set(paths) != set(authority):
        raise ToolError("create group must match the exact authorized path set")
    prepared = []
    selected = []
    for item in creates:
        path = item["path"]
        content = item["content"]
        if not isinstance(content, str):
            raise ToolError("created content must be UTF-8 text")
        staging_started = time.perf_counter()
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ToolError("created content is not valid UTF-8") from error
        if len(data) > MAX_CREATE_BYTES:
            raise ToolError("created content exceeds 64 KiB")
        if (
            any(byte < 32 and byte not in {9, 10, 13} for byte in data)
            or b"\x7f" in data
        ):
            raise ToolError("created content contains binary control bytes")
        candidate = authority[path]
        if candidate.generation != generation:
            raise ToolError("create candidate generation is stale")
        _, target, parent_info = _target(context.workspace, path)
        if (parent_info.st_dev, parent_info.st_ino) != (
            candidate.parent_device,
            candidate.parent_inode,
        ):
            raise ToolError("create parent identity changed")
        digest = hashlib.sha256(data).hexdigest()
        prepared.append(
            _PreparedCreate(
                candidate,
                target,
                data,
                digest,
                time.perf_counter() - staging_started,
            )
        )
        selected.append(candidate)
    if group_id != create_group_id(tuple(creates), tuple(selected), generation):
        raise ToolError(
            "create group identity does not match exact content and authority"
        )
    return tuple(prepared), group_id, generation


def preview_create_text_files(
    arguments: Mapping[str, object], context: ExecutionContext
) -> MultiFileMutationPreview:
    prepared, group_id, generation = _prepare(arguments, context)
    files = []
    for item in prepared:
        text = item.data.decode("utf-8")
        diff = "".join(
            difflib.unified_diff(
                [],
                text.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{item.candidate.path}",
            )
        )
        diff = (
            f"CREATE {item.candidate.path} | {len(item.data)} bytes | "
            f"SHA-256 {item.sha256} | mode 0644\n" + (diff or "(empty file)\n")
        )
        files.append(
            MutationPreview(item.candidate.path, "create", diff, None, item.sha256)
        )
    return MultiFileMutationPreview(tuple(files), generation, group_id)


CreateOperation = Callable[[Path, bytes], tuple[int, int]]
RemoveOperation = Callable[[Path], None]


def _exclusive_create(target: Path, data: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), CREATE_MODE)
            os.fsync(stream.fileno())
            info = os.fstat(stream.fileno())
            return info.st_dev, info.st_ino
    except BaseException:
        # A partially written, still-owned file is not a completed child.
        try:
            current = target.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                target.unlink()
        except OSError:
            pass
        raise


class CreateTextFilesTool(Tool):
    """One approval and one handled-failure transaction for one or two files."""

    _metadata = ToolMetadata(
        "repository.create_text_files",
        "Create one or two preauthorized new UTF-8 regular files, without overwrite.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "group_id", ArgumentType.STRING, "Canonical create identity."
                ),
                ArgumentSpec(
                    "workspace_generation", ArgumentType.INTEGER, "Bound generation."
                ),
                ArgumentSpec(
                    "creates",
                    ArgumentType.TEXT_FILE_CREATES,
                    "Ordered exact new files.",
                ),
            )
        ),
        ToolRisk.WRITE,
        ToolEvidence.WRITE_SUCCESS,
        ToolCapability.WRITE,
    )

    def __init__(
        self,
        *,
        create_operation: CreateOperation = _exclusive_create,
        remove_operation: RemoveOperation = Path.unlink,
    ) -> None:
        self._create_operation = create_operation
        self._remove_operation = remove_operation

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        validation_started = time.perf_counter()
        prepared, group_id, generation = _prepare(arguments, context)
        # Complete-group precheck, immediately before the first write.
        for item in prepared:
            _, _, parent = _target(context.workspace, item.candidate.path)
            if (parent.st_dev, parent.st_ino) != (
                item.candidate.parent_device,
                item.candidate.parent_inode,
            ):
                raise ToolError("create parent identity changed before application")
        validation_seconds = time.perf_counter() - validation_started
        application_started = time.perf_counter()
        created: list[tuple[_PreparedCreate, tuple[int, int]]] = []
        in_flight: _PreparedCreate | None = None
        try:
            for item in prepared:
                in_flight = item
                identity = self._create_operation(item.target, item.data)
                created.append((item, identity))
                in_flight = None
                info = item.target.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != CREATE_MODE
                    or (info.st_dev, info.st_ino) != identity
                    or item.target.read_bytes() != item.data
                ):
                    raise ToolError("created file verification failed")
        except Exception as error:
            attempted = bool(created)
            unowned_target_present = in_flight is not None and os.path.lexists(
                in_flight.target
            )
            rollback_started = time.perf_counter()
            try:
                for item, identity in reversed(created):
                    if not os.path.lexists(item.target):
                        continue
                    info = item.target.lstat()
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or (info.st_dev, info.st_ino) != identity
                        or hashlib.sha256(item.target.read_bytes()).hexdigest()
                        != item.sha256
                    ):
                        raise ToolError("created target changed before rollback")
                    self._remove_operation(item.target)
            except Exception as rollback_error:
                raise ToolError(
                    "fatal creation integrity failure: rollback failed",
                    output={
                        "workspace_integrity_failure": True,
                        "mutation_group_apply_result": "failed",
                        "mutation_group_rollback_attempted": True,
                        "mutation_group_rollback_result": "failed",
                        "mutation_group_id": group_id,
                        "rollback_seconds": time.perf_counter() - rollback_started,
                    },
                ) from rollback_error
            if unowned_target_present:
                raise ToolError(
                    "fatal creation integrity failure: unowned target remains",
                    output={
                        "workspace_integrity_failure": True,
                        "mutation_group_apply_result": "failed",
                        "mutation_group_rollback_attempted": attempted,
                        "mutation_group_rollback_result": "failed",
                        "mutation_group_id": group_id,
                        "rollback_seconds": time.perf_counter() - rollback_started,
                    },
                ) from error
            raise ToolError(
                "creation group application failed; created files removed",
                output={
                    "workspace_integrity_failure": False,
                    "mutation_group_apply_result": "failed",
                    "mutation_group_rollback_attempted": attempted,
                    "mutation_group_rollback_result": "restored"
                    if attempted
                    else "not_needed",
                    "mutation_group_id": group_id,
                    "rollback_seconds": time.perf_counter() - rollback_started,
                },
            ) from error
        return {
            "paths": tuple(item.candidate.path for item in prepared),
            "operation": "create_text_files",
            "created_file_count": len(prepared),
            "modified_file_count": 0,
            "changes": tuple(
                {
                    "path": item.candidate.path,
                    "operation": "create",
                    "new_sha256": item.sha256,
                    "bytes_written": len(item.data),
                }
                for item in prepared
            ),
            "verified": True,
            "workspace_generation": generation,
            "mutation_group_id": group_id,
            "mutation_group_apply_result": "applied",
            "mutation_group_rollback_attempted": False,
            "mutation_group_rollback_result": "not_needed",
            "workspace_integrity_failure": False,
            "validation_seconds": validation_seconds,
            "staging_seconds": sum(item.staging_seconds for item in prepared),
            "application_seconds": time.perf_counter() - application_started,
            "rollback_seconds": 0.0,
        }

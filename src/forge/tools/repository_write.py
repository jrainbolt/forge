"""Confined, preconditioned, atomic repository text mutations."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from forge.tools.paths import (
    WorkspacePathError,
    resolve_workspace_write_path,
    workspace_relative_path,
)
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    ExecutionContext,
    StructuredValue,
    ToolCapability,
    ToolEvidence,
    ToolMetadata,
    ToolRisk,
)

MAX_WRITE_BYTES = 256 * 1024
MAX_MULTI_FILE_PATCHES = 4
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MutationPreview:
    path: str
    operation: str
    diff: str
    old_sha256: str | None
    new_sha256: str


@dataclass(frozen=True, slots=True)
class MultiFileMutationPreview:
    """One immutable, canonically ordered approval surface for a patch group."""

    files: tuple[MutationPreview, ...]
    workspace_generation: int
    proposal_identity: str

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def diff(self) -> str:
        return "\n".join(
            f"File {index}: {item.path}\n{item.diff}"
            for index, item in enumerate(self.files, 1)
        )


class WriteFileTool(Tool):
    _metadata = ToolMetadata(
        "repository.write_file",
        "Create or atomically replace one UTF-8 workspace file. Existing files "
        "require a prior read SHA-256; every invocation requires user approval.",
        ArgumentSchema(
            (
                ArgumentSpec("path", ArgumentType.STRING, "Workspace-relative file."),
                ArgumentSpec(
                    "content", ArgumentType.STRING, "Exact UTF-8 text to write."
                ),
                ArgumentSpec(
                    "mode", ArgumentType.STRING, "Explicit create or replace mode."
                ),
                ArgumentSpec(
                    "expected_sha256",
                    ArgumentType.STRING,
                    "Required observed byte hash for replace mode.",
                    False,
                ),
            )
        ),
        ToolRisk.WRITE,
        ToolEvidence.WRITE_SUCCESS,
        ToolCapability.WRITE,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        prepared = _prepare_write(arguments, context)
        if prepared.operation == "create":
            _exclusive_create(prepared.path, prepared.new_bytes)
        else:
            assert prepared.old_sha256 is not None
            _verify_current_hash(prepared.path, prepared.old_sha256)
            _atomic_replace(prepared.path, prepared.new_bytes, prepared.mode_bits)
        _verify_result(prepared.path, prepared.new_bytes, prepared.new_sha256)
        return _mutation_result(prepared, created=prepared.operation == "create")


class ApplyPatchTool(Tool):
    _metadata = ToolMetadata(
        "repository.apply_patch",
        "Apply exact unique text replacements to one previously read UTF-8 file. "
        "The observed SHA-256 and explicit user approval are mandatory.",
        ArgumentSchema(
            (
                ArgumentSpec("path", ArgumentType.STRING, "Workspace-relative file."),
                ArgumentSpec(
                    "expected_sha256",
                    ArgumentType.STRING,
                    "Observed SHA-256 of the current file bytes.",
                ),
                ArgumentSpec(
                    "edits",
                    ArgumentType.TEXT_EDITS,
                    "Ordered exact unique old/new text replacements.",
                ),
            )
        ),
        ToolRisk.WRITE,
        ToolEvidence.PATCH_SUCCESS,
        ToolCapability.WRITE,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        prepared = _prepare_patch(arguments, context)
        assert prepared.old_sha256 is not None
        _verify_current_hash(prepared.path, prepared.old_sha256)
        _atomic_replace(prepared.path, prepared.new_bytes, prepared.mode_bits)
        _verify_result(prepared.path, prepared.new_bytes, prepared.new_sha256)
        return _mutation_result(prepared, created=False)


ReplaceOperation = Callable[[Path, Path], None]


class MultiFilePatchTool(Tool):
    """Apply two to four prevalidated existing-file patches as one transaction."""

    _metadata = ToolMetadata(
        "repository.apply_multi_patch",
        "Atomically apply a bounded group of exact patches to previously read "
        "UTF-8 files with application-level rollback on handled failure.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "group_id", ArgumentType.STRING, "Canonical proposal identity."
                ),
                ArgumentSpec(
                    "workspace_generation",
                    ArgumentType.INTEGER,
                    "Workspace generation bound to the grouped proposal.",
                ),
                ArgumentSpec(
                    "patches",
                    ArgumentType.MULTI_FILE_PATCHES,
                    "Two to four canonically ordered existing-file patches.",
                ),
            )
        ),
        ToolRisk.WRITE,
        ToolEvidence.PATCH_SUCCESS,
        ToolCapability.WRITE,
    )

    def __init__(
        self,
        *,
        replace_operation: ReplaceOperation = os.replace,
        rollback_operation: ReplaceOperation | None = None,
    ) -> None:
        self._replace_operation = replace_operation
        self._rollback_operation = rollback_operation or os.replace

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        prepared, group_id, generation = _prepare_multi_patch(arguments, context)
        return _apply_multi_patch_transaction(
            prepared,
            group_id,
            generation,
            context,
            self._replace_operation,
            self._rollback_operation,
        )


def preview_repository_mutation(
    tool_name: str, arguments: Mapping[str, object], context: ExecutionContext
) -> MutationPreview:
    """Validate and render a deterministic mutation preview without writing."""
    if tool_name == "repository.write_file":
        prepared = _prepare_write(arguments, context)
    elif tool_name == "repository.apply_patch":
        prepared = _prepare_patch(arguments, context)
    else:
        raise ToolError("tool does not support mutation preview")
    old_text = prepared.old_text or ""
    new_text = prepared.new_bytes.decode("utf-8")
    fromfile = (
        "/dev/null" if prepared.operation == "create" else f"a/{prepared.display}"
    )
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=f"b/{prepared.display}",
        )
    )
    if not diff:
        diff = f"{prepared.operation} {prepared.display} (empty content)\n"
    return MutationPreview(
        prepared.display,
        prepared.operation,
        diff,
        prepared.old_sha256,
        prepared.new_sha256,
    )


def preview_multi_file_mutation(
    arguments: Mapping[str, object], context: ExecutionContext
) -> MultiFileMutationPreview:
    """Validate and render a complete deterministic group without writing."""
    prepared, group_id, generation = _prepare_multi_patch(arguments, context)
    previews = tuple(_preview_prepared(item) for item in prepared)
    return MultiFileMutationPreview(previews, generation, group_id)


@dataclass(frozen=True, slots=True)
class _PreparedMutation:
    path: Path
    display: str
    operation: str
    old_bytes: bytes | None
    old_text: str | None
    old_sha256: str | None
    new_bytes: bytes
    new_sha256: str
    mode_bits: int | None


def _preview_prepared(prepared: _PreparedMutation) -> MutationPreview:
    old_text = prepared.old_text or ""
    new_text = prepared.new_bytes.decode("utf-8")
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{prepared.display}",
            tofile=f"b/{prepared.display}",
        )
    )
    return MutationPreview(
        prepared.display,
        prepared.operation,
        diff,
        prepared.old_sha256,
        prepared.new_sha256,
    )


def _prepare_multi_patch(
    arguments: Mapping[str, object], context: ExecutionContext
) -> tuple[tuple[_PreparedMutation, ...], str, int]:
    group_id = _text(arguments, "group_id")
    if not SHA256_PATTERN.fullmatch(group_id):
        raise ToolError("group_id must be 64 lowercase hexadecimal characters")
    generation = arguments.get("workspace_generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ToolError("workspace_generation must be a non-negative integer")
    patches = arguments.get("patches")
    if not isinstance(patches, (list, tuple)) or not (
        2 <= len(patches) <= MAX_MULTI_FILE_PATCHES
    ):
        raise ToolError("multi-file patch requires between 2 and 4 files")
    paths = [patch.get("path") for patch in patches if isinstance(patch, Mapping)]
    if len(paths) != len(patches) or any(not isinstance(path, str) for path in paths):
        raise ToolError("each grouped patch requires a text path")
    if len(set(paths)) != len(paths):
        raise ToolError("multi-file patch paths must be unique")
    if paths != sorted(paths):
        raise ToolError("multi-file patch paths must use canonical ordering")
    identity_patches: list[dict[str, object]] = []
    prepared: list[_PreparedMutation] = []
    for patch in patches:
        assert isinstance(patch, Mapping)
        edits = patch.get("edits")
        assert isinstance(edits, (list, tuple))
        identity_patches.append(
            {
                "path": patch["path"],
                "expected_sha256": patch.get("expected_sha256"),
                "edits": [
                    {"old": edit.get("old"), "new": edit.get("new")}
                    for edit in edits
                    if isinstance(edit, Mapping)
                ],
            }
        )
        prepared.append(_prepare_patch(patch, context))
    expected_group_id = hashlib.sha256(
        json.dumps(
            {
                "workspace_generation": generation,
                "patches": identity_patches,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if group_id != expected_group_id:
        raise ToolError("group_id does not match the canonical grouped proposal")
    return tuple(prepared), group_id, generation


def _apply_multi_patch_transaction(
    prepared: tuple[_PreparedMutation, ...],
    group_id: str,
    generation: int,
    context: ExecutionContext,
    replace_operation: ReplaceOperation,
    rollback_operation: ReplaceOperation,
) -> StructuredValue:
    """Apply a group; handled replacement failures restore exact original bytes."""
    transaction_root = context.workspace / ".forge-exec" / "transactions"
    transaction = transaction_root / f"multi-{uuid.uuid4().hex}"
    replaced: list[_PreparedMutation] = []
    rollback_attempted = False
    try:
        transaction.mkdir(parents=True, mode=0o700)
        staged: dict[str, Path] = {}
        for index, item in enumerate(prepared):
            stage = transaction / f"{index:02d}.new"
            _write_staged(stage, item.new_bytes, item.mode_bits)
            staged[item.display] = stage
        # Complete-group compare-before-write. No replacement occurs before this loop.
        for item in prepared:
            assert item.old_sha256 is not None
            _revalidate_regular_target(item.path, context.workspace)
            _verify_current_hash(item.path, item.old_sha256)
        try:
            for item in prepared:
                replace_operation(staged[item.display], item.path)
                replaced.append(item)
                _verify_result(item.path, item.new_bytes, item.new_sha256)
        except Exception as apply_error:
            rollback_attempted = bool(replaced)
            rollback_error: Exception | None = None
            for index, item in enumerate(reversed(replaced)):
                try:
                    _revalidate_regular_target(item.path, context.workspace)
                    original = transaction / f"rollback-{index:02d}.old"
                    assert item.old_bytes is not None
                    _write_staged(original, item.old_bytes, item.mode_bits)
                    rollback_operation(original, item.path)
                    assert item.old_sha256 is not None
                    _verify_result(
                        item.path,
                        item.old_bytes,
                        item.old_sha256,
                    )
                except Exception as error:
                    rollback_error = error
                    break
            if rollback_error is not None:
                raise ToolError(
                    "fatal mutation integrity failure: grouped rollback failed; "
                    "workspace may be inconsistent",
                    output={
                        "mutation_group_id": group_id,
                        "mutation_file_count": len(prepared),
                        "mutation_group_apply_result": "failed",
                        "mutation_group_rollback_attempted": rollback_attempted,
                        "mutation_group_rollback_result": "failed",
                        "workspace_integrity_failure": True,
                    },
                ) from rollback_error
            raise ToolError(
                "grouped mutation application failed; original files restored",
                output={
                    "mutation_group_id": group_id,
                    "mutation_file_count": len(prepared),
                    "mutation_group_apply_result": "failed",
                    "mutation_group_rollback_attempted": rollback_attempted,
                    "mutation_group_rollback_result": (
                        "restored" if rollback_attempted else "not_needed"
                    ),
                    "workspace_integrity_failure": False,
                },
            ) from apply_error
        return {
            "paths": tuple(item.display for item in prepared),
            "changes": tuple(
                {
                    "path": item.display,
                    "old_sha256": item.old_sha256,
                    "new_sha256": item.new_sha256,
                    "bytes_written": len(item.new_bytes),
                }
                for item in prepared
            ),
            "operation": "multi_patch",
            "verified": True,
            "workspace_generation": generation,
            "mutation_group_id": group_id,
            "mutation_file_count": len(prepared),
            "mutation_group_validation_result": "passed",
            "mutation_group_preview_created": True,
            "mutation_group_apply_result": "applied",
            "mutation_group_rollback_attempted": False,
            "mutation_group_rollback_result": "not_needed",
            "workspace_integrity_failure": False,
        }
    finally:
        with suppress(OSError):
            shutil.rmtree(transaction)
        with suppress(OSError):
            transaction_root.rmdir()
        with suppress(OSError):
            transaction_root.parent.rmdir()


def _write_staged(path: Path, data: bytes, mode_bits: int | None) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if mode_bits is not None:
        os.chmod(path, mode_bits)


def _revalidate_regular_target(path: Path, workspace: Path) -> None:
    try:
        info = path.lstat()
        resolved = resolve_workspace_write_path(
            workspace, workspace_relative_path(workspace, path)
        )
    except (OSError, WorkspacePathError) as error:
        raise ToolError(
            "grouped target is no longer an authorized regular file"
        ) from error
    if resolved != path or not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ToolError("grouped target is no longer an authorized regular file")


def _prepare_write(
    arguments: Mapping[str, object], context: ExecutionContext
) -> _PreparedMutation:
    requested = _text(arguments, "path")
    content = _text(arguments, "content")
    mode = _text(arguments, "mode")
    if mode not in {"create", "replace"}:
        raise ToolError("mode must be create or replace")
    path = _resolve_write(context, requested)
    new_bytes = _bounded_utf8(content)
    if mode == "create":
        if "expected_sha256" in arguments:
            raise ToolError("create mode must not include expected_sha256")
        if path.exists():
            raise ToolError("create target already exists")
        return _prepared(path, context, mode, None, None, None, new_bytes, None)
    expected = _hash_argument(arguments)
    old_bytes, old_text, mode_bits = _existing_text_file(path, requested)
    old_hash = _sha256(old_bytes)
    if old_hash != expected:
        raise ToolError("precondition failed: current SHA-256 does not match")
    return _prepared(
        path, context, mode, old_bytes, old_text, old_hash, new_bytes, mode_bits
    )


def _prepare_patch(
    arguments: Mapping[str, object], context: ExecutionContext
) -> _PreparedMutation:
    requested = _text(arguments, "path")
    expected = _hash_argument(arguments)
    edits = arguments.get("edits")
    if not isinstance(edits, (list, tuple)) or not edits:
        raise ToolError("patch edits must be a non-empty sequence")
    path = _resolve_write(context, requested)
    old_bytes, old_text, mode_bits = _existing_text_file(path, requested)
    old_hash = _sha256(old_bytes)
    if old_hash != expected:
        raise ToolError("precondition failed: current SHA-256 does not match")
    updated = old_text
    for edit in edits:
        if not isinstance(edit, Mapping) or set(edit) != {"old", "new"}:
            raise ToolError("each patch edit must contain exactly old and new text")
        old = edit["old"]
        new = edit["new"]
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolError("patch old and new values must be text")
        if not old:
            raise ToolError("patch old text must not be empty")
        if old == new:
            raise ToolError("patch edit must change text")
        occurrences = updated.count(old)
        if occurrences == 0:
            raise ToolError("patch conflict: old text was not found exactly")
        if occurrences > 1:
            raise ToolError("patch conflict: old text is ambiguous")
        updated = updated.replace(old, new, 1)
    new_bytes = _bounded_utf8(updated)
    if new_bytes == old_bytes:
        raise ToolError("patch must change file content")
    return _prepared(
        path,
        context,
        "patch",
        old_bytes,
        old_text,
        old_hash,
        new_bytes,
        mode_bits,
    )


def _prepared(
    path: Path,
    context: ExecutionContext,
    operation: str,
    old_bytes: bytes | None,
    old_text: str | None,
    old_sha256: str | None,
    new_bytes: bytes,
    mode_bits: int | None,
) -> _PreparedMutation:
    return _PreparedMutation(
        path,
        workspace_relative_path(context.workspace, path),
        operation,
        old_bytes,
        old_text,
        old_sha256,
        new_bytes,
        _sha256(new_bytes),
        mode_bits,
    )


def _resolve_write(context: ExecutionContext, requested: str) -> Path:
    try:
        return resolve_workspace_write_path(context.workspace, requested)
    except WorkspacePathError as error:
        raise ToolError(str(error)) from error


def _existing_text_file(path: Path, requested: str) -> tuple[bytes, str, int]:
    if not path.exists():
        raise ToolError("replace or patch target does not exist")
    try:
        info = path.stat()
    except OSError as error:
        raise ToolError(f"cannot inspect file: {requested}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ToolError("write target must be a regular file")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ToolError(f"cannot read file: {requested}") from error
    if len(data) > MAX_WRITE_BYTES:
        raise ToolError(f"file exceeds the {MAX_WRITE_BYTES}-byte write limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ToolError("write target is not valid UTF-8 text") from error
    return data, text, stat.S_IMODE(info.st_mode)


def _bounded_utf8(content: str) -> bytes:
    data = content.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ToolError(f"content exceeds the {MAX_WRITE_BYTES}-byte write limit")
    return data


def _hash_argument(arguments: Mapping[str, object]) -> str:
    value = arguments.get("expected_sha256")
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ToolError("expected_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _text(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolError(f"{name} must be text")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_current_hash(path: Path, expected: str) -> None:
    try:
        current = path.read_bytes()
    except OSError as error:
        raise ToolError("cannot verify current file precondition") from error
    if _sha256(current) != expected:
        raise ToolError("precondition failed: file changed before mutation")


def _exclusive_create(path: Path, data: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ToolError("create target appeared before mutation") from error
    except OSError as error:
        if created:
            with suppress(OSError):
                path.unlink()
        raise ToolError("cannot create file atomically") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_replace(path: Path, data: bytes, mode_bits: int | None) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".forge-write-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode_bits is not None:
            os.chmod(temporary, mode_bits)
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise ToolError("cannot atomically replace file") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _verify_result(path: Path, intended: bytes, expected_hash: str) -> None:
    try:
        actual = path.read_bytes()
    except OSError as error:
        raise ToolError("cannot verify written file") from error
    if actual != intended or _sha256(actual) != expected_hash:
        raise ToolError("post-write verification failed")


def _mutation_result(prepared: _PreparedMutation, *, created: bool) -> StructuredValue:
    return {
        "path": prepared.display,
        "operation": prepared.operation,
        "created": created,
        "old_sha256": prepared.old_sha256,
        "new_sha256": prepared.new_sha256,
        "bytes_written": len(prepared.new_bytes),
        "verified": True,
    }

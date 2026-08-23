"""Confined, preconditioned, atomic repository text mutations."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping
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
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MutationPreview:
    path: str
    operation: str
    diff: str
    old_sha256: str | None
    new_sha256: str


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


@dataclass(frozen=True, slots=True)
class _PreparedMutation:
    path: Path
    display: str
    operation: str
    old_text: str | None
    old_sha256: str | None
    new_bytes: bytes
    new_sha256: str
    mode_bits: int | None


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
        return _prepared(path, context, mode, None, None, new_bytes, None)
    expected = _hash_argument(arguments)
    old_bytes, old_text, mode_bits = _existing_text_file(path, requested)
    old_hash = _sha256(old_bytes)
    if old_hash != expected:
        raise ToolError("precondition failed: current SHA-256 does not match")
    return _prepared(path, context, mode, old_text, old_hash, new_bytes, mode_bits)


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
    return _prepared(path, context, "patch", old_text, old_hash, new_bytes, mode_bits)


def _prepared(
    path: Path,
    context: ExecutionContext,
    operation: str,
    old_text: str | None,
    old_sha256: str | None,
    new_bytes: bytes,
    mode_bits: int | None,
) -> _PreparedMutation:
    return _PreparedMutation(
        path,
        workspace_relative_path(context.workspace, path),
        operation,
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

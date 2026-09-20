"""A54 candidate-bound edit/create transactions; no general filesystem API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from forge.tools.controlled_creation import (
    CREATE_MODE,
    MAX_CREATE_BYTES,
    _exclusive_create,
    _target,
)
from forge.tools.repository_write import (
    MultiFileMutationPreview,
    MutationPreview,
    _prepare_patch,
    _PreparedMutation,
    _preview_prepared,
    _revalidate_regular_target,
    _verify_current_hash,
    _verify_result,
    _write_staged,
)
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


@dataclass(frozen=True, slots=True)
class ExistingFileEdit:
    prepared: _PreparedMutation


@dataclass(frozen=True, slots=True)
class NewTextFileCreate:
    candidate: CreateCandidate
    target: Path
    data: bytes
    sha256: str


type FileOperation = ExistingFileEdit | NewTextFileCreate
type ReplaceOperation = Callable[[Path, Path], None]
type CreateOperation = Callable[[Path, bytes], tuple[int, int]]
type RemoveOperation = Callable[[Path], None]


def mixed_group_id(
    operations: tuple[Mapping[str, object], ...],
    candidates: tuple[CreateCandidate, ...],
    generation: int,
    workspace: Path,
) -> str:
    """Bind exact child arguments and trusted CREATE authority to one invocation."""
    authority = {item.path: item for item in candidates}
    children = []
    for item in sorted(operations, key=lambda child: str(child["path"])):
        path = item["path"]
        if item["type"] == "create":
            if path not in authority:
                raise ToolError("create path lacks candidate authority")
            candidate = authority[path]
            content = item["content"]
            if not isinstance(content, str):
                raise ToolError("created content must be text")
            children.append(
                {
                    "type": "create",
                    "path": path,
                    "new_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "parent_device": candidate.parent_device,
                    "parent_inode": candidate.parent_inode,
                    "mode": CREATE_MODE,
                    "provenance": candidate.provenance,
                }
            )
        else:
            edits = item.get("edits")
            if not isinstance(edits, (tuple, list)):
                raise ToolError("edit child requires exact replacements")
            children.append(
                {
                    "type": "edit",
                    "path": path,
                    "expected_sha256": item.get("expected_sha256"),
                    "edits": [dict(edit) for edit in edits],
                }
            )
    payload = {
        "workspace": str(workspace),
        "workspace_generation": generation,
        "operations": children,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepare(
    arguments: Mapping[str, object], context: ExecutionContext
) -> tuple[tuple[FileOperation, ...], str, int, float, float]:
    normalization_started = time.perf_counter()
    operations = arguments.get("operations")
    generation = arguments.get("workspace_generation")
    group_id = arguments.get("group_id")
    if not isinstance(operations, (tuple, list)) or not 2 <= len(operations) <= 4:
        raise ToolError("mixed transaction requires two to four operations")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ToolError("invalid workspace generation")
    if not isinstance(group_id, str) or len(group_id) != 64:
        raise ToolError("invalid mixed group identity")
    if any(not isinstance(item, Mapping) for item in operations):
        raise ToolError("invalid file operation")
    paths = [item.get("path") for item in operations]
    if any(not isinstance(path, str) for path in paths) or paths != sorted(set(paths)):
        raise ToolError("mixed operation paths must be unique and canonical")
    creates = [item for item in operations if item.get("type") == "create"]
    edits = [item for item in operations if item.get("type") == "edit"]
    if (
        not creates
        or not edits
        or len(creates) > 2
        or len(creates) + len(edits) != len(operations)
    ):
        raise ToolError("mixed transaction requires edit and create children")
    create_authority = {item.path: item for item in context.create_candidates}
    if {item["path"] for item in creates} != set(create_authority):
        raise ToolError("mixed creates must match exact authorized path set")
    if {item["path"] for item in edits} != set(context.edit_candidates):
        raise ToolError("mixed edits must match exact authorized path set")
    normalization_seconds = time.perf_counter() - normalization_started
    materialization_started = time.perf_counter()
    prepared: list[FileOperation] = []
    for item in operations:
        path = item["path"]
        if item["type"] == "edit":
            if set(item) != {"type", "path", "expected_sha256", "edits"}:
                raise ToolError("invalid edit child")
            prepared.append(ExistingFileEdit(_prepare_patch(item, context)))
            continue
        if set(item) != {"type", "path", "content"}:
            raise ToolError("invalid create child")
        content = item["content"]
        if not isinstance(content, str):
            raise ToolError("created content must be text")
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ToolError("created content is not UTF-8") from error
        if (
            len(data) > MAX_CREATE_BYTES
            or any(byte < 32 and byte not in {9, 10, 13} for byte in data)
            or b"\x7f" in data
        ):
            raise ToolError("created content violates text/size policy")
        candidate = create_authority[path]
        if candidate.generation != generation:
            raise ToolError("create candidate generation is stale")
        _, target, parent = _target(context.workspace, path)
        if (parent.st_dev, parent.st_ino) != (
            candidate.parent_device,
            candidate.parent_inode,
        ):
            raise ToolError("create parent identity changed")
        prepared.append(
            NewTextFileCreate(candidate, target, data, hashlib.sha256(data).hexdigest())
        )
    if group_id != mixed_group_id(
        tuple(operations), context.create_candidates, generation, context.workspace
    ):
        raise ToolError("mixed group identity mismatch")
    return (
        tuple(prepared),
        group_id,
        generation,
        normalization_seconds,
        time.perf_counter() - materialization_started,
    )


def preview_mixed_file_transaction(
    arguments: Mapping[str, object], context: ExecutionContext
) -> MultiFileMutationPreview:
    import difflib

    prepared, group_id, generation, _, _ = _prepare(arguments, context)
    previews = []
    for operation in prepared:
        if isinstance(operation, ExistingFileEdit):
            item = _preview_prepared(operation.prepared)
            previews.append(
                MutationPreview(
                    item.path,
                    "modify",
                    f"MODIFY {item.path}\n{item.diff}",
                    item.old_sha256,
                    item.new_sha256,
                )
            )
        else:
            path = operation.candidate.path
            diff = "".join(
                difflib.unified_diff(
                    [],
                    operation.data.decode("utf-8").splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{path}",
                )
            )
            previews.append(
                MutationPreview(
                    path,
                    "create",
                    f"CREATE {path} | {len(operation.data)} bytes | "
                    f"SHA-256 {operation.sha256} | mode 0644\n"
                    + (diff or "(empty file)\n"),
                    None,
                    operation.sha256,
                )
            )
    return MultiFileMutationPreview(tuple(previews), generation, group_id)


class MixedFileTransactionTool(Tool):
    """Apply only fully authorized typed MODIFY and CREATE operations."""

    _metadata = ToolMetadata(
        "repository.apply_file_operations",
        "Apply one bounded, approval-gated edit and create transaction.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "group_id", ArgumentType.STRING, "Canonical group identity."
                ),
                ArgumentSpec(
                    "workspace_generation", ArgumentType.INTEGER, "Bound generation."
                ),
                ArgumentSpec(
                    "operations",
                    ArgumentType.FILE_OPERATIONS,
                    "Canonical typed operations.",
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
        rollback_operation: ReplaceOperation = os.replace,
        create_operation: CreateOperation = _exclusive_create,
        remove_operation: RemoveOperation = Path.unlink,
    ) -> None:
        self._replace = replace_operation
        self._rollback = rollback_operation
        self._create = create_operation
        self._remove = remove_operation

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        (
            prepared,
            group_id,
            generation,
            normalization_seconds,
            materialization_seconds,
        ) = _prepare(arguments, context)
        transaction_root = context.workspace / ".forge-exec" / "transactions"
        transaction = transaction_root / f"mixed-{uuid.uuid4().hex}"
        journal: list[tuple[FileOperation, tuple[int, int] | None]] = []
        try:
            staging_started = time.perf_counter()
            transaction.mkdir(parents=True, mode=0o700)
            staged: dict[str, Path] = {}
            for index, operation in enumerate(prepared):
                if isinstance(operation, ExistingFileEdit):
                    stage = transaction / f"{index:02d}.new"
                    _write_staged(
                        stage,
                        operation.prepared.new_bytes,
                        operation.prepared.mode_bits,
                    )
                    staged[operation.prepared.display] = stage
            staging_seconds = time.perf_counter() - staging_started
            validation_started = time.perf_counter()
            for operation in prepared:
                if isinstance(operation, ExistingFileEdit):
                    item = operation.prepared
                    assert item.old_sha256 is not None
                    _revalidate_regular_target(item.path, context.workspace)
                    _verify_current_hash(item.path, item.old_sha256)
                else:
                    _, _, parent = _target(context.workspace, operation.candidate.path)
                    if (parent.st_dev, parent.st_ino) != (
                        operation.candidate.parent_device,
                        operation.candidate.parent_inode,
                    ):
                        raise ToolError("create parent identity changed before apply")
            validation_seconds = time.perf_counter() - validation_started
            application_started = time.perf_counter()
            in_flight_create: NewTextFileCreate | None = None
            try:
                for operation in prepared:
                    if isinstance(operation, ExistingFileEdit):
                        item = operation.prepared
                        self._replace(staged[item.display], item.path)
                        journal.append((operation, None))
                        _verify_result(item.path, item.new_bytes, item.new_sha256)
                    else:
                        in_flight_create = operation
                        identity = self._create(operation.target, operation.data)
                        journal.append((operation, identity))
                        in_flight_create = None
                        info = operation.target.lstat()
                        if (
                            not stat.S_ISREG(info.st_mode)
                            or stat.S_IMODE(info.st_mode) != CREATE_MODE
                            or (info.st_dev, info.st_ino) != identity
                            or operation.target.read_bytes() != operation.data
                        ):
                            raise ToolError("created file verification failed")
            except Exception as error:
                unowned_target_present = (
                    in_flight_create is not None
                    and os.path.lexists(in_flight_create.target)
                )
                rollback_started = time.perf_counter()
                try:
                    for index, (operation, identity) in enumerate(reversed(journal)):
                        if isinstance(operation, NewTextFileCreate):
                            info = operation.target.lstat()
                            if (
                                not stat.S_ISREG(info.st_mode)
                                or (info.st_dev, info.st_ino) != identity
                                or hashlib.sha256(
                                    operation.target.read_bytes()
                                ).hexdigest()
                                != operation.sha256
                            ):
                                raise ToolError(
                                    "transaction-created file changed before rollback"
                                )
                            self._remove(operation.target)
                        else:
                            item = operation.prepared
                            _revalidate_regular_target(item.path, context.workspace)
                            original = transaction / f"rollback-{index:02d}.old"
                            assert (
                                item.old_bytes is not None
                                and item.old_sha256 is not None
                            )
                            _write_staged(original, item.old_bytes, item.mode_bits)
                            self._rollback(original, item.path)
                            _verify_result(item.path, item.old_bytes, item.old_sha256)
                except Exception as rollback_error:
                    raise ToolError(
                        "fatal mixed transaction integrity failure: rollback failed",
                        output={
                            "workspace_integrity_failure": True,
                            "mutation_group_apply_result": "failed",
                            "mutation_group_rollback_attempted": bool(journal),
                            "mutation_group_rollback_result": "failed",
                            "mutation_group_id": group_id,
                            "rollback_seconds": time.perf_counter() - rollback_started,
                        },
                    ) from rollback_error
                if unowned_target_present:
                    raise ToolError(
                        "fatal mixed transaction integrity failure: "
                        "unowned create target remains",
                        output={
                            "workspace_integrity_failure": True,
                            "mutation_group_apply_result": "failed",
                            "mutation_group_rollback_attempted": bool(journal),
                            "mutation_group_rollback_result": "failed",
                            "mutation_group_id": group_id,
                            "rollback_seconds": time.perf_counter() - rollback_started,
                        },
                    ) from error
                raise ToolError(
                    "mixed transaction failed; applied operations restored",
                    output={
                        "workspace_integrity_failure": False,
                        "mutation_group_apply_result": "failed",
                        "mutation_group_rollback_attempted": bool(journal),
                        "mutation_group_rollback_result": "restored"
                        if journal
                        else "not_needed",
                        "mutation_group_id": group_id,
                        "rollback_seconds": time.perf_counter() - rollback_started,
                    },
                ) from error
            return {
                "paths": tuple(
                    operation.prepared.display
                    if isinstance(operation, ExistingFileEdit)
                    else operation.candidate.path
                    for operation in prepared
                ),
                "changes": tuple(
                    {
                        "path": operation.prepared.display
                        if isinstance(operation, ExistingFileEdit)
                        else operation.candidate.path,
                        "operation": "modify"
                        if isinstance(operation, ExistingFileEdit)
                        else "create",
                        "old_sha256": operation.prepared.old_sha256
                        if isinstance(operation, ExistingFileEdit)
                        else None,
                        "new_sha256": operation.prepared.new_sha256
                        if isinstance(operation, ExistingFileEdit)
                        else operation.sha256,
                        "result": "applied",
                    }
                    for operation in prepared
                ),
                "operation": "file_operations",
                "mutation_file_count": len(prepared),
                "modified_file_count": sum(
                    isinstance(item, ExistingFileEdit) for item in prepared
                ),
                "created_file_count": sum(
                    isinstance(item, NewTextFileCreate) for item in prepared
                ),
                "verified": True,
                "workspace_generation": generation,
                "mutation_group_id": group_id,
                "mutation_group_apply_result": "applied",
                "mutation_group_rollback_attempted": False,
                "mutation_group_rollback_result": "not_needed",
                "workspace_integrity_failure": False,
                "validation_seconds": validation_seconds,
                "normalization_seconds": normalization_seconds,
                "materialization_seconds": materialization_seconds,
                "staging_seconds": staging_seconds,
                "application_seconds": time.perf_counter() - application_started,
            }
        finally:
            with suppress(OSError):
                shutil.rmtree(transaction)
            with suppress(OSError):
                transaction_root.rmdir()
            with suppress(OSError):
                transaction_root.parent.rmdir()

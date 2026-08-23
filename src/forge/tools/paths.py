"""Central resolved-path confinement for repository capabilities."""

from __future__ import annotations

from pathlib import Path

PROTECTED_WRITE_PARTS = frozenset({".git"})


class WorkspacePathError(ValueError):
    """A requested path cannot be safely resolved inside its workspace."""


def resolve_workspace_path(workspace: Path, requested: str) -> Path:
    """Resolve an existing relative path and prove it remains in the workspace."""
    if not isinstance(workspace, Path) or not workspace.is_absolute():
        raise TypeError("workspace must be an absolute Path")
    if not isinstance(requested, str) or not requested:
        raise WorkspacePathError("requested path must be non-empty text")
    relative = Path(requested)
    if relative.is_absolute():
        raise WorkspacePathError("absolute requested paths are not allowed")
    try:
        resolved = (workspace / relative).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspacePathError(
            f"requested path does not exist: {requested}"
        ) from error
    if resolved != workspace and workspace not in resolved.parents:
        raise WorkspacePathError("requested path resolves outside the workspace")
    return resolved


def workspace_relative_path(workspace: Path, resolved: Path) -> str:
    """Render an already-confined path without exposing the host workspace path."""
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as error:
        raise WorkspacePathError("resolved path is outside the workspace") from error
    rendered = relative.as_posix()
    return rendered if rendered else "."


def resolve_workspace_write_path(workspace: Path, requested: str) -> Path:
    """Resolve an existing or creatable file target inside the workspace."""
    if not isinstance(workspace, Path) or not workspace.is_absolute():
        raise TypeError("workspace must be an absolute Path")
    if not isinstance(requested, str) or not requested:
        raise WorkspacePathError("requested path must be non-empty text")
    relative = Path(requested)
    if relative.is_absolute():
        raise WorkspacePathError("absolute requested paths are not allowed")
    if any(part in PROTECTED_WRITE_PARTS for part in relative.parts):
        raise WorkspacePathError("writes to protected repository metadata are denied")
    lexical_target = workspace / relative
    if lexical_target.is_symlink():
        raise WorkspacePathError("write target must not be a symlink")
    try:
        parent = lexical_target.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspacePathError(
            f"write target parent does not exist: {requested}"
        ) from error
    if parent != workspace and workspace not in parent.parents:
        raise WorkspacePathError("write target parent resolves outside the workspace")
    try:
        resolved_relative = parent.relative_to(workspace)
    except ValueError as error:
        raise WorkspacePathError(
            "write target resolves outside the workspace"
        ) from error
    if any(part in PROTECTED_WRITE_PARTS for part in resolved_relative.parts):
        raise WorkspacePathError("writes to protected repository metadata are denied")
    target = parent / lexical_target.name
    if target.exists():
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                f"cannot resolve write target: {requested}"
            ) from error
        if resolved != workspace and workspace not in resolved.parents:
            raise WorkspacePathError("write target resolves outside the workspace")
        return resolved
    return target

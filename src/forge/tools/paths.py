"""Central resolved-path confinement for repository capabilities."""

from __future__ import annotations

from pathlib import Path


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

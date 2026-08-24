from __future__ import annotations

from pathlib import Path

import pytest

from forge.embeddings import MockEmbeddingModel
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.semantic_index import SemanticIndex
from forge.tools import (
    ExecutionContext,
    PermissionDecision,
    ToolCapability,
    ToolError,
    ToolEvidence,
    ToolRisk,
    create_readonly_repository_registry,
    create_repository_registry,
)


def test_semantic_tool_is_conditional_read_discovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    index = SemanticIndex(
        workspace, MockEmbeddingModel(), cache_root=tmp_path / "cache"
    )
    without = create_readonly_repository_registry()
    with_semantic = create_readonly_repository_registry(semantic_index=index)
    assert "repository.semantic_search" not in {
        metadata.name for metadata in without.metadata
    }
    tool = with_semantic.get("repository.semantic_search")
    assert tool.metadata.risk is ToolRisk.READ_ONLY
    assert tool.metadata.capability is ToolCapability.READ
    assert tool.metadata.evidence is ToolEvidence.DISCOVERY


def test_semantic_tool_returns_structured_candidates_not_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "policy.py").write_text(
        "def require_approval():\n    return 'write permission'\n"
    )
    index = SemanticIndex(
        workspace, MockEmbeddingModel(), cache_root=tmp_path / "cache"
    )
    registry = create_readonly_repository_registry(semantic_index=index)
    tool = registry.get("repository.semantic_search")
    result = tool.execute(
        {"query": "approval for repository write"},
        ExecutionContext(workspace.resolve()),
    )
    assert result["evidence"] == "discovery_only"
    assert result["requires_source_read"] is True
    assert result["matches"][0]["path"] == "policy.py"
    assert "source" not in result["matches"][0]
    assert "recommended_range" in result["matches"][0]


def test_semantic_limits_scope_and_permission_exposure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "inside").mkdir(parents=True)
    (workspace / "outside.py").write_text("def repairs(): pass\n")
    (workspace / "inside" / "context.py").write_text("def budgets(): pass\n")
    index = SemanticIndex(
        workspace, MockEmbeddingModel(), cache_root=tmp_path / "cache"
    )
    policy = resolve_interaction_policy(AutonomyMode.READ, "safe")
    registry = create_repository_registry(policy, semantic_index=index)
    assert (
        registry.get("repository.semantic_search").metadata.name
        == "repository.semantic_search"
    )
    tool = registry.get("repository.semantic_search")
    result = tool.execute(
        {"query": "context budgets", "path": "inside", "limit": 1},
        ExecutionContext(workspace.resolve()),
    )
    assert result["matches"][0]["path"] == "inside/context.py"
    with pytest.raises(ToolError):
        tool.execute(
            {"query": "x", "path": "../"}, ExecutionContext(workspace.resolve())
        )
    assert policy.decision_for(ToolCapability.READ) is PermissionDecision.ALLOW

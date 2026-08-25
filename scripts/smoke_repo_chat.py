"""Run one opt-in read-only repository-aware request through a model profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.embedding_config import load_embedding_profile
from forge.models import GenerationConfig, default_backend_registry, load_model_catalog
from forge.orchestration import RepositoryChatSession, ToolActivity
from forge.repository_index import RepositoryIndex
from forge.semantic_index import SemanticIndex
from forge.tools import create_readonly_repository_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-config", type=Path)
    parser.add_argument("--embedding-profile")
    args = parser.parse_args()

    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.profile)
    if bool(args.embedding_config) != bool(args.embedding_profile):
        parser.error("--embedding-config and --embedding-profile are required together")
    repository_index = RepositoryIndex(args.workspace)
    semantic_index = (
        SemanticIndex(
            args.workspace,
            load_embedding_profile(args.embedding_config, args.embedding_profile),
        )
        if args.embedding_config is not None
        else None
    )

    def show_activity(activity: ToolActivity) -> None:
        path = f": {activity.path}" if activity.path is not None else ""
        print(f"[tool] {activity.tool_name}{path} ({activity.status})", flush=True)

    with (
        model,
        RepositoryChatSession(
            args.profile,
            model,
            args.workspace,
            generation=GenerationConfig(
                max_tokens=args.max_tokens, temperature=0.0, seed=args.seed
            ),
            registry=create_readonly_repository_registry(
                repository_index, semantic_index
            ),
            repository_index=repository_index,
            semantic_index=semantic_index,
            activity_callback=show_activity,
        ) as session,
    ):
        response = session.ask(args.question)
        if response.protocol_corrections:
            print(f"[protocol] corrections: {response.protocol_corrections}")
        for goal in response.evidence_goals:
            print(
                f"[goal] {goal.goal_id} {goal.status.value}: "
                f"{goal.description} paths={goal.source_paths}"
            )
        print(
            f"[coverage] complete={response.coverage_complete} "
            f"transitions={response.goal_transitions} "
            f"premature_finals={response.premature_finals} "
            f"wrong_goal_reads={response.wrong_goal_reads}"
        )
        bootstrap = response.bootstrap_metrics
        print(
            f"[bootstrap] executions={bootstrap.executions} "
            f"successes={bootstrap.successes} candidates={bootstrap.candidates} "
            f"empty={bootstrap.empty_results} failures={bootstrap.failures} "
            f"model_discovery_after={bootstrap.model_discovery_calls_after_bootstrap}"
        )
        print(response.text)
    if semantic_index is not None:
        semantic_index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

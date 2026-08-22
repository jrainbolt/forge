"""Run one opt-in read-only repository-aware request through a model profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.models import GenerationConfig, default_backend_registry, load_model_catalog
from forge.orchestration import RepositoryChatSession, ToolActivity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.profile)

    def show_activity(activity: ToolActivity) -> None:
        path = f": {activity.path}" if activity.path is not None else ""
        print(f"[tool] {activity.tool_name}{path} ({activity.status})", flush=True)

    with (
        model,
        RepositoryChatSession(
            args.profile,
            model,
            args.workspace,
            generation=GenerationConfig(max_tokens=args.max_tokens, temperature=0.0),
            activity_callback=show_activity,
        ) as session,
    ):
        response = session.ask(args.question)
        if response.protocol_corrections:
            print(f"[protocol] corrections: {response.protocol_corrections}")
        print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

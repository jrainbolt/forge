"""Run paired-semantic-planning-v1 with one configured local model profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.evaluation import (
    build_paired_planning_run,
    render_paired_planning,
    run_paired_semantic_planning,
    write_paired_planning_json,
)
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError(
            "paired-semantic-planning-v1 requires context capacity 8192, got "
            f"{model.context_capacity}"
        )
    try:
        results = run_paired_semantic_planning(args.model, model)
    finally:
        model.close()
    run = build_paired_planning_run(results, context_capacity=8192)
    write_paired_planning_json(run, args.output)
    print(render_paired_planning(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

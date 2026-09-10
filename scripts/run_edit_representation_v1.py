"""Run edit-representation-v1 with one configured local model profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    build_edit_representation_run,
    foundation_realworld_tasks,
    hash_workspace,
    render_edit_representation,
    run_edit_representation_diagnostics,
    write_edit_representation_json,
)
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"{args.model}: P01/P02/P03/E04 R1-R3; seed 42")
    task = next(task for task in foundation_realworld_tasks() if task.task_id == "E04")
    task = replace(task, seeds=(42,))
    canonical_before = hash_workspace(args.repository)
    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.model)
    context_capacity = model.context_capacity
    if context_capacity != 8192:
        model.close()
        raise RuntimeError(
            f"edit-representation-v1 requires context capacity 8192, got "
            f"{context_capacity}"
        )
    try:
        with model:
            results = run_edit_representation_diagnostics(
                args.model,
                model,
                foundation_repository=args.repository,
                foundation_task=task,
            )
    finally:
        model.close()
    if canonical_before != hash_workspace(args.repository):
        raise RuntimeError("canonical Foundation repository changed")
    run = build_edit_representation_run(results, context_capacity=context_capacity)
    write_edit_representation_json(run, args.output)
    print(render_edit_representation(run))
    print("Canonical unchanged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

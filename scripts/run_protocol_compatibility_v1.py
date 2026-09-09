"""Run protocol-compatibility-v1 with one configured local model profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    RealWorldEvaluationRunner,
    build_protocol_compatibility_run,
    foundation_realworld_tasks,
    hash_workspace,
    inspect_repository,
    render_protocol_compatibility,
    run_protocol_diagnostics,
    write_protocol_compatibility_json,
)
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"{args.model}: P01/P02/P03 L1-L3; Foundation E04 L1-L4; seed 42")
    configure = ("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON")
    build = ("cmake", "--build", "build")
    test = ("ctest", "--test-dir", "build", "--output-on-failure")
    snapshot = inspect_repository(
        args.repository,
        name=args.repository.name,
        language="C17",
        source_suffixes=frozenset({".c", ".h"}),
        setup_commands=(configure,),
        build_command=build,
        test_command=test,
    )
    if snapshot.baseline_outcome.value != "PASS":
        raise RuntimeError("benchmark baseline build/tests failed")
    task = next(task for task in foundation_realworld_tasks() if task.task_id == "E04")
    task = replace(task, seeds=(42,))
    canonical_before = hash_workspace(args.repository)
    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.model)
    try:
        with model:
            l4 = RealWorldEvaluationRunner(args.model, model, args.repository).run(
                (task,), snapshot
            )
            results = run_protocol_diagnostics(
                args.model,
                model,
                foundation_repository=args.repository,
                foundation_task=task,
                l4_result=l4.results[0],
            )
    finally:
        model.close()
    if canonical_before != hash_workspace(args.repository):
        raise RuntimeError("canonical Foundation repository changed")
    run = build_protocol_compatibility_run(results)
    write_protocol_compatibility_json(run, args.output)
    print(render_protocol_compatibility(run))
    print("Canonical unchanged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

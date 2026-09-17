"""Run A43 grouped-protocol diagnostics with one configured local model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    RealWorldEvaluationRunner,
    RecordingModel,
    build_grouped_protocol_run,
    foundation_realworld_tasks,
    full_forge_grouped_result,
    hash_workspace,
    inspect_repository,
    render_grouped_protocol,
    run_grouped_diagnostics,
    write_grouped_protocol_json,
)
from forge.models import (
    MutationRepresentationPolicy,
    default_backend_registry,
    load_model_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--skip-g5", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_only and args.repository is None:
        parser.error("--repository is required unless --synthetic-only is used")

    repository = args.repository.resolve(strict=True) if args.repository else None
    before = hash_workspace(repository) if repository is not None else None
    task = next(task for task in foundation_realworld_tasks() if task.task_id == "E08")
    task = replace(task, seeds=(42,))
    catalog = load_model_catalog(args.config, default_backend_registry())
    underlying = catalog.create(args.model)
    if underlying.context_capacity != 8192:
        underlying.close()
        raise RuntimeError(
            "grouped-protocol-compatibility-v1 requires context capacity 8192, "
            f"got {underlying.context_capacity}"
        )
    model = RecordingModel(underlying)
    try:
        results = list(
            run_grouped_diagnostics(
                args.model,
                model,
                repository=repository,
                foundation_task=task if repository is not None else None,
            )
        )
        if repository is not None and not args.skip_g5:
            configure = ("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON")
            build = ("cmake", "--build", "build")
            test = ("ctest", "--test-dir", "build", "--output-on-failure")
            snapshot = inspect_repository(
                repository,
                name=repository.name,
                language="C17",
                source_suffixes=frozenset({".c", ".h"}),
                setup_commands=(configure,),
                build_command=build,
                test_command=test,
            )
            if snapshot.baseline_outcome.value != "PASS":
                raise RuntimeError("benchmark baseline build/tests failed")
            exchange_start = len(model.exchanges)
            realworld = RealWorldEvaluationRunner(
                args.model,
                model,
                repository,
                mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
            ).run((task,), snapshot)
            results.append(
                full_forge_grouped_result(
                    args.model,
                    realworld.results[0],
                    tuple(model.exchanges[exchange_start:]),
                )
            )
    finally:
        model.close()
    unchanged = repository is None or before == hash_workspace(repository)
    if not unchanged:
        raise RuntimeError("canonical Foundation repository changed")
    run = build_grouped_protocol_run(
        tuple(results), context_capacity=8192, canonical_unchanged=unchanged
    )
    write_grouped_protocol_json(run, args.output)
    print(render_grouped_protocol(run))
    print(f"Canonical unchanged: {unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

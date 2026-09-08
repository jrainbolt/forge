"""Run the bounded local model-capability-v1 Foundation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.evaluation import (
    ModelCapabilityRunner,
    foundation_realworld_tasks,
    inspect_repository,
    render_model_capability_report,
    write_model_capability_json,
)
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", default="qwen-small,qwen-large")
    parser.add_argument("--tasks", default="E01,E03,E04,E07")
    args = parser.parse_args()

    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    task_ids = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    catalog = load_model_catalog(args.config, default_backend_registry())
    unknown_profiles = set(profiles) - set(catalog.profile_names)
    if unknown_profiles:
        parser.error(f"unconfigured profiles: {', '.join(sorted(unknown_profiles))}")
    definitions = foundation_realworld_tasks()
    tasks = tuple(task for task in definitions if task.task_id in task_ids)
    if tuple(task.task_id for task in tasks) != task_ids:
        parser.error(
            "tasks must use existing Foundation definitions in definition order"
        )
    seeds = {task.task_id: (42,) for task in tasks}
    print("Selected model-capability-v1 matrix:")
    for profile in profiles:
        print(f"{profile}:")
        print("  smoke")
        for task in tasks:
            print(f"  {task.task_id} seed 42")

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
    result = ModelCapabilityRunner(catalog, args.repository, snapshot).run(
        profiles, tasks, seeds
    )
    write_model_capability_json(result, args.output)
    print(render_model_capability_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

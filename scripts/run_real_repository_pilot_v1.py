"""Freeze and run the exactly-once A60 Foundation model matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from benchmarks.real_repository_pilot_v1.runner import (
    checkpoint_path,
    commit_cell,
    freeze_manifest,
    read_checkpoint,
    run_cell,
    summarize,
)
from benchmarks.real_repository_pilot_v1.suite import (
    BUILD,
    CONFIGURE,
    TEST,
    create_snapshot,
    repository_identity,
    snapshot_description,
    tasks,
    validate_integrity,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--profile", choices=("qwen-small", "qwen-large", "codestral-22b")
    )
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

    canonical = args.repository.resolve()
    root = args.output_root.resolve()
    snapshot = root / "foundation-snapshot"
    before = repository_identity(canonical)
    if not snapshot.exists():
        root.mkdir(parents=True, exist_ok=True)
        create_snapshot(canonical, snapshot)
    if repository_identity(snapshot) != before:
        raise RuntimeError("A60 snapshot differs from canonical source identity")
    definitions = tasks()
    integrity_path = root / "integrity.json"
    if integrity_path.exists():
        from benchmarks.real_repository_pilot_v1.suite import Integrity

        integrity = tuple(
            Integrity(**item) for item in json.loads(integrity_path.read_text())
        )
    else:
        integrity = validate_integrity(snapshot, definitions)
        from benchmarks.real_repository_pilot_v1.runner import _atomic_json

        _atomic_json(
            integrity_path,
            [
                item.__dict__
                if hasattr(item, "__dict__")
                else {
                    field: getattr(item, field) for field in item.__dataclass_fields__
                }
                for item in integrity
            ],
        )
    manifest = freeze_manifest(root, canonical, snapshot, definitions, integrity)
    if args.integrity_only:
        baseline_started = time.perf_counter()
        baseline = all(
            subprocess.run(command, cwd=snapshot, shell=False, check=False).returncode
            == 0
            for command in (CONFIGURE, BUILD, TEST)
        )
        print(
            json.dumps(
                {
                    "snapshot": snapshot_description(
                        snapshot, time.perf_counter() - baseline_started
                    ).__dict__
                    if hasattr(snapshot_description(snapshot), "__dict__")
                    else {
                        field: getattr(snapshot_description(snapshot), field)
                        for field in snapshot_description(snapshot).__dataclass_fields__
                    },
                    "integrity": [
                        {
                            field: getattr(item, field)
                            for field in item.__dataclass_fields__
                        }
                        for item in integrity
                    ],
                    "baseline": baseline,
                    "canonical_unchanged": repository_identity(canonical) == before,
                },
                default=str,
                sort_keys=True,
            )
        )
        return 0
    if args.config is None or args.profile is None:
        parser.error("--config and --profile are required for model cells")
    selected = {item.task_id for item in definitions}
    if args.tasks != "all":
        selected = {value.strip() for value in args.tasks.split(",") if value.strip()}
        if not selected.issubset({item.task_id for item in definitions}):
            parser.error("unknown A60 task ID")
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if (
        not isinstance(profile.backend_config, LlamaCppConfig)
        or profile.backend_config.context_size != 8192
    ):
        raise RuntimeError("A60 requires unchanged local llama.cpp 8192-token profiles")
    artifact = artifact_identity(profile.backend_config.model_path)
    missing = []
    for definition in definitions:
        if definition.task_id not in selected:
            continue
        path = checkpoint_path(root, args.profile, definition.task_id)
        expected = {
            "suite": manifest["suite"],
            "schema_version": 1,
            "task_id": definition.task_id,
            "task_version": 1,
            "model_profile": args.profile,
            "model_artifact": artifact,
            "seed": 42,
            "repository_identity": manifest["repository_identity"],
            "manifest_identity": manifest["manifest_identity"],
        }
        if read_checkpoint(path, expected) is None:
            missing.append(definition)
        else:
            print(f"skip committed {args.profile} {definition.task_id}", flush=True)
    if missing:
        model = catalog.create(args.profile)
        try:
            for definition in missing:
                cell = run_cell(
                    definition,
                    model,
                    snapshot,
                    profile=args.profile,
                    artifact=artifact,
                    manifest=manifest,
                    representation=profile.mutation_representation,
                )
                if repository_identity(canonical) != before:
                    raise RuntimeError("canonical Foundation changed during A60")
                commit_cell(
                    checkpoint_path(root, args.profile, definition.task_id), cell
                )
                print(
                    f"checkpointed {args.profile} {definition.task_id}: "
                    f"{cell.first_failure_layer} semantic={cell.semantic_pass}",
                    flush=True,
                )
        finally:
            model.close()
    cells = tuple(
        json.loads(path.read_text())
        for path in sorted((root / "results").glob("*.json"))
    )
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    from benchmarks.real_repository_pilot_v1.runner import _atomic_json

    _atomic_json(summary_path, summarize(cells))
    if repository_identity(canonical) != before:
        raise RuntimeError("canonical Foundation identity changed after A60")
    print(json.dumps(summarize(cells), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

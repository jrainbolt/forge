"""Freeze and run the exactly-once A62 Foundation model matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.real_repository_pilot_v1.suite import (
    Integrity,
    create_snapshot,
    repository_identity,
    source_manifest,
    validate_integrity,
)
from benchmarks.real_repository_pilot_v2.runner import (
    atomic_json,
    checkpoint_path,
    commit_cell,
    read_checkpoint,
    run_cell,
    summarize,
)
from benchmarks.real_repository_pilot_v2.suite import (
    SEED,
    SUITE,
    VERSION,
    definition_identity,
    integrity_payload,
    tasks,
    validate_boundary,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog

PROFILES = ("qwen-small", "qwen-large", "codestral-22b")


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    args = parser.parse_args()

    selected_profiles = tuple(
        item.strip() for item in args.profiles.split(",") if item.strip()
    )
    if any(item not in PROFILES for item in selected_profiles):
        parser.error("unknown A62 profile")

    canonical = args.repository.resolve(strict=True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "foundation-snapshot"
    before = repository_identity(canonical)
    if not snapshot.exists():
        create_snapshot(canonical, snapshot)
    if repository_identity(snapshot) != before:
        raise RuntimeError("A62 snapshot differs from canonical Foundation")

    definitions = tasks()
    if len(definitions) != 8:
        raise RuntimeError("A62 requires exactly eight frozen tasks")
    for definition in definitions:
        validate_boundary(definition)

    integrity_path = root / "integrity.json"
    if integrity_path.exists():
        integrity = tuple(
            Integrity(**item) for item in json.loads(integrity_path.read_text())
        )
    else:
        integrity = validate_integrity(snapshot, definitions)  # type: ignore[arg-type]
        atomic_json(integrity_path, integrity_payload(integrity))
    if len(integrity) != 8 or not all(item.eligible for item in integrity):
        raise RuntimeError("A62 task integrity is not fully discriminating")

    manifest = {
        "suite": SUITE,
        "schema_version": VERSION,
        "canonical_identity": before,
        "repository_identity": repository_identity(snapshot),
        "source_file_manifest": source_manifest(snapshot),
        "manifest_identity": definition_identity(definitions, snapshot),
        "task_ids": [item.versioned_id for item in definitions],
        "integrity": integrity_payload(integrity),
        "seed": SEED,
        "temperature": 0,
        "context": 8192,
        "output": 512,
        "ephemeral_acceptance": False,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError("A62 frozen manifest changed")
    else:
        atomic_json(manifest_path, manifest)

    catalog = load_model_catalog(args.config, default_backend_registry())
    for profile_name in selected_profiles:
        profile = catalog.profile(profile_name)
        if (
            not isinstance(profile.backend_config, LlamaCppConfig)
            or profile.backend_config.context_size != 8192
        ):
            raise RuntimeError("A62 requires unchanged 8192-token profiles")
        artifact = artifact_identity(profile.backend_config.model_path)
        pending = []
        for definition in definitions:
            path = checkpoint_path(root, profile_name, definition.task_id)
            expected = {
                "suite": SUITE,
                "schema_version": VERSION,
                "task_id": definition.task_id,
                "task_version": VERSION,
                "model_profile": profile_name,
                "model_artifact": artifact,
                "seed": SEED,
                "repository_identity": manifest["repository_identity"],
                "manifest_identity": manifest["manifest_identity"],
            }
            if read_checkpoint(path, expected) is None:
                pending.append(definition)
            else:
                print(f"skip committed {profile_name} {definition.task_id}", flush=True)
        if not pending:
            continue
        model = catalog.create(profile_name)
        try:
            for definition in pending:
                cell = run_cell(
                    definition,
                    model,
                    snapshot,
                    profile=profile_name,
                    artifact=artifact,
                    repository_identity=str(manifest["repository_identity"]),
                    manifest_identity=str(manifest["manifest_identity"]),
                    representation=profile.mutation_representation,
                )
                if repository_identity(canonical) != before:
                    raise RuntimeError("canonical Foundation changed during A62")
                commit_cell(
                    checkpoint_path(root, profile_name, definition.task_id), cell
                )
                print(
                    f"checkpointed {profile_name} {definition.task_id}: "
                    f"{cell.first_failure_layer} context={cell.context_quality} "
                    f"semantic={cell.semantic_pass}",
                    flush=True,
                )
        finally:
            model.close()

    cells = tuple(
        json.loads(path.read_text())
        for path in sorted((root / "results").glob("*.json"))
    )
    summary = summarize(cells)
    summary_path = root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing != summary:
            raise RuntimeError("A62 summary already exists with different content")
    else:
        atomic_json(summary_path, summary)
    if repository_identity(canonical) != before:
        raise RuntimeError("canonical Foundation identity changed after A62")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

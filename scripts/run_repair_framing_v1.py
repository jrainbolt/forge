"""Capture A57 primaries, freeze their identities, and resume condition cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import _atomic_json
from benchmarks.realistic_coding_v2.suite import manifest_identity, tasks
from benchmarks.repair_framing_v1.runner import (
    capture_primary,
    diagnose,
    freeze_manifest,
    load_primary,
    load_trial,
    primary_path,
    run_condition,
)
from benchmarks.repair_framing_v1.suite import (
    REPOSITORY,
    cases,
    load_a56_cell,
)
from forge.evaluation.realworld import hash_workspace
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a56-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--stage", choices=("capture", "freeze", "conditions"), required=True
    )
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    selected = tuple(case for case in cases() if case.profile == args.profile)
    if args.cases != "all":
        requested = set(args.cases.split(","))
        selected = tuple(case for case in selected if case.case_id in requested)
        if requested != {case.case_id for case in selected}:
            parser.error("unknown or cross-profile case ID")
    if not selected:
        parser.error("no cases selected")
    if args.a56_dir is not None:
        for case in selected:
            load_a56_cell(args.a56_dir, case)
    if args.stage == "freeze":
        if args.cases != "all":
            parser.error("freeze requires the full selected corpus")
        records = tuple(load_primary(args.root, case)[0] for case in cases())
        frozen = freeze_manifest(args.root, records)
        print(
            json.dumps(
                {
                    "cases": len(records),
                    "eligible": sum(record.eligible for record in records),
                    "manifest_hash": hashlib.sha256(
                        json.dumps(frozen, sort_keys=True).encode()
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A57 requires the unchanged local llama.cpp profile")
    if profile.backend_config.context_size != 8192:
        raise RuntimeError("A57 requires context 8192")
    artifact = _artifact_identity(profile.backend_config.model_path)
    canonical_before = hash_workspace(REPOSITORY)
    if args.stage == "conditions":
        manifest_path = args.root / "results" / "manifest.json"
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest()
        if frozen["a56_definition_identity"] != manifest_identity(tasks()):
            raise ValueError("frozen A56 definition identity changed")
        if (
            frozen["canonical_repository_hash"]
            != hashlib.sha256(repr(canonical_before).encode()).hexdigest()
        ):
            raise ValueError("canonical A56 repository identity changed")
    missing = []
    for case in selected:
        if args.stage == "capture":
            if primary_path(args.root, case).exists():
                record, _ = load_primary(args.root, case)
                if record.model_artifact != artifact:
                    raise ValueError("committed primary model identity changed")
                print(f"skip committed primary {case.case_id}", flush=True)
                continue
            missing.append(case)
        else:
            record, _ = load_primary(args.root, case)
            if not record.eligible:
                continue
            if record.model_artifact != artifact:
                raise ValueError("committed primary model identity changed")
            for condition in ("R0", "R1", "R2"):
                if load_trial(args.root, case, condition, manifest_hash) is None:
                    missing.append((case, condition))
                else:
                    print(f"skip committed {case.case_id} {condition}", flush=True)
    if not missing:
        print("all selected cells committed", flush=True)
        return 0
    model = catalog.create(args.profile)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("loaded model context does not match A57 profile")
    try:
        for item in missing:
            if args.stage == "capture":
                case = item
                record = capture_primary(
                    args.root,
                    case,
                    model,
                    artifact=artifact,
                    a56_manifest=manifest_identity(tasks()),
                    representation=profile.mutation_representation,
                )
                print(
                    f"captured {case.case_id}: eligible={record.eligible} "
                    f"authority={record.authority_sufficient} "
                    f"exclusion={record.exclusion}",
                    flush=True,
                )
            else:
                case, condition = item
                record, replay = load_primary(args.root, case)
                trial = run_condition(
                    args.root,
                    case,
                    record,
                    replay,
                    model,
                    profile.mutation_representation,
                    condition,
                    manifest_hash,
                )
                print(
                    f"checkpointed {case.case_id} {condition}: "
                    f"{trial.classification} semantic={trial.semantic_recovered}",
                    flush=True,
                )
            if canonical_before != hash_workspace(REPOSITORY):
                raise RuntimeError("canonical A56 repository changed")
    finally:
        model.close()
    if args.stage == "conditions":
        for case in selected:
            record, _ = load_primary(args.root, case)
            if not record.eligible:
                continue
            trials = tuple(
                load_trial(args.root, case, condition, manifest_hash)
                for condition in ("R0", "R1", "R2")
            )
            if all(trial is not None for trial in trials):
                _atomic_json(
                    args.root / "results" / f"{case.case_id}-summary.json",
                    {
                        "case_id": case.case_id,
                        "diagnosis": diagnose(trials),
                        "conditions": [asdict(trial) for trial in trials],
                    },
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

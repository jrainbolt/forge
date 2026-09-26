"""Freeze and exactly-once resume A59 primary and repair cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from benchmarks.cross_model_repair_confirmation_v1.runner import (
    candidate_cases,
    freeze_cases,
    freeze_plan,
    identity,
    load_cell,
    run_repair,
)
from benchmarks.cross_model_repair_confirmation_v1.suite import PRIMARY_PROFILES
from benchmarks.realistic_coding_v2.runner import _atomic_json
from benchmarks.repair_framing_v1.runner import (
    capture_primary,
    load_primary,
    primary_path,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog

PROFILES = (*PRIMARY_PROFILES, "qwen-large")


def _artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def _record_load(root: Path, key: str, profile: str, seconds: float) -> None:
    path = root / "results" / f"load-{key}.json"
    if not path.exists():
        _atomic_json(
            path,
            {"key": key, "profile": profile, "load_seconds": round(seconds, 3)},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("plan", "primary", "freeze", "same", "handoff"),
        required=True,
    )
    parser.add_argument("--profile", choices=PROFILES)
    args = parser.parse_args()
    catalog = load_model_catalog(args.config, default_backend_registry())
    artifacts = {}
    representations = {}
    for name in PROFILES:
        profile = catalog.profile(name)
        if not isinstance(profile.backend_config, LlamaCppConfig):
            raise RuntimeError("A59 requires trusted local llama.cpp profiles")
        if profile.backend_config.context_size != 8192:
            raise RuntimeError("A59 requires context 8192")
        artifacts[name] = _artifact_identity(profile.backend_config.model_path)
        representations[name] = profile.mutation_representation
    if args.stage == "plan":
        plan = freeze_plan(args.root, artifacts, representations)
        print(
            json.dumps(
                {
                    "plan_hash": identity(plan),
                    "primary_cells": len(plan["candidates"]),
                    "task_definitions": 8,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    plan = json.loads((args.root / "plan.json").read_text(encoding="utf-8"))
    if artifacts != plan["model_artifacts"]:
        raise ValueError("A59 model artifacts changed after plan freeze")
    cases = candidate_cases()
    if args.stage == "primary":
        if args.profile not in PRIMARY_PROFILES:
            parser.error("primary requires --profile qwen-small or codestral-22b")
        selected = tuple(case for case in cases if case.profile == args.profile)
        missing = [
            case for case in selected if not primary_path(args.root, case).exists()
        ]
        for case in selected:
            if case not in missing:
                print(f"skip committed primary {case.case_id}", flush=True)
        if not missing:
            print("all selected primaries committed", flush=True)
            return 0
        load_started = time.perf_counter()
        model = catalog.create(args.profile)
        load_seconds = time.perf_counter() - load_started
        _record_load(args.root, f"primary-{args.profile}", args.profile, load_seconds)
        try:
            for case in missing:
                record = capture_primary(
                    args.root,
                    case,
                    model,
                    artifact=artifacts[args.profile],
                    a56_manifest=str(plan["definition_identity"]),
                    representation=representations[args.profile],
                )
                print(
                    f"captured {case.case_id} {case.task_id}: "
                    f"eligible={record.eligible} exclusion={record.exclusion}",
                    flush=True,
                )
        finally:
            model.close()
        return 0
    if args.stage == "freeze":
        manifest = freeze_cases(args.root, plan)
        print(
            json.dumps(
                {
                    "manifest_hash": identity(manifest),
                    "planned": len(manifest["records"]),
                    "eligible": manifest["eligible_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    manifest = json.loads(
        (args.root / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_id = identity(manifest)
    scored = set(manifest["scored_case_ids"])
    selected = tuple(case for case in cases if case.case_id in scored)
    if args.stage == "same":
        if args.profile not in PRIMARY_PROFILES:
            parser.error("same requires a primary --profile")
        selected = tuple(case for case in selected if case.profile == args.profile)
        repair_profile = args.profile
        condition = "R0"
    else:
        if args.profile not in {None, "qwen-large"}:
            parser.error("handoff uses qwen-large")
        repair_profile = "qwen-large"
        condition = "R1"
    missing = [
        case
        for case in selected
        if load_cell(args.root, case, condition, manifest_id) is None
    ]
    for case in selected:
        if case not in missing:
            print(f"skip committed repair {case.case_id}-{condition}", flush=True)
    if not missing:
        print("all selected repairs committed", flush=True)
        return 0
    load_started = time.perf_counter()
    model = catalog.create(repair_profile)
    load_seconds = time.perf_counter() - load_started
    _record_load(
        args.root, f"{condition}-{repair_profile}", repair_profile, load_seconds
    )
    try:
        for case in missing:
            record, replay = load_primary(args.root, case)
            cell = run_repair(
                args.root,
                case,
                record,
                replay,
                model,
                repair_profile,
                artifacts[repair_profile],
                representations[case.profile],
                condition,
                manifest_id,
            )
            print(
                f"checkpointed {case.case_id}-{condition}: "
                f"{cell.failure_taxonomy} semantic={cell.semantic_recovered}",
                flush=True,
            )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

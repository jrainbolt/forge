"""Freeze and exactly-once resume the A58 cross-model repair matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.cross_model_repair_v1.runner import (
    freeze_manifest,
    load_cell,
    manifest_hash,
    reuse_diagonal,
    run_cross_cell,
    summarize_case,
)
from benchmarks.cross_model_repair_v1.suite import REPAIR_PROFILES, pairs
from benchmarks.realistic_coding_v2.runner import _atomic_json
from benchmarks.repair_framing_v1.runner import load_primary
from benchmarks.repair_framing_v1.suite import cases
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
    parser.add_argument("--a57-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "conditions"), required=True)
    parser.add_argument("--repair-profile", choices=REPAIR_PROFILES)
    args = parser.parse_args()
    pair_values = pairs(cases())
    catalog = load_model_catalog(args.config, default_backend_registry())
    artifacts: dict[str, str] = {}
    representations = {}
    for name in REPAIR_PROFILES:
        profile = catalog.profile(name)
        if not isinstance(profile.backend_config, LlamaCppConfig):
            raise RuntimeError("A58 requires unchanged local llama.cpp profiles")
        if profile.backend_config.context_size != 8192:
            raise RuntimeError("A58 requires context 8192")
        artifacts[name] = _artifact_identity(profile.backend_config.model_path)
        representations[name] = profile.mutation_representation
    if args.stage == "freeze":
        frozen = freeze_manifest(
            args.root, args.a57_root, pair_values, artifacts, representations
        )
        print(
            json.dumps(
                {
                    "cases": len(frozen["cases"]),
                    "repair_profiles": len(frozen["repair_model_artifacts"]),
                    "cells": len(pair_values),
                    "manifest_hash": manifest_hash(frozen),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.repair_profile is None:
        parser.error("--repair-profile is required for conditions")
    manifest = json.loads(
        (args.root / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    expected_manifest = manifest_hash(manifest)
    if manifest["repair_model_artifacts"] != artifacts:
        raise ValueError("A58 repair model artifacts changed after freeze")
    selected = tuple(
        pair for pair in pair_values if pair.repair_profile == args.repair_profile
    )
    missing_cross = []
    for pair in selected:
        record, replay = load_primary(args.a57_root, pair.case)
        existing = load_cell(args.root, pair, expected_manifest)
        if existing is not None:
            print(f"skip committed {pair.key}", flush=True)
            continue
        if pair.same_model:
            cell = reuse_diagonal(
                args.root,
                args.a57_root,
                pair,
                record,
                artifacts[pair.repair_profile],
                representations[pair.case.profile],
                expected_manifest,
            )
            print(
                f"reused A57 diagonal {pair.key}: {cell.failure_taxonomy}", flush=True
            )
        else:
            missing_cross.append((pair, record, replay))
    if missing_cross:
        model = catalog.create(args.repair_profile)
        if model.context_capacity != 8192:
            model.close()
            raise RuntimeError("loaded repair model context does not match A58")
        try:
            for pair, record, replay in missing_cross:
                cell = run_cross_cell(
                    args.root,
                    args.a57_root,
                    pair,
                    record,
                    replay,
                    model,
                    artifacts[pair.repair_profile],
                    representations[pair.case.profile],
                    expected_manifest,
                )
                print(
                    f"checkpointed {pair.key}: {cell.failure_taxonomy} "
                    f"semantic={cell.semantic_recovered}",
                    flush=True,
                )
        finally:
            model.close()
    for case in cases():
        case_pairs = tuple(pair for pair in pair_values if pair.case == case)
        if not case_pairs:
            continue
        cells = tuple(
            cell
            for pair in case_pairs
            if (cell := load_cell(args.root, pair, expected_manifest))
        )
        if len(cells) == 3:
            summary = summarize_case(cells)
            summary_path = args.root / "results" / f"{case.case_id}-summary.json"
            if summary_path.exists():
                if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
                    raise ValueError("A58 committed case summary changed")
            else:
                _atomic_json(summary_path, summary)
    if not missing_cross:
        print("all selected cells committed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

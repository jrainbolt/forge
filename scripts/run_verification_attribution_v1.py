"""Run or resume one profile's frozen A73 diagnostic cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.verification_attribution_v1.runner import (
    checkpoint_path,
    commit_case,
    read_case,
    run_case,
)
from benchmarks.verification_attribution_v1.suite import (
    CASES,
    CONTEXT_SIZE,
    PROFILES,
    corpus_identity,
    task_map,
    validate_corpus,
)
from forge.evaluation.replay import source_state_identity
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    validate_corpus()
    corpus = corpus_identity()
    baseline_path = args.checkpoint_dir / "baseline.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["corpus_identity"] != corpus:
        raise RuntimeError("A73 baseline corpus identity mismatch")
    definitions = task_map()
    selected = tuple(case for case in CASES if case.profile == args.profile)
    repository_identity = source_state_identity(REPOSITORY)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A73 requires local llama.cpp profiles")
    if profile.backend_config.context_size != CONTEXT_SIZE:
        raise RuntimeError("A73 requires context 8192")
    artifact = _file_identity(profile.backend_config.model_path)
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    pending = []
    for case in selected:
        existing = read_case(
            checkpoint_path(args.checkpoint_dir, case.case_id),
            {
                "case_id": case.case_id,
                "profile": case.profile,
                "corpus_identity": corpus,
                "repository_identity": repository_identity,
                "model_artifact": artifact,
                "model_config_identity": config_identity,
            },
        )
        if existing is None:
            pending.append(case)
        else:
            print(f"skip committed {case.case_id}", flush=True)
    if not pending:
        return 0
    canonical_before = source_state_identity(REPOSITORY)
    model = catalog.create(args.profile)
    try:
        for case in pending:
            result = run_case(
                case,
                definitions[case.task_id],
                model,
                artifact=artifact,
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus,
                representation=profile.mutation_representation,
                baseline_valid=bool(baseline["baselines"][case.task_id]["eligible"]),
            )
            commit_case(checkpoint_path(args.checkpoint_dir, case.case_id), result)
            print(
                json.dumps({"case": case.case_id, "applied": len(result.proposals)}),
                flush=True,
            )
    finally:
        model.close()
    if source_state_identity(REPOSITORY) != canonical_before:
        raise RuntimeError("canonical source identity changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

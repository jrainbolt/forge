"""Prepare A46 from trusted profiles without downloading model artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    CandidateStatus,
    build_bakeoff_run,
    enumerate_trusted_candidates,
    hash_workspace,
    load_a45_baseline,
    write_alternative_model_bakeoff_json,
)
from forge.models import default_backend_registry, load_model_catalog


def _repository_identity() -> str:
    return hashlib.sha256(repr(hash_workspace(REPOSITORY)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        required=True,
        help="immutable A45 result artifact; specify once per baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical_before = hash_workspace(REPOSITORY)
    integrity = validate_integrity(realistic_semantic_tasks((42,)))
    if len(integrity) != 8 or not all(
        not item.baseline_oracle_pass
        and item.reference_oracle_pass
        and not item.wrong_mutation_oracle_pass
        and item.semantic_score_eligible
        for item in integrity
    ):
        raise RuntimeError("frozen A45 integrity check failed")

    repository_identity = _repository_identity()
    baselines = tuple(load_a45_baseline(path) for path in args.baseline)
    catalog = load_model_catalog(args.config, default_backend_registry())
    candidates = enumerate_trusted_candidates(catalog)
    eligible = tuple(
        item for item in candidates if item.status is CandidateStatus.ELIGIBLE
    )
    run = build_bakeoff_run(
        repository_identity=repository_identity,
        candidates=candidates,
        baselines=baselines,
        canonical_unchanged=canonical_before == hash_workspace(REPOSITORY),
    )
    write_alternative_model_bakeoff_json(run, args.output)

    for item in candidates:
        print(
            f"{item.profile}: {item.status.value} "
            f"artifact={item.artifact or '-'} reason={item.reason}"
        )
    if not eligible:
        print("BLOCKED: no genuinely new eligible local model profile")
        return 2
    print(f"Eligible alternatives: {len(eligible)}")
    print("Run load/protocol smoke in fresh processes before the model matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

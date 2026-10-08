"""Run or resume one profile's frozen A76 diagnostic cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.behavioral_obligation_diagnosis_v1.runner import (
    checkpoint_path,
    commit_cell,
    read_cell,
    run_cell,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    CONTEXT_SIZE,
    PROFILES,
    RUN_ID,
    Condition,
    cases,
    corpus_identity,
    obligations,
    task_map,
    validate_corpus,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
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
    definitions = task_map()
    repository_identity = source_state_identity(REPOSITORY)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A76 requires local llama.cpp profiles")
    if profile.backend_config.context_size != CONTEXT_SIZE:
        raise RuntimeError("A76 requires context 8192")
    artifact = _file_identity(profile.backend_config.model_path)
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    pending = []
    for case in (item for item in cases() if item.profile == args.profile):
        cells = [(Condition.F0, None)]
        cells.extend(
            (Condition.F1, obligation)
            for obligation in obligations(case)
            if obligation.isolatable
        )
        for condition, obligation in cells:
            cell_id = f"{case.case_id}-{condition.name}" + (
                f"-{obligation.goal_id}" if obligation else ""
            )
            expected = {
                "run_identity": RUN_ID,
                "cell_id": cell_id,
                "corpus_identity": corpus,
                "repository_identity": repository_identity,
                "model_artifact": artifact,
                "model_config_identity": config_identity,
            }
            if (
                read_cell(checkpoint_path(args.checkpoint_dir, cell_id), expected)
                is None
            ):
                pending.append((case, condition, obligation))
            else:
                print(f"skip committed {cell_id}", flush=True)
    if not pending:
        return 0
    canonical_before = source_state_identity(REPOSITORY)
    model = catalog.create(args.profile)
    try:
        for case, condition, obligation in pending:
            result = run_cell(
                case,
                condition,
                definitions[case.task_id],
                model,
                obligation=obligation,
                artifact=artifact,
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus,
                representation=profile.mutation_representation,
            )
            commit_cell(checkpoint_path(args.checkpoint_dir, result.cell_id), result)
            print(
                json.dumps(
                    {
                        "cell": result.cell_id,
                        "applied": result.proposal["transaction_applied"],
                        "semantic_pass": result.full_task_semantic_pass,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    if source_state_identity(REPOSITORY) != canonical_before:
        raise RuntimeError("canonical source identity changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

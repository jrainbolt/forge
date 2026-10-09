"""Freeze A81 tasks, obligations, and checks before model execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.semantic_obligation_completeness_v1.suite import (
    CHECK_SET_IDENTITY,
    OBLIGATIONS,
    RUN_ID,
    TASK_IDS,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A81 qualification is missing")
    if not all(qualification["task_health"][task]["eligible"] for task in TASK_IDS):
        raise RuntimeError("A81 contains an unqualified task")
    atomic_checkpoint(
        args.output,
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "task_ids": TASK_IDS,
            "obligation_identities": {
                task: [item.identity for item in OBLIGATIONS[task]] for task in TASK_IDS
            },
            "obligations_frozen_before_execution": True,
            "production_visible_origins_only": True,
            "oracle_wording_in_model_input": False,
            "check_set_identity": CHECK_SET_IDENTITY,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

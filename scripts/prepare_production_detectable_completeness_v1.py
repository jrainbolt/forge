"""Freeze A82 structural checks before model execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.production_detectable_completeness_v1.suite import (
    CASES,
    RUN_ID,
    TASK_IDS,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    atomic_checkpoint(
        args.output,
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "task_ids": TASK_IDS,
            "check_identities": {
                case.task_id: [check.identity for check in case.checks]
                for case in CASES
            },
            "checks_frozen_before_execution": True,
            "production_visible_only": True,
            "hidden_oracle_data": False,
            "behavioral_inference": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

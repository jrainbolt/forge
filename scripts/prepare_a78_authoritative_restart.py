"""Prepare, but never execute, the authoritative post-correction A78 namespace."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.suite import (
    CHECKPOINT_NAMESPACE,
    PROFILES,
    QUALIFICATION_RUN_ID,
    RUN_ID,
    corpus_identity,
    tasks,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("historical A78 qualification is missing")
    if qualification.get("run_identity") != QUALIFICATION_RUN_ID:
        raise RuntimeError("historical A78 qualification identity mismatch")
    if qualification.get("corpus_identity") != corpus_identity():
        raise RuntimeError("frozen A78 corpus changed")
    if (args.checkpoint_dir / "cells").exists():
        raise RuntimeError("authoritative A78 restart namespace already contains cells")
    atomic_checkpoint(
        args.checkpoint_dir / "restart.json",
        {
            "run_identity": RUN_ID,
            "checkpoint_namespace": CHECKPOINT_NAMESPACE,
            "corpus_identity": corpus_identity(),
            "qualification_identity": QUALIFICATION_RUN_ID,
            "profiles": PROFILES,
            "task_ids": tuple(item.task_id for item in tasks()),
            "planned_cells": len(tasks()) * len(PROFILES),
            "executed_cells": 0,
            "historical_cells_reused": 0,
            "request_identity_version": 1,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

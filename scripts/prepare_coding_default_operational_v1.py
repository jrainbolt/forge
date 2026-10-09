"""Freeze the qualified A80 operational corpus before model execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.coding_default_operational_v1.suite import (
    CHECKPOINT_NAMESPACE,
    REPRESENTATIVE_TASK_IDS,
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
        raise RuntimeError("A78 qualification is missing")
    health = qualification["task_health"]
    reachability = qualification["workflow_reachability"]
    if not all(health[task_id]["eligible"] for task_id in TASK_IDS):
        raise RuntimeError("A80 contains a semantically unqualified task")
    if not all(
        reachability[task_id]["classification"] == "WORKFLOW_REACHABLE"
        for task_id in TASK_IDS
    ):
        raise RuntimeError("A80 contains a workflow-unreachable task")
    atomic_checkpoint(
        args.output,
        {
            "run_identity": RUN_ID,
            "checkpoint_namespace": CHECKPOINT_NAMESPACE,
            "corpus_identity": corpus_identity(),
            "task_ids": TASK_IDS,
            "representative_task_ids": REPRESENTATIVE_TASK_IDS,
            "semantic_qualified": len(TASK_IDS),
            "workflow_reachable": len(TASK_IDS),
            "authority_valid": len(TASK_IDS),
            "frozen_before_model_execution": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

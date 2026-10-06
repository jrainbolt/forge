"""Run deterministic source-free controls for A72 proposal evidence binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.runner import bind_proposal_evidence
from forge.evaluation.mutation_ready import atomic_checkpoint


def _proposal(proposal_id: str) -> dict[str, object]:
    return {"proposal_observation_id": proposal_id}


def _transaction(proposal_id: str, attempt_id: str, outcome: str) -> dict[str, object]:
    return {
        "event": "transaction_result",
        "proposal_observation_id": proposal_id,
        "group_identity": f"group-{proposal_id}",
        "workspace_generation": 1,
        "transaction_attempt_id": attempt_id,
        "transaction_outcome": outcome,
    }


def controls() -> dict[str, object]:
    cases = {
        "single_accepted": (
            (_proposal("p1"),),
            (_transaction("p1", "a1", "success"),),
        ),
        "single_rejected": (
            (_proposal("p1"),),
            (_transaction("p1", "a1", "failure"),),
        ),
        "multi_one_applied": (
            (_proposal("p1"), _proposal("p2")),
            (
                _transaction("p1", "a1", "failure"),
                _transaction("p2", "a2", "success"),
            ),
        ),
        "mixed_create_multi_attempt": (
            (_proposal("mixed-1"), _proposal("mixed-2")),
            (
                _transaction("mixed-1", "m1", "success"),
                _transaction("mixed-2", "m2", "failure"),
            ),
        ),
    }
    results = {
        name: bind_proposal_evidence(metadata, evidence)
        for name, (metadata, evidence) in cases.items()
    }
    return {
        "suite": "transaction-association-controls-v1",
        "cases": len(results),
        "results": results,
        "all_consistent": all(
            sum(bool(value["transaction_applied"]) for value in result.values())
            == (0 if name == "single_rejected" else 1)
            for name, result in results.items()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = controls()
    if not result["all_consistent"] or not standard_result_is_source_free(result):
        raise RuntimeError("A72 association controls failed")
    atomic_checkpoint(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge non-duplicated realistic-semantic cell checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    RealisticSemanticResult,
    RealisticSemanticRun,
    SemanticIntegrityResult,
    aggregate_realistic_results,
    write_realistic_semantic_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.checkpoint_dir.glob("*.json"))
    ]
    if not documents:
        raise RuntimeError("no checkpoints found")
    first = documents[0]
    invariant_keys = (
        "suite",
        "suite_version",
        "schema_version",
        "forge_milestone",
        "repository_identity",
        "model_profile",
        "model_artifact",
        "context_capacity",
        "output_budget",
        "temperature",
    )
    if any(
        any(document[key] != first[key] for key in invariant_keys)
        for document in documents[1:]
    ):
        raise RuntimeError("checkpoint invariants differ")
    raw_results = [document["results"][0] for document in documents]
    if len({item["task_id"] for item in raw_results}) != len(raw_results):
        raise RuntimeError("duplicate checkpoint task IDs")
    results = tuple(_terminal_classification(item) for item in raw_results)
    integrity = tuple(
        SemanticIntegrityResult(**document["integrity"][0]) for document in documents
    )
    seeds = {item.seed for item in results}
    if len(seeds) != 1:
        raise RuntimeError("checkpoints contain multiple seeds")
    seed = seeds.pop()
    run = RealisticSemanticRun(
        first["suite"],
        first["suite_version"],
        first["schema_version"],
        first["forge_milestone"],
        first["repository_identity"],
        first["model_profile"],
        first["model_artifact"],
        first["context_capacity"],
        first["output_budget"],
        first["temperature"],
        integrity,
        results,
        (aggregate_realistic_results(first["model_profile"], seed, results),),
        all(document["canonical_unchanged"] for document in documents),
    )
    write_realistic_semantic_json(run, args.output)
    return 0


def _terminal_classification(raw: dict[str, object]) -> RealisticSemanticResult:
    result = RealisticSemanticResult(**raw)  # type: ignore[arg-type]
    if result.final_semantic and result.verification_status in {"pass", "passed"}:
        return replace(result, failure_layer="PASS")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

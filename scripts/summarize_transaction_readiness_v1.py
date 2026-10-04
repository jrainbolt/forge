"""Summarize committed source-free A71 cells without replay or reconstruction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.transaction_readiness_v1.protocol import STAGES
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v1.suite import PROFILES, tasks
from forge.evaluation.mutation_ready import atomic_checkpoint


def _records(cells: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    records = []
    for cell in cells:
        proposals = cell["proposals"]
        assert isinstance(proposals, list)
        if not proposals:
            records.append(
                {
                    "model_profile": cell["model_profile"],
                    "operation_class": cell["operation_class"],
                    "transitions": tuple((stage.value, False) for stage in STAGES),
                    "failure_layer": "MODEL_PROTOCOL_FAILURE",
                }
            )
            continue
        for proposal in proposals:
            assert isinstance(proposal, dict)
            records.append(
                {
                    "model_profile": cell["model_profile"],
                    "operation_class": cell["operation_class"],
                    "transitions": proposal["transitions"],
                    "failure_layer": proposal["failure_layer"],
                }
            )
    return tuple(records)


def _aggregate(records: tuple[dict[str, object], ...], key: str) -> dict[str, object]:
    values = {}
    for name in sorted({str(item[key]) for item in records}):
        selected = tuple(item for item in records if item[key] == name)
        transitions = {
            stage.value: sum(
                bool(dict(item["transitions"])[stage.value])  # type: ignore[arg-type]
                for item in selected
            )
            for stage in STAGES
        }
        values[name] = {
            "records": len(selected),
            "transitions": transitions,
            "first_failures": dict(
                sorted(Counter(str(item["failure_layer"]) for item in selected).items())
            ),
        }
    return values


def summarize(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    records = _records(cells)
    transitions = {
        stage.value: sum(
            bool(dict(item["transitions"])[stage.value])  # type: ignore[arg-type]
            for item in records
        )
        for stage in STAGES
    }
    ordered = tuple(stage.value for stage in STAGES)
    attrition = {
        f"{left}->{right}": transitions[left] - transitions[right]
        for left, right in zip(ordered, ordered[1:], strict=False)
    }

    def ratio(numerator: str, denominator: str) -> float | None:
        value = transitions[denominator]
        return round(transitions[numerator] / value, 4) if value else None

    return {
        "suite": "transaction-readiness-v1",
        "schema_version": 1,
        "cells": len(cells),
        "proposal_records_including_no-output_cells": len(records),
        "transitions": transitions,
        "conversion": {
            "schema_to_materializable": ratio(
                "MECHANICALLY_MATERIALIZABLE", "SCHEMA_VALID"
            ),
            "materializable_to_production_validatable": ratio(
                "PRODUCTION_VALIDATABLE", "MECHANICALLY_MATERIALIZABLE"
            ),
            "production_validatable_to_transaction_ready": ratio(
                "TRANSACTION_READY", "PRODUCTION_VALIDATABLE"
            ),
        },
        "attrition": attrition,
        "largest_attrition": max(attrition, key=attrition.get),  # type: ignore[arg-type]
        "first_failures": dict(
            sorted(Counter(str(item["failure_layer"]) for item in records).items())
        ),
        "by_model": _aggregate(records, "model_profile"),
        "by_task_class": _aggregate(records, "operation_class"),
        "cell_outcomes": {
            "transaction_applied": sum(
                bool(item["transaction_applied"]) for item in cells
            ),
            "verification_pass": sum(bool(item["verification_pass"]) for item in cells),
            "semantic_pass": sum(bool(item["semantic_pass"]) for item in cells),
            "metadata_exactly_once": sum(
                bool(item["proposal_metadata_exactly_once"]) for item in cells
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    definitions = tasks()
    paths = tuple(
        args.checkpoint_dir / "cells" / f"{profile}-seed42-{task.task_id}.json"
        for profile in PROFILES
        for task in definitions
    )
    missing = tuple(path.name for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"A71 matrix incomplete: {missing}")
    cells = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    summary = summarize(cells)
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A71 summary is not source-free")
    output = args.checkpoint_dir / "summary.json"
    atomic_checkpoint(output, summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

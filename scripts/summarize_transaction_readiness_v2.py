"""Summarize committed source-free A72 cells without replay or reconstruction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.transaction_readiness_v1.protocol import STAGES
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.suite import PROFILES, SUITE, tasks
from forge.evaluation.mutation_ready import atomic_checkpoint


def _records(cells: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    records = []
    for cell in cells:
        proposals = cell["proposals"]
        assert isinstance(proposals, list)
        if not proposals:
            model_output = bool(cell["model_output_received"])
            records.append(
                {
                    "model_profile": cell["model_profile"],
                    "operation_class": cell["operation_class"],
                    "transitions": tuple(
                        (stage.value, model_output if index == 0 else False)
                        for index, stage in enumerate(STAGES)
                    ),
                    "failure_layer": "MODEL_PROTOCOL_FAILURE"
                    if model_output
                    else "NO_MODEL_OUTPUT",
                    "transaction_failure_subtype": None,
                    "observation_metadata_classification": None,
                }
            )
        else:
            records.extend(
                {
                    "model_profile": cell["model_profile"],
                    "operation_class": cell["operation_class"],
                    "transitions": proposal["transitions"],
                    "failure_layer": proposal["failure_layer"],
                    "transaction_failure_subtype": proposal[
                        "transaction_failure_subtype"
                    ],
                    "observation_metadata_classification": proposal[
                        "observation_metadata_classification"
                    ],
                }
                for proposal in proposals
            )
    return tuple(records)


def _aggregate(records: tuple[dict[str, object], ...], key: str) -> dict[str, object]:
    result = {}
    for name in sorted({str(item[key]) for item in records}):
        selected = tuple(item for item in records if item[key] == name)
        result[name] = {
            "records": len(selected),
            "transitions": {
                stage.value: sum(
                    bool(dict(item["transitions"])[stage.value]) for item in selected
                )  # type: ignore[arg-type]
                for stage in STAGES
            },
            "first_failures": dict(
                sorted(Counter(str(item["failure_layer"]) for item in selected).items())
            ),
        }
    return result


def summarize(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    records = _records(cells)
    transitions = {
        stage.value: sum(
            bool(dict(item["transitions"])[stage.value]) for item in records
        )  # type: ignore[arg-type]
        for stage in STAGES
    }
    ordered = tuple(stage.value for stage in STAGES)
    attrition = {
        f"{left}->{right}": transitions[left] - transitions[right]
        for left, right in zip(ordered, ordered[1:], strict=False)
    }
    conversions = {
        f"{left.lower()}_to_{right.lower()}": (
            round(transitions[right] / transitions[left], 4)
            if transitions[left]
            else None
        )
        for left, right in zip(ordered, ordered[1:], strict=False)
    }
    observations = Counter(
        str(item["observation_metadata_classification"])
        for item in records
        if item["observation_metadata_classification"] is not None
    )
    summary = {
        "suite": SUITE,
        "schema_version": 2,
        "cells": len(cells),
        "proposal_records_including_no_output_cells": len(records),
        "transitions": transitions,
        "conversions": conversions,
        "attrition": attrition,
        "largest_attrition": max(attrition, key=attrition.get),  # type: ignore[arg-type]
        "first_failures": dict(
            sorted(Counter(str(item["failure_layer"]) for item in records).items())
        ),
        "transaction_failure_subtypes": dict(
            sorted(
                Counter(
                    str(item["transaction_failure_subtype"])
                    for item in records
                    if item["transaction_failure_subtype"]
                ).items()
            )
        ),
        "observation_classifications": dict(sorted(observations.items())),
        "observation_metadata_incomplete": observations[
            "OBSERVATION_METADATA_INCOMPLETE"
        ],
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
    if summary["observation_metadata_incomplete"]:
        raise RuntimeError("A72 contains OBSERVATION_METADATA_INCOMPLETE")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = tuple(
        args.checkpoint_dir / "cells" / f"{profile}-seed42-{task.task_id}.json"
        for profile in PROFILES
        for task in tasks()
    )
    missing = tuple(path.name for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"A72 matrix incomplete: {missing}")
    cells = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    summary = summarize(cells)
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A72 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Classify pre-correction A71 observation coverage without rewriting cells."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from forge.evaluation.mutation_ready import (
    ObservationMetadataClassification,
    atomic_checkpoint,
    classify_observation_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    cells = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.checkpoint_dir / "cells").glob("*.json"))
    )
    if not cells:
        raise RuntimeError("no A71 cells found")
    counts: Counter[str] = Counter()
    proposal_records = 0
    for cell in cells:
        for proposal in cell.get("proposals", ()):
            proposal_records += 1
            metadata = proposal.get("mutation_ready_metadata")
            classification = (
                classify_observation_metadata(metadata)
                if isinstance(metadata, dict)
                else ObservationMetadataClassification.OBSERVATION_METADATA_INCOMPLETE
            )
            counts[classification.value] += 1
    payload = {
        "suite": "transaction-readiness-v1-pre-correction-observation-audit",
        "cells": len(cells),
        "proposal_records": proposal_records,
        "classifications": dict(sorted(counts.items())),
        "historical_cells_rewritten": False,
        "post_hoc_metadata_reconstructed": False,
    }
    atomic_checkpoint(args.checkpoint_dir / "observation-coverage-audit.json", payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

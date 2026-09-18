"""Run deterministic A48 verification alignment and write its source-free artifact."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    load_realistic_semantic_run,
    map_model_mutations,
    run_verification_alignment_v1,
    write_verification_alignment_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, action="append", default=[])
    parser.add_argument("--workspace-root", type=Path)
    args = parser.parse_args()

    definitions = realistic_semantic_tasks((42,))
    integrity = validate_integrity(definitions)
    if not all(
        not item.baseline_oracle_pass
        and item.reference_oracle_pass
        and not item.wrong_mutation_oracle_pass
        and item.semantic_score_eligible
        for item in integrity
    ):
        raise RuntimeError("frozen realistic-semantic-v1 integrity changed")

    historical = tuple(load_realistic_semantic_run(path) for path in args.model_result)
    mutations = map_model_mutations(historical)
    if args.workspace_root is not None:
        run = run_verification_alignment_v1(
            args.workspace_root.resolve(),
            REPOSITORY,
            definitions,
            model_mutations=mutations,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="forge-a48-") as name:
            run = run_verification_alignment_v1(
                Path(name), REPOSITORY, definitions, model_mutations=mutations
            )
    write_verification_alignment_json(run, args.output)
    print(f"tasks={len(run.tasks)} observations={len(run.tasks) * 3}")
    print(f"confusion={run.confusion_counts}")
    print(f"model_mutations={len(run.model_mutations)}")
    print(f"canonical_unchanged={run.canonical_unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

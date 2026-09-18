"""Merge per-model A47 repair-effectiveness artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.evaluation import (
    build_repair_case_corpus,
    build_repair_effectiveness_run,
    load_realistic_semantic_run,
    load_repair_effectiveness_run,
    write_repair_effectiveness_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--corpus-result", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = tuple(load_repair_effectiveness_run(path) for path in args.inputs)
    identities = {run.repository_identity for run in runs}
    if len(identities) != 1:
        raise RuntimeError("partial A47 artifacts use different repositories")
    cases = (
        build_repair_case_corpus(
            tuple(load_realistic_semantic_run(path) for path in args.corpus_result)
        )
        if args.corpus_result
        else tuple({case.case_id: case for run in runs for case in run.cases}.values())
    )
    trials = tuple(trial for run in runs for trial in run.trials)
    merged = build_repair_effectiveness_run(
        repository_identity=next(iter(identities)),
        cases=cases,
        trials=trials,
        canonical_unchanged=all(run.canonical_unchanged for run in runs),
    )
    write_repair_effectiveness_json(merged, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

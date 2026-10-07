"""Freeze A73 corpus and establish deterministic baseline health."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import (
    BUILD,
    CONFIGURE,
    REPOSITORY,
    TEST,
    baseline_workspace,
)
from benchmarks.verification_attribution_v1.suite import (
    CASES,
    SUITE,
    corpus_identity,
    task_map,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evaluation.realworld import EvaluationOutcome, run_oracle


def _health(definition):  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory(prefix=f"forge-a73-{definition.task_id}-") as name:
        root = Path(name).resolve()
        baseline = baseline_workspace(definition, root / "baseline")
        baseline_oracle = run_oracle(
            baseline, definition.production_task.oracle_commands
        )
        baseline_verification = run_oracle(baseline, (CONFIGURE, BUILD, TEST))
        reference = root / "reference"
        shutil.copytree(
            baseline, reference, ignore=shutil.ignore_patterns("build", "__pycache__")
        )
        for relative in definition.production_task.expected_changed_paths:
            target = reference / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPOSITORY / relative).read_bytes())
        reference_oracle = run_oracle(
            reference, definition.production_task.oracle_commands
        )
        reference_verification = run_oracle(reference, (CONFIGURE, BUILD, TEST))
    return {
        "baseline_oracle": baseline_oracle.value,
        "baseline_full_verification": baseline_verification.value,
        "reference_oracle": reference_oracle.value,
        "reference_full_verification": reference_verification.value,
        "eligible": (
            baseline_oracle is EvaluationOutcome.FAIL
            and reference_oracle is EvaluationOutcome.PASS
            and reference_verification is EvaluationOutcome.PASS
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    definitions = task_map()
    selected = tuple(
        definitions[task_id]
        for task_id in dict.fromkeys(case.task_id for case in CASES)
    )
    baselines = {definition.task_id: _health(definition) for definition in selected}
    if not all(item["eligible"] for item in baselines.values()):
        raise RuntimeError("A73 baseline invalid")
    atomic_checkpoint(
        args.output,
        {
            "suite": SUITE,
            "corpus_identity": corpus_identity(),
            "cases": [case.case_id for case in CASES],
            "baselines": baselines,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

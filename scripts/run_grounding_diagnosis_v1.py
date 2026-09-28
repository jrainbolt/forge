"""Run the bounded A63 grounding and post-grounding diagnostic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.grounding_diagnosis_v1.coding import run_coding_diagnostic
from benchmarks.grounding_diagnosis_v1.grounding import diagnose_grounding
from benchmarks.grounding_diagnosis_v1.suite import (
    CODING_CASES,
    GROUNDING_CASES,
    SUITE,
    VERSION,
)
from benchmarks.real_repository_pilot_v1.suite import repository_identity
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from benchmarks.real_repository_pilot_v2.suite import tasks
from forge.models import default_backend_registry, load_model_catalog


def _identity() -> str:
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "grounding": [asdict(item) for item in GROUNDING_CASES],
        "coding": [asdict(item) for item in CODING_CASES],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    canonical = args.repository.resolve(strict=True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    before = repository_identity(canonical)
    identity = _identity()
    definitions = {item.task_id: item for item in tasks()}

    for case in GROUNDING_CASES:
        path = root / "grounding" / f"{case.case_id}.json"
        expected = {"case_id": case.case_id, "profile": case.profile}
        if read_checkpoint(path, expected) is not None:
            print(f"skip committed {case.case_id}", flush=True)
            continue
        result = diagnose_grounding(case, definitions[case.task_id], canonical)
        atomic_json(path, asdict(result))
        print(
            f"checkpointed {case.case_id}: {result.stopping_cause} "
            f"G0={result.g0_sufficient} G1={result.g1_sufficient} "
            f"G2={result.g2_sufficient}",
            flush=True,
        )

    catalog = load_model_catalog(args.config, default_backend_registry())
    for profile_name in ("qwen-small", "qwen-large", "codestral-22b"):
        selected = tuple(case for case in CODING_CASES if case.profile == profile_name)
        pending = []
        for case in selected:
            path = root / "coding" / f"{case.case_id}.json"
            expected = {"case_id": case.case_id, "profile": case.profile}
            if read_checkpoint(path, expected) is None:
                pending.append(case)
            else:
                print(f"skip committed {case.case_id}", flush=True)
        if not pending:
            continue
        profile = catalog.profile(profile_name)
        model = catalog.create(profile_name)
        try:
            for case in pending:
                result = run_coding_diagnostic(
                    case,
                    definitions[case.task_id],
                    canonical,
                    model,
                    profile.mutation_representation,
                )
                if repository_identity(canonical) != before:
                    raise RuntimeError("canonical Foundation changed during A63")
                atomic_json(root / "coding" / f"{case.case_id}.json", asdict(result))
                print(
                    f"checkpointed {case.case_id}: {result.interpretation} "
                    f"C0={result.c0.failure} C1={result.c1.failure}",
                    flush=True,
                )
        finally:
            model.close()

    grounding = tuple(
        json.loads(path.read_text())
        for path in sorted((root / "grounding").glob("*.json"))
    )
    coding = tuple(
        json.loads(path.read_text())
        for path in sorted((root / "coding").glob("*.json"))
    )
    summary = {
        "suite": SUITE,
        "version": VERSION,
        "identity": identity,
        "canonical_pre": before,
        "canonical_post": repository_identity(canonical),
        "canonical_unchanged": repository_identity(canonical) == before,
        "grounding_cases": len(grounding),
        "coding_cases": len(coding),
        "stopping_causes": {
            cause: sum(item["stopping_cause"] == cause for item in grounding)
            for cause in sorted({item["stopping_cause"] for item in grounding})
        },
        "g0_success": sum(bool(item["g0_sufficient"]) for item in grounding),
        "g1_success": sum(bool(item["g1_sufficient"]) for item in grounding),
        "g2_success": sum(bool(item["g2_sufficient"]) for item in grounding),
        "interpretations": {
            value: sum(item["interpretation"] == value for item in coding)
            for value in sorted({item["interpretation"] for item in coding})
        },
    }
    summary_path = root / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text()) != summary:
            raise RuntimeError("A63 summary differs from committed summary")
    else:
        atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

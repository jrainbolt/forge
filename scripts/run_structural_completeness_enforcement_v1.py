"""Run the frozen A85 qwen-large ENFORCE validation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell
from benchmarks.production_detectable_completeness_v1.checker import (
    AMBIGUOUS,
    COMPLETE,
    INCOMPLETE,
    NOT_CHECKABLE,
    evaluate_detailed,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.structural_completeness_enforcement_v1.suite import (
    CASES,
    CONTROL_CASE_IDS,
    RUN_ID,
    corpus_identity,
    validate_corpus,
)
from benchmarks.structural_completeness_readiness_v1.suite import (
    REPRESENTATION,
    tasks,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog
from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessEvaluation,
    StructuralCompletenessMode,
    evaluate_structural_completeness,
)


def _evaluator(checks, capture):  # type: ignore[no-untyped-def]
    def evaluate(workspace: Path) -> StructuralCompletenessEvaluation:
        started = time.perf_counter()
        details = tuple(
            (check, evaluate_detailed(check, workspace)) for check in checks
        )
        failed = tuple(
            check.check_id for check, detail in details if detail.status == INCOMPLETE
        )
        operational = tuple(
            detail.operational_failure
            for _, detail in details
            if detail.operational_failure is not None
        )
        if failed:
            decision = StructuralCompletenessDecision.FAIL
        elif any(detail.status in {NOT_CHECKABLE, AMBIGUOUS} for _, detail in details):
            decision = StructuralCompletenessDecision.PARTIAL
        elif details and all(detail.status == COMPLETE for _, detail in details):
            decision = StructuralCompletenessDecision.PASS
        else:
            decision = StructuralCompletenessDecision.NOT_APPLICABLE
        result = StructuralCompletenessEvaluation(
            decision,
            failed,
            operational,
            time.perf_counter() - started,
            sum(
                check.kind.value == "REQUIRED_COMPONENT_ROLE"
                and check.source_path.endswith(".c")
                for check, _ in details
            ),
        )
        capture.append(result)
        return result

    return evaluate


def _evaluation_payload(result):  # type: ignore[no-untyped-def]
    if result is None:
        return None
    payload = asdict(result)
    payload["decision"] = result.decision.value
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A85 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A85 requires the unchanged qwen-large coding default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A85 model configuration changed")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    repository_identity = source_state_identity(REPOSITORY)
    definitions = {item.task_id: item for item in tasks()}
    manifest = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cases": [case.case_id for case in CASES] + list(CONTROL_CASE_IDS),
        "mode": "enforce",
        "default_mode": "shadow",
        "blocking_authority": "PROVEN_STRUCTURAL_FAIL_ONLY",
    }
    atomic_checkpoint(args.output_dir / "manifest.json", manifest)
    model = catalog.create(selected)
    cells = []
    try:
        for case in CASES:
            captured = []
            before = source_state_identity(REPOSITORY)
            cell = run_cell(
                definitions[case.case_id],
                selected,
                model,
                artifact=artifact,
                artifact_size=inventory["file_size"],
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus_identity(),
                representation=REPRESENTATION,
                run_identity=RUN_ID,
                structural_completeness_mode=StructuralCompletenessMode.ENFORCE,
                structural_completeness_evaluator=_evaluator(case.checks, captured),
            )
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical repository changed")
            observation = captured[0] if captured else None
            payload = {
                "case_id": case.case_id,
                "truth": case.truth,
                "transaction_applied": cell.proposal.get("transaction_applied", False),
                "verification_pass": cell.proposal.get("verification_pass", False),
                "semantic_pass": cell.primary_semantic_pass,
                "final_status": cell.workflow_trace.get("final_status"),
                "structural": _evaluation_payload(observation),
                "generation_latency_seconds": cell.median_generation_latency_seconds,
            }
            atomic_checkpoint(args.output_dir / f"{case.case_id}.json", payload)
            cells.append(payload)
            print(
                json.dumps(
                    {"case": case.case_id, "structural": payload["structural"]},
                    default=str,
                ),
                flush=True,
            )
    finally:
        model.close()
    for control in CONTROL_CASE_IDS:
        decision = StructuralCompletenessDecision(control.lower())
        record = evaluate_structural_completeness(
            StructuralCompletenessMode.ENFORCE,
            REPOSITORY,
            lambda _workspace, value=decision: StructuralCompletenessEvaluation(value),
        )
        payload = {"case_id": control, "structural": asdict(record), "continued": True}
        atomic_checkpoint(args.output_dir / f"{control}.json", payload)
        cells.append(payload)
    observations = [
        cell["structural"] for cell in cells[: len(CASES)] if cell["structural"]
    ]
    latencies = [item["latency_seconds"] for item in observations]
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "model_cells": len(CASES),
        "proven_rejections": [
            cell["case_id"]
            for cell in cells[: len(CASES)]
            if cell["final_status"] == "structural_completeness_rejected"
        ],
        "valid_continued": [
            cell["case_id"]
            for cell in cells[: len(CASES)]
            if cell["structural"] and cell["structural"]["decision"] == "pass"
        ],
        "semantic_negative_continued": [
            cell["case_id"]
            for cell in cells[: len(CASES)]
            if cell["truth"] == "SEMANTIC_NEGATIVE_STRUCTURAL_PASS"
            and cell["final_status"] != "structural_completeness_rejected"
        ],
        "median_checker_latency_seconds": statistics.median(latencies),
        "worst_checker_latency_seconds": max(latencies),
        "subprocess_count": sum(item["subprocess_count"] for item in observations),
        "canonical_unchanged": True,
        "default_mode": "shadow",
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

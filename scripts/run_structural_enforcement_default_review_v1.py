"""Run the frozen A86 corpus through the implicit SHADOW production default."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
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
from benchmarks.structural_completeness_readiness_v1.suite import (
    REPRESENTATION,
    tasks,
)
from benchmarks.structural_enforcement_default_review_v1.suite import (
    CASES,
    RUN_ID,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog
from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessEvaluation,
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


def _payload(result):  # type: ignore[no-untyped-def]
    value = asdict(result)
    value["decision"] = result.decision.value
    return value


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
        raise RuntimeError("A86 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A86 requires qwen-large")
    if catalog.project_commands.structural_completeness_mode.value != "shadow":
        raise RuntimeError("A86 primary run requires the unchanged SHADOW default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A86 model configuration changed")
    repository_identity = source_state_identity(REPOSITORY)
    definitions = {item.task_id: item for item in tasks()}
    atomic_checkpoint(
        args.output_dir / "manifest.json",
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "cases": [case.case_id for case in CASES],
            "mode": "shadow",
            "default_unchanged_before_review": True,
            "hidden_oracle_data": False,
        },
    )
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
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
                structural_completeness_evaluator=_evaluator(case.checks, captured),
            )
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical repository changed")
            observation = captured[0] if captured else None
            structural = (
                _payload(observation)
                if observation is not None
                else {
                    "decision": "not_applicable",
                    "failed_check_ids": (),
                    "operational_failures": (),
                    "latency_seconds": 0.0,
                    "subprocess_count": 0,
                }
            )
            payload = {
                "case_id": case.case_id,
                "truth": case.truth,
                "evaluator_available": True,
                "ordinary_implicit_evaluator_available": False,
                "obligation_count": len(case.checks),
                "check_kinds": sorted({check.kind.value for check in case.checks}),
                "operation_class": cell.operation_class,
                "structural": structural,
                "transaction_applied": cell.proposal.get("transaction_applied", False),
                "verification_pass": cell.proposal.get("verification_pass", False),
                "semantic_pass": cell.primary_semantic_pass,
                "terminal_outcome": cell.workflow_trace.get("final_status"),
                "generation_latency_seconds": cell.median_generation_latency_seconds,
            }
            atomic_checkpoint(args.output_dir / f"{case.case_id}.json", payload)
            cells.append(payload)
            print(
                json.dumps({"case": case.case_id, "decision": structural["decision"]}),
                flush=True,
            )
    finally:
        model.close()
    applicable = [cell for cell in cells if cell["obligation_count"]]
    decisions = Counter(cell["structural"]["decision"] for cell in cells)
    failures = [cell for cell in cells if cell["structural"]["decision"] == "fail"]
    established_false = [
        cell
        for cell in failures
        if cell["verification_pass"] is True and cell["semantic_pass"] is True
    ]
    known = [
        cell
        for cell in cells
        if cell["truth"]
        in {"KNOWN_STRUCTURAL_MISS", "SEMANTIC_NEGATIVE_STRUCTURAL_MISS"}
    ]
    missed = [cell for cell in known if cell["structural"]["decision"] != "fail"]
    semantic_pass_fail = [
        cell
        for cell in cells
        if cell["structural"]["decision"] == "pass" and cell["semantic_pass"] is False
    ]
    latencies = [cell["structural"]["latency_seconds"] for cell in cells]
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    operational = Counter(
        item for cell in cells for item in cell["structural"]["operational_failures"]
    )
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "task_level_obligation_availability": len(applicable) / len(cells),
        "ordinary_implicit_evaluator_availability": 0.0,
        "not_applicable_rate": decisions["not_applicable"] / len(cells),
        "decisions": dict(decisions),
        "check_kinds": dict(
            Counter(kind for cell in cells for kind in cell["check_kinds"])
        ),
        "established_false_fails": [cell["case_id"] for cell in established_false],
        "known_defects": [cell["case_id"] for cell in known],
        "known_defects_missed": [cell["case_id"] for cell in missed],
        "structural_pass_semantic_fail": [
            cell["case_id"] for cell in semantic_pass_fail
        ],
        "offline_blocks": [cell["case_id"] for cell in failures],
        "offline_justified_blocks": [
            cell["case_id"] for cell in failures if cell not in established_false
        ],
        "correct_candidates_blocked": len(established_false),
        "verification_failures_prevented_early": sum(
            cell["verification_pass"] is False for cell in failures
        ),
        "semantic_failures_unaffected": len(semantic_pass_fail),
        "no_enforcement_benefit": len(cells) - len(failures),
        "operational_failures": dict(operational),
        "partial_rate": decisions["partial"] / len(cells),
        "median_checker_latency_seconds": statistics.median(latencies),
        "p95_checker_latency_seconds": p95,
        "worst_checker_latency_seconds": max(latencies),
        "subprocess_count": sum(
            cell["structural"]["subprocess_count"] for cell in cells
        ),
        "median_generation_latency_seconds": statistics.median(
            cell["generation_latency_seconds"]
            for cell in cells
            if cell["generation_latency_seconds"] is not None
        ),
        "canonical_unchanged": True,
        "default_decision": "DEFAULT_REMAINS_SHADOW_INSUFFICIENT_COVERAGE",
        "decision_reason": (
            "ordinary production workflows do not derive or supply structural "
            "obligations"
        ),
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

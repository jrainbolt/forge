"""Run the frozen A88 autonomous-derivation expansion matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.structural_completeness_readiness_v1.suite import (
    REPRESENTATION,
    tasks,
)
from benchmarks.structural_obligation_derivation_expansion_v1.suite import (
    CASES,
    MIN_TASK_CLASSES,
    MIN_WORKFLOW_COVERAGE,
    RUN_ID,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.realworld import (
    apply_task_absences,
    apply_task_setup,
    copy_repository,
)
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog
from forge.structural_obligations import derive_structural_obligations


def _hash(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


def _truth_key(check) -> tuple[object, ...]:  # type: ignore[no-untyped-def]
    return (
        check.kind.value,
        _hash(check.source_path),
        _hash(check.target_path),
        _hash(check.symbol),
        _hash(check.caller),
    )


def _observed_key(item: dict[str, object]) -> tuple[object, ...]:
    return (
        item.get("kind"),
        item.get("source_identity"),
        item.get("target_identity"),
        item.get("symbol_identity"),
        item.get("caller_identity"),
    )


def _determinism(definition) -> bool:  # type: ignore[no-untyped-def]
    task = definition.production_task
    with tempfile.TemporaryDirectory(prefix="forge-a88-determinism-") as temporary:
        workspace = copy_repository(REPOSITORY, Path(temporary) / "workspace")
        apply_task_setup(workspace, task.setup)
        apply_task_absences(workspace, task.setup_absent_paths)
        authority = tuple(
            sorted({*task.required_candidate_paths, *task.create_candidate_paths})
        )
        first = derive_structural_obligations(
            task.prompt, workspace, mutation_authority=authority
        )
        second = derive_structural_obligations(
            task.prompt, workspace, mutation_authority=authority
        )
    return (
        first.identity == second.identity
        and tuple(item.kind for item in first.obligations)
        == tuple(item.kind for item in second.obligations)
        and tuple(item.provenance_identity for item in first.obligations)
        == tuple(item.provenance_identity for item in second.obligations)
    )


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
        raise RuntimeError("A88 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A88 requires unchanged qwen-large coding default")
    if catalog.project_commands.structural_completeness_mode.value != "shadow":
        raise RuntimeError("A88 requires unchanged SHADOW structural default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A88 model configuration changed")
    repository_identity = source_state_identity(REPOSITORY)
    definitions = {item.task_id: item for item in tasks()}
    manifest = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cases": [case.case_id for case in CASES],
        "minimum_workflow_coverage": MIN_WORKFLOW_COVERAGE,
        "minimum_task_classes": MIN_TASK_CLASSES,
        "mode": "shadow",
        "evaluator_obligation_injection": False,
        "default_unchanged": True,
        "truth_frozen_before_execution": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    existing = resume_checkpoint(manifest_path)
    if existing is None:
        atomic_checkpoint(manifest_path, manifest)
    elif existing != manifest:
        raise RuntimeError("A88 manifest identity mismatch")
    deterministic = {
        case.case_id: _determinism(definitions[case.case_id]) for case in CASES[:10]
    }
    if not all(deterministic.values()):
        raise RuntimeError("A88 deterministic derivation requirement failed")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    model = catalog.create(selected)
    cells = []
    try:
        for case in CASES:
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
            )
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical repository changed")
            trace = cell.workflow_trace
            metadata = tuple(trace.get("structural_obligation_metadata", ()))
            derived = {_observed_key(dict(item)) for item in metadata}
            truth = {_truth_key(check) for check in case.checks}
            payload = {
                "case_id": case.case_id,
                "truth": case.truth,
                "operation_class": cell.operation_class,
                "language": definitions[case.case_id].language,
                "obligation_count": len(derived),
                "obligation_kinds": sorted(key[0] for key in derived),
                "obligation_set_identity": trace.get(
                    "structural_obligation_set_identity"
                ),
                "true_obligations": len(derived & truth),
                "required_obligations_missed": len(truth - derived),
                "spurious_obligations": len(derived - truth),
                "shadow_decision": trace.get("structural_decision", "not_run"),
                "failed_check_ids": trace.get("structural_failed_check_ids", ()),
                "operational_failures": trace.get(
                    "structural_operational_failures", ()
                ),
                "derivation_latency_seconds": trace.get(
                    "structural_derivation_latency_seconds", 0.0
                ),
                "source_operations": trace.get("structural_source_operations", 0),
                "check_latency_seconds": trace.get(
                    "structural_check_latency_seconds", 0.0
                ),
                "subprocess_count": trace.get("structural_subprocess_count", 0),
                "verification_pass": cell.proposal.get("verification_pass", False),
                "semantic_pass": cell.primary_semantic_pass,
                "transaction_applied": cell.proposal.get("transaction_applied", False),
                "terminal_outcome": trace.get("final_status"),
            }
            atomic_checkpoint(args.output_dir / f"{case.case_id}.json", payload)
            cells.append(payload)
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "obligations": len(derived),
                        "decision": payload["shadow_decision"],
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    covered = [cell for cell in cells if cell["true_obligations"]]
    classes = {cell["operation_class"] for cell in covered}
    spurious = sum(int(cell["spurious_obligations"]) for cell in cells)
    false_fails = [
        cell["case_id"]
        for cell in cells
        if cell["shadow_decision"] == "fail"
        and cell["verification_pass"]
        and cell["semantic_pass"]
    ]
    coverage = len(covered) / len(cells)
    ready = (
        not spurious
        and not false_fails
        and coverage >= MIN_WORKFLOW_COVERAGE
        and len(classes) >= MIN_TASK_CLASSES
    )
    classification = (
        "DERIVATION_FALSE_POSITIVE_RISK"
        if spurious or false_fails
        else "AUTONOMOUS_COVERAGE_READY_FOR_ENFORCEMENT_REVIEW"
        if ready
        else "AUTONOMOUS_COVERAGE_INSUFFICIENT"
    )
    derivation_latencies = [float(cell["derivation_latency_seconds"]) for cell in cells]
    check_latencies = [float(cell["check_latency_seconds"]) for cell in cells]
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "covered_workflows": len(covered),
        "autonomous_coverage": coverage,
        "covered_task_classes": sorted(classes),
        "true_obligations": sum(int(cell["true_obligations"]) for cell in cells),
        "required_obligations_missed": sum(
            int(cell["required_obligations_missed"]) for cell in cells
        ),
        "spurious_obligations": spurious,
        "derivation_miss_count": sum(
            bool(cell["required_obligations_missed"]) for cell in cells
        ),
        "not_applicable_count": sum(
            cell["shadow_decision"] == "not_applicable" for cell in cells
        ),
        "obligation_kinds": dict(
            Counter(kind for cell in cells for kind in cell["obligation_kinds"])
        ),
        "coverage_by_task_class": dict(
            Counter(cell["operation_class"] for cell in covered)
        ),
        "shadow_decisions": dict(Counter(cell["shadow_decision"] for cell in cells)),
        "established_false_structural_fails": false_fails,
        "h12_diagnosis": {
            "ResponseStore": "AMBIGUOUS_RELATIONSHIP",
            "parse_enabled": "EVIDENCE_NOT_AVAILABLE",
            "parse_timeout": "EVIDENCE_NOT_AVAILABLE",
            "config_component_role": "EVIDENCE_NOT_AVAILABLE",
        },
        "determinism_repeats": deterministic,
        "operational_failures": dict(
            Counter(item for cell in cells for item in cell["operational_failures"])
        ),
        "median_derivation_latency_seconds": statistics.median(derivation_latencies),
        "worst_derivation_latency_seconds": max(derivation_latencies),
        "median_check_latency_seconds": statistics.median(check_latencies),
        "worst_check_latency_seconds": max(check_latencies),
        "source_operations": sum(int(cell["source_operations"]) for cell in cells),
        "subprocess_count": sum(int(cell["subprocess_count"]) for cell in cells),
        "canonical_unchanged": True,
        "threshold": {
            "minimum_workflow_coverage": MIN_WORKFLOW_COVERAGE,
            "minimum_task_classes": MIN_TASK_CLASSES,
            "met": ready,
        },
        "readiness_classification": classification,
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

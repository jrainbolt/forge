"""Run A87 through ordinary qwen-large coding with autonomous SHADOW checks."""

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
from benchmarks.structural_obligation_derivation_v1.suite import (
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


def _key(item) -> tuple[str, str, str | None, str | None]:  # type: ignore[no-untyped-def]
    return (item.kind.value, item.source_path, item.target_path, item.symbol)


def _truth_key(item) -> tuple[str, str, str | None, str | None]:  # type: ignore[no-untyped-def]
    return (item.kind.value, item.source_path, item.target_path, item.symbol)


def _frozen_derivation(definition, registered_truth):  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory(prefix="forge-a87-derive-") as temporary:
        workspace = copy_repository(REPOSITORY, Path(temporary) / "workspace")
        task = definition.production_task
        apply_task_setup(workspace, task.setup)
        apply_task_absences(workspace, task.setup_absent_paths)
        result = derive_structural_obligations(
            task.prompt,
            workspace,
            mutation_authority=tuple(
                sorted({*task.required_candidate_paths, *task.create_candidate_paths})
            ),
        )
    truth = {_truth_key(item) for item in registered_truth}
    derived = {_key(item) for item in result.obligations}
    return result, len(derived & truth), len(truth - derived), len(derived - truth)


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
        raise RuntimeError("A87 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A87 requires unchanged qwen-large coding default")
    if catalog.project_commands.structural_completeness_mode.value != "shadow":
        raise RuntimeError("A87 requires unchanged SHADOW structural default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A87 model configuration changed")
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
    }
    manifest_path = args.output_dir / "manifest.json"
    existing_manifest = resume_checkpoint(manifest_path)
    if existing_manifest is None:
        atomic_checkpoint(manifest_path, manifest)
    elif existing_manifest != manifest:
        raise RuntimeError("A87 manifest identity mismatch")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    model = catalog.create(selected)
    cells = []
    try:
        for case in CASES:
            definition = definitions[case.case_id]
            frozen, true_count, missed_count, spurious_count = _frozen_derivation(
                definition, case.checks
            )
            before = source_state_identity(REPOSITORY)
            cell = run_cell(
                definition,
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
            observed_count = int(trace.get("structural_obligation_count", 0))
            observed_identity = trace.get("structural_obligation_set_identity")
            if (
                observed_count != len(frozen.obligations)
                or observed_identity != frozen.identity
            ):
                raise RuntimeError(f"A87 derivation mismatch in {case.case_id}")
            payload = {
                "case_id": case.case_id,
                "truth": case.truth,
                "operation_class": cell.operation_class,
                "language": definition.language,
                "obligation_count": observed_count,
                "obligation_kinds": trace.get("structural_obligation_kinds", ()),
                "obligation_set_identity": observed_identity,
                "true_obligations": true_count,
                "required_obligations_missed": missed_count,
                "spurious_obligations": spurious_count,
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
                        "obligations": observed_count,
                        "decision": payload["shadow_decision"],
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    covered = [cell for cell in cells if cell["obligation_count"]]
    classes = {cell["operation_class"] for cell in covered}
    spurious = sum(int(cell["spurious_obligations"]) for cell in cells)
    false_fails = [
        cell["case_id"]
        for cell in cells
        if cell["shadow_decision"] == "fail"
        and cell["verification_pass"]
        and cell["semantic_pass"]
    ]
    known = {"H11", "C11", "H12"}
    ready = (
        not spurious
        and not false_fails
        and len(covered) / len(cells) >= MIN_WORKFLOW_COVERAGE
        and len(classes) >= MIN_TASK_CLASSES
    )
    classification = (
        "DERIVATION_FALSE_POSITIVE_RISK"
        if spurious or false_fails
        else "AUTONOMOUS_COVERAGE_READY_FOR_A88"
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
        "autonomous_coverage": len(covered) / len(cells),
        "covered_task_classes": sorted(classes),
        "true_obligations": sum(int(cell["true_obligations"]) for cell in cells),
        "required_obligations_missed": sum(
            int(cell["required_obligations_missed"]) for cell in cells
        ),
        "spurious_obligations": spurious,
        "zero_obligation_tasks": [
            cell["case_id"] for cell in cells if not cell["obligation_count"]
        ],
        "obligation_kinds": dict(
            Counter(kind for cell in cells for kind in cell["obligation_kinds"])
        ),
        "shadow_decisions": dict(Counter(cell["shadow_decision"] for cell in cells)),
        "established_false_structural_fails": false_fails,
        "known_defects": {
            cell["case_id"]: {
                "derived": bool(cell["obligation_count"]),
                "decision": cell["shadow_decision"],
            }
            for cell in cells
            if cell["case_id"] in known
        },
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

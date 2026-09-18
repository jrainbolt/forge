"""Run one A49 real-model synthesis matrix with per-task checkpoints."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION,
    ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION,
    ACCEPTANCE_TEST_SYNTHESIS_V1,
    AcceptanceTestSynthesisRun,
    SynthesisCondition,
    load_acceptance_test_synthesis_json,
    run_acceptance_test_synthesis_v1,
    write_acceptance_test_synthesis_json,
)
from forge.models import default_backend_registry, load_model_catalog


def _full_signals(path: Path) -> dict[str, tuple[str, str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("suite") != "verification-alignment-v1":
        raise ValueError("A49 requires the A48 verification-alignment artifact")
    return {
        item["task_id"]: (
            item["baseline"]["final_verification_result"],
            item["reference"]["final_verification_result"],
            item["wrong"]["final_verification_result"],
        )
        for item in payload["tasks"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--a48-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", default=[], help="defaults to R01,R02,R05-R08"
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=[item.value for item in SynthesisCondition],
        default=[],
    )
    parser.add_argument("--preserve-unselected", action="store_true")
    args = parser.parse_args()

    selected_ids = tuple(args.task) or ("R01", "R02", "R05", "R06", "R07", "R08")
    definitions = tuple(
        item
        for item in realistic_semantic_tasks((42,))
        if item.metadata.task_id in selected_ids
    )
    if {item.metadata.task_id for item in definitions} != set(selected_ids):
        raise ValueError("unknown or duplicate task selection")
    integrity = validate_integrity(definitions)
    if not all(
        not item.baseline_oracle_pass
        and item.reference_oracle_pass
        and not item.wrong_mutation_oracle_pass
        for item in integrity
    ):
        raise RuntimeError("frozen benchmark integrity changed")

    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("acceptance-test-synthesis-v1 requires context 8192")
    signals = _full_signals(args.a48_result)
    conditions = tuple(SynthesisCondition(item) for item in args.condition) or tuple(
        SynthesisCondition
    )
    accumulated = []
    if args.preserve_unselected and args.output.is_file():
        existing = load_acceptance_test_synthesis_json(args.output)
        if existing.model_profile != args.model:
            raise ValueError("existing checkpoint model does not match")
        accumulated.extend(
            item for item in existing.results if item.condition not in conditions
        )
    identity = ""
    with tempfile.TemporaryDirectory(prefix="forge-a49-") as name:
        root = Path(name)
        try:
            for definition in definitions:
                partial = run_acceptance_test_synthesis_v1(
                    root,
                    REPOSITORY,
                    (definition,),
                    args.model,
                    model,
                    full_verification_signals=signals,
                    conditions=conditions,
                )
                identity = partial.repository_identity
                accumulated.extend(partial.results)
                checkpoint = AcceptanceTestSynthesisRun(
                    ACCEPTANCE_TEST_SYNTHESIS_V1,
                    ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION,
                    ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION,
                    identity,
                    args.model,
                    model.context_capacity or 0,
                    42,
                    512,
                    tuple(accumulated),
                    True,
                )
                write_acceptance_test_synthesis_json(checkpoint, args.output)
                print(
                    f"checkpoint {args.model} {definition.metadata.task_id}: "
                    f"{len(accumulated)} cells",
                    flush=True,
                )
        finally:
            model.close()
    print(f"completed {args.model}: {len(accumulated)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate exactly-once A50 patch or generated-test replay cells."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation.acceptance_test_synthesis import (
    SynthesisCondition,
    build_acceptance_test_request,
    parse_generated_acceptance_test,
)
from forge.evaluation.realistic_semantic import REALISTIC_SEMANTIC_V1
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    apply_task_setup,
    copy_repository,
    hash_workspace,
)
from forge.evaluation.replay import (
    ReplayArtifactType,
    ReplayBundle,
    ReplayCell,
    ReplayFailure,
    build_mutation_payload,
    create_replay_bundle,
    load_committed_cell,
    record_cell_result_atomic,
    record_unavailable_cell_atomic,
    repository_identity,
    source_state_identity,
    write_replay_bundle_atomic,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog
from scripts.run_realistic_semantic_v1 import _snapshot

TASK_IDS = ("R05", "R06", "R07", "R08")


def _definitions():  # type: ignore[no-untyped-def]
    return tuple(
        definition
        for definition in realistic_semantic_tasks((42,))
        if definition.metadata.task_id in TASK_IDS
    )


def _artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def _baseline(definition: object, destination: Path) -> Path:
    task = definition.metadata.production_task
    workspace = copy_repository(REPOSITORY, destination)
    apply_task_setup(workspace, task.setup)
    return workspace


def _patch_cell(
    definition: object,
    profile_name: str,
    model: object,
    mutation_representation: object,
    replay_root: Path,
    checkpoint_root: Path,
    repository_id: str,
) -> None:
    metadata = definition.metadata
    artifact_type = ReplayArtifactType.MUTATION_PROPOSAL
    committed = load_committed_cell(
        replay_root,
        checkpoint_root,
        task_id=metadata.task_id,
        model_identity=profile_name,
        artifact_type=artifact_type,
    )
    if committed is not None:
        print(f"skip committed patch {profile_name} {metadata.task_id}", flush=True)
        return
    captured: list[ReplayBundle] = []

    def capture(task, workspace, result):  # type: ignore[no-untyped-def]
        if result.metrics.mutations < 1:
            return
        with tempfile.TemporaryDirectory(prefix="forge-a50-baseline-") as name:
            baseline = _baseline(definition, Path(name) / "workspace")
            payload = build_mutation_payload(
                baseline, workspace, task.expected_changed_paths
            )
            bundle = create_replay_bundle(
                benchmark_identity=REALISTIC_SEMANTIC_V1,
                repository_identity_value=repository_id,
                task_id=metadata.task_id,
                task_version=metadata.task_version,
                source_state_identity_value=source_state_identity(baseline),
                model_identity=profile_name,
                seed=42,
                artifact_type=artifact_type,
                payload=payload,
            )
        write_replay_bundle_atomic(bundle, replay_root)
        captured.append(bundle)

    started = time.perf_counter()
    runner = RealWorldEvaluationRunner(
        profile_name,
        model,
        REPOSITORY,
        mutation_representation=mutation_representation,
        result_callback=capture,
    )
    result = runner.run((metadata.production_task,), _snapshot()).results[0]
    elapsed = time.perf_counter() - started
    if not captured:
        record_unavailable_cell_atomic(
            checkpoint_root,
            task_id=metadata.task_id,
            model_identity=profile_name,
            artifact_type=artifact_type,
            reason=ReplayFailure.PATCH_UNAVAILABLE,
            elapsed_seconds=elapsed,
        )
        print(f"checkpoint patch unavailable {profile_name} {metadata.task_id}")
        return
    bundle = captured[0]
    semantic_truth = "PASS" if result.oracle is EvaluationOutcome.PASS else "FAIL"
    record_cell_result_atomic(
        checkpoint_root,
        ReplayCell(
            metadata.task_id,
            profile_name,
            artifact_type.value,
            "available",
            bundle.artifact_id,
            bundle.payload_sha256,
            elapsed,
            semantic_truth,
        ),
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        tool_calls=result.metrics.tool_executions,
        model_calls=result.metrics.model_calls,
        verification_result=result.metrics.verification_plan_result,
    )
    print(
        f"checkpoint patch {profile_name} {metadata.task_id}: {semantic_truth}",
        flush=True,
    )


def _test_cell(
    definition: object,
    profile_name: str,
    model: object,
    replay_root: Path,
    checkpoint_root: Path,
    repository_id: str,
) -> None:
    metadata = definition.metadata
    artifact_type = ReplayArtifactType.GENERATED_ACCEPTANCE_TEST
    committed = load_committed_cell(
        replay_root,
        checkpoint_root,
        task_id=metadata.task_id,
        model_identity=profile_name,
        artifact_type=artifact_type,
    )
    if committed is not None:
        print(f"skip committed test {profile_name} {metadata.task_id}", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="forge-a50-context-") as name:
        baseline = _baseline(definition, Path(name) / "workspace")
        state_identity = source_state_identity(baseline)
        request = build_acceptance_test_request(
            metadata, SynthesisCondition.C1_GROUNDED, baseline
        )
    started = time.perf_counter()
    response = model.generate(request)
    elapsed = time.perf_counter() - started
    candidate = parse_generated_acceptance_test(response.text)
    if candidate is None:
        record_unavailable_cell_atomic(
            checkpoint_root,
            task_id=metadata.task_id,
            model_identity=profile_name,
            artifact_type=artifact_type,
            reason=ReplayFailure.TEST_UNAVAILABLE,
            elapsed_seconds=elapsed,
        )
        print(f"checkpoint test unavailable {profile_name} {metadata.task_id}")
        return
    bundle = create_replay_bundle(
        benchmark_identity=REALISTIC_SEMANTIC_V1,
        repository_identity_value=repository_id,
        task_id=metadata.task_id,
        task_version=metadata.task_version,
        source_state_identity_value=state_identity,
        model_identity=profile_name,
        seed=42,
        artifact_type=artifact_type,
        payload={
            "language": metadata.language,
            "test_name": candidate.test_name,
            "test_source": candidate.test_source,
            "validation_version": "acceptance-test-synthesis-v1",
        },
    )
    write_replay_bundle_atomic(bundle, replay_root)
    record_cell_result_atomic(
        checkpoint_root,
        ReplayCell(
            metadata.task_id,
            profile_name,
            artifact_type.value,
            "available",
            bundle.artifact_id,
            bundle.payload_sha256,
            elapsed,
        ),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model_calls=1,
    )
    print(f"checkpoint test {profile_name} {metadata.task_id}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("qwen-small", "codestral-22b"), required=True
    )
    parser.add_argument("--phase", choices=("patch", "test"), required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    definitions = _definitions()
    if not all(
        item.semantic_score_eligible for item in validate_integrity(definitions)
    ):
        raise RuntimeError("frozen realistic-semantic-v1 integrity failed")
    canonical_before = hash_workspace(REPOSITORY)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.model)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A50 requires unchanged llama.cpp profiles")
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("A50 requires unchanged 8192 context")
    print(f"model artifact {_artifact_identity(profile.backend_config.model_path)}")
    repository_id = repository_identity(REPOSITORY)
    try:
        for definition in definitions:
            if args.phase == "patch":
                _patch_cell(
                    definition,
                    args.model,
                    model,
                    profile.mutation_representation,
                    args.replay_root,
                    args.checkpoint_root,
                    repository_id,
                )
            else:
                _test_cell(
                    definition,
                    args.model,
                    model,
                    args.replay_root,
                    args.checkpoint_root,
                    repository_id,
                )
    finally:
        model.close()
    if canonical_before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical realistic-semantic-v1 repository changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

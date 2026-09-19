"""Replay A50 artifacts and write the source-free cross-model result."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
)
from forge.evaluation.acceptance_test_synthesis import (
    AcceptanceQualification,
    GeneratedAcceptanceTest,
    execute_alternative_correct,
)
from forge.evaluation.realistic_semantic import REALISTIC_SEMANTIC_V1
from forge.evaluation.realworld import hash_workspace, run_oracle
from forge.evaluation.replay import (
    EVALUATION_REPLAY_V1,
    AlternativeObservation,
    CrossModelObservation,
    EvaluationReplayReport,
    ReplayArtifactType,
    ReplayBundle,
    ReplayCell,
    ReplayFailure,
    load_replay_bundle,
    load_replay_manifest,
    qualify_replayed_test,
    replay_generated_test_bundle,
    replay_mutation_bundle,
    repository_identity,
    write_evaluation_replay_report_json,
)

MODELS = ("qwen-small", "codestral-22b")
TASK_IDS = ("R05", "R06", "R07", "R08")


def _bundles(replay_root: Path) -> dict[tuple[str, str, str], ReplayBundle]:
    values = {}
    for entry in load_replay_manifest(replay_root):
        bundle = load_replay_bundle(replay_root / entry.filename)
        values[(bundle.task_id, bundle.model_identity, bundle.artifact_type.value)] = (
            bundle
        )
    return values


def _cell_records(
    checkpoint_root: Path,
    bundles: dict[tuple[str, str, str], ReplayBundle],
    artifact_type: ReplayArtifactType,
) -> tuple[ReplayCell, ...]:
    records = []
    for task_id in TASK_IDS:
        for model in MODELS:
            bundle = bundles.get((task_id, model, artifact_type.value))
            pattern = (
                f"{task_id.casefold()}-{model}-{artifact_type.value.casefold()}.json"
            )
            path = checkpoint_root / pattern
            checkpoint = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
            if bundle is not None:
                records.append(
                    ReplayCell(
                        task_id,
                        model,
                        artifact_type.value,
                        "available",
                        bundle.artifact_id,
                        bundle.payload_sha256,
                        float(checkpoint.get("elapsed_seconds", 0.0)),
                        checkpoint.get("semantic_truth"),
                    )
                )
                continue
            status = "not_attempted"
            elapsed = 0.0
            if checkpoint:
                status = str(checkpoint.get("status", "unavailable"))
                elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
            records.append(
                ReplayCell(
                    task_id,
                    model,
                    artifact_type.value,
                    status,
                    None,
                    None,
                    elapsed,
                )
            )
    return tuple(records)


def _agreement(semantic: str, outcome: ReplayFailure) -> str:
    if outcome not in {ReplayFailure.REPLAY_PASS, ReplayFailure.REPLAY_FAIL}:
        return outcome.value
    accepts = outcome is ReplayFailure.REPLAY_PASS
    if semantic == "PASS":
        return "CORRECT_PATCH_ACCEPTING" if accepts else "CORRECT_PATCH_REJECTING"
    return "WRONG_PATCH_ACCEPTING" if accepts else "WRONG_PATCH_REJECTING"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = hash_workspace(REPOSITORY)
    definitions = {
        item.metadata.task_id: item
        for item in realistic_semantic_tasks((42,))
        if item.metadata.task_id in TASK_IDS
    }
    bundles = _bundles(args.replay_root)
    qualifications = []
    alternatives = []
    observations = []
    with tempfile.TemporaryDirectory(prefix="forge-a50-analysis-") as name:
        root = Path(name)
        for task_id in TASK_IDS:
            definition = definitions[task_id]
            for test_model in MODELS:
                test = bundles.get(
                    (
                        task_id,
                        test_model,
                        ReplayArtifactType.GENERATED_ACCEPTANCE_TEST.value,
                    )
                )
                if test is None:
                    continue
                qualification = qualify_replayed_test(
                    test,
                    root,
                    REPOSITORY,
                    definition.metadata,
                    definition.wrong,
                )
                qualifications.append(qualification)
                if (
                    qualification.qualification
                    == AcceptanceQualification.QUALIFIED.value
                ):
                    candidate = GeneratedAcceptanceTest(
                        str(test.payload["test_name"]),
                        str(test.payload["test_source"]),
                    )
                    alternative = execute_alternative_correct(
                        root, REPOSITORY, definition.metadata, candidate
                    )
                    alternatives.append(
                        AlternativeObservation(
                            task_id,
                            test_model,
                            alternative,
                            test.artifact_id,
                            test.payload_sha256,
                        )
                    )
                for patch_model in MODELS:
                    patch = bundles.get(
                        (
                            task_id,
                            patch_model,
                            ReplayArtifactType.MUTATION_PROPOSAL.value,
                        )
                    )
                    if patch is None:
                        continue
                    cell_root = root / f"{task_id}-{test_model}-{patch_model}"
                    replayed = replay_mutation_bundle(
                        patch,
                        REPOSITORY,
                        cell_root,
                        definition.metadata.production_task.setup,
                        expected_benchmark_identity=REALISTIC_SEMANTIC_V1,
                        expected_task_id=task_id,
                    )
                    oracle = run_oracle(
                        replayed.workspace,
                        definition.metadata.production_task.oracle_commands,
                    )
                    semantic = oracle.value
                    outcome, seconds = replay_generated_test_bundle(
                        test,
                        replayed.workspace,
                        expected_benchmark_identity=REALISTIC_SEMANTIC_V1,
                        canonical_repository=REPOSITORY,
                        expected_task_id=task_id,
                        source_setup=definition.metadata.production_task.setup,
                    )
                    observations.append(
                        CrossModelObservation(
                            task_id,
                            test_model,
                            patch_model,
                            semantic,
                            outcome.value,
                            _agreement(semantic, outcome),
                            test.artifact_id,
                            test.payload_sha256,
                            patch.artifact_id,
                            patch.payload_sha256,
                            REALISTIC_SEMANTIC_V1,
                            seconds,
                        )
                    )
    report = EvaluationReplayReport(
        EVALUATION_REPLAY_V1,
        1,
        1,
        REALISTIC_SEMANTIC_V1,
        repository_identity(REPOSITORY),
        42,
        _cell_records(
            args.checkpoint_root, bundles, ReplayArtifactType.MUTATION_PROPOSAL
        ),
        _cell_records(
            args.checkpoint_root,
            bundles,
            ReplayArtifactType.GENERATED_ACCEPTANCE_TEST,
        ),
        tuple(qualifications),
        tuple(observations),
        tuple(alternatives),
        False,
        before == hash_workspace(REPOSITORY),
    )
    if not report.canonical_unchanged:
        raise RuntimeError("canonical realistic-semantic-v1 repository changed")
    write_evaluation_replay_report_json(report, args.output)
    print(
        f"wrote {len(observations)} cross-model observations and "
        f"{len(qualifications)} qualifications"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

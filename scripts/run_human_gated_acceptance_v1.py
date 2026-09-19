"""Model-free human-gated-acceptance-v1 replay in disposable benchmark copies."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import REPOSITORY, realistic_semantic_tasks
from forge.ephemeral_acceptance import (
    EphemeralAcceptanceCandidate,
    EphemeralAcceptanceGate,
    EphemeralAcceptanceMode,
    EphemeralAcceptanceState,
)
from forge.evaluation.realistic_semantic import REALISTIC_SEMANTIC_V1
from forge.evaluation.realworld import apply_task_setup, copy_repository, hash_workspace
from forge.evaluation.replay import (
    ReplayArtifactType,
    load_replay_bundle,
    load_replay_manifest,
    replay_mutation_bundle,
)


def _full_verification(workspace: Path) -> tuple[bool, float]:
    started = time.perf_counter()
    for command in ("configure", "build", "test"):
        result = subprocess.run(
            (sys.executable, f"tools/{command}.py"),
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=30,
        )
        if result.returncode != 0:
            return False, time.perf_counter() - started
    return True, time.perf_counter() - started


def main() -> int:
    replay_root = Path("eval-results/replay-a50")
    before = hash_workspace(REPOSITORY)
    definitions = {
        item.metadata.task_id: item for item in realistic_semantic_tasks((42,))
    }
    bundles = {}
    for entry in load_replay_manifest(replay_root):
        bundle = load_replay_bundle(replay_root / entry.filename)
        bundles[(bundle.task_id, bundle.model_identity, bundle.artifact_type)] = bundle
    test = bundles[("R05", "qwen-small", ReplayArtifactType.GENERATED_ACCEPTANCE_TEST)]
    r05 = definitions["R05"].metadata.production_task
    candidate = EphemeralAcceptanceCandidate(
        str(test.payload["test_name"]), str(test.payload["test_source"])
    )
    report: dict[str, object] = {"suite": "human-gated-acceptance-v1"}

    with tempfile.TemporaryDirectory(prefix="forge-a51-replay-") as name:
        root = Path(name)
        for label in ("correct", "wrong", "codestral"):
            workspace = copy_repository(REPOSITORY, root / label)
            apply_task_setup(workspace, r05.setup)
            gate = EphemeralAcceptanceGate(
                EphemeralAcceptanceMode.REQUIRED,
                context_paths=("pyservice/state.py", "tests/test_state.py"),
                import_root="pyservice",
            )
            reviews = []
            baseline = gate.prepare_candidate(
                candidate,
                r05.prompt,
                workspace,
                0,
                lambda preview, seen=reviews: seen.append(preview) or True,
            )
            if baseline is not EphemeralAcceptanceState.APPROVED or not reviews:
                raise RuntimeError(f"R05 {label} candidate was not approved")
            if not gate.before_mutation(r05.prompt, workspace, 0):
                raise RuntimeError("R05 approval was stale before mutation")
            if label == "correct":
                for path in r05.expected_changed_paths:
                    (workspace / path).write_bytes((REPOSITORY / path).read_bytes())
            elif label == "wrong":
                path = r05.expected_changed_paths[1]
                (workspace / path).write_bytes((REPOSITORY / path).read_bytes())
            else:
                patch = bundles[
                    ("R05", "codestral-22b", ReplayArtifactType.MUTATION_PROPOSAL)
                ]
                replayed = replay_mutation_bundle(
                    patch,
                    REPOSITORY,
                    root / "codestral-replayed",
                    r05.setup,
                    expected_benchmark_identity=REALISTIC_SEMANTIC_V1,
                    expected_task_id="R05",
                )
                if not replayed.file_hashes:
                    raise RuntimeError("Codestral R05 replay did not apply")
                for path in r05.expected_changed_paths:
                    (workspace / path).write_bytes(
                        (root / "codestral-replayed" / path).read_bytes()
                    )
            acceptance = gate.postmutation(workspace, 1)
            full, full_seconds = (
                _full_verification(workspace)
                if acceptance is EphemeralAcceptanceState.POSTMUTATION_PASS
                else (False, 0.0)
            )
            report[f"R05_{label}"] = {
                "baseline": gate.metrics.baseline_outcome,
                "review": "shown",
                "approval": gate.metrics.approval_outcome,
                "acceptance": gate.metrics.postmutation_outcome,
                "full_verification": "pass" if full else "not_run_or_fail",
                "baseline_seconds": round(gate.metrics.baseline_latency_seconds, 4),
                "postmutation_seconds": round(
                    gate.metrics.postmutation_latency_seconds, 4
                ),
                "full_verification_seconds": round(full_seconds, 4),
                "verified": acceptance is EphemeralAcceptanceState.POSTMUTATION_PASS
                and full,
            }
        # Simulated decisions are evaluator-owned, based on predeclared A50 B/R/W
        # qualification. They are never passed into production policy logic.
        a50 = json.loads(Path("eval-results/a50-evaluation-replay.json").read_text())
        qualifications = {}
        for row in a50["qualifications"]:
            key = (
                row["task_id"],
                row["test_model"],
                ReplayArtifactType.GENERATED_ACCEPTANCE_TEST,
            )
            bundle = bundles[key]
            if (
                row["artifact_id"] != bundle.artifact_id
                or row["payload_sha256"] != bundle.payload_sha256
            ):
                raise RuntimeError("A50 qualification does not match replay bundle")
            qualifications[key[:2]] = row["qualification"]
        for task_id in ("R06", "R07", "R08"):
            report[task_id] = {
                model: "approve"
                if qualifications[(task_id, model)] == "QUALIFIED"
                else "reject"
                for model in ("qwen-small", "codestral-22b")
            }

    report["canonical_unchanged"] = before == hash_workspace(REPOSITORY)
    print(json.dumps(report, indent=2, sort_keys=True))
    if (
        not report["canonical_unchanged"]
        or not report["R05_correct"]["verified"]
        or report["R05_wrong"]["verified"]
        or report["R05_wrong"]["acceptance"] != "fail"
        or not report["R05_codestral"]["verified"]
        or any(
            "approve" in report[task_id].values() for task_id in ("R06", "R07", "R08")
        )
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

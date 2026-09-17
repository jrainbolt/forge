"""Run frozen realistic-semantic-v1 through production Forge orchestration."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    BUILD,
    CONFIGURE,
    REPOSITORY,
    TEST,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    EvaluationOutcome,
    RepositorySnapshot,
    copy_repository,
    hash_workspace,
    render_realistic_semantic,
    run_oracle,
    run_realistic_semantic_v1,
    write_realistic_semantic_json,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def _snapshot() -> RepositorySnapshot:
    identity = hashlib.sha256(repr(hash_workspace(REPOSITORY)).encode()).hexdigest()
    suffixes = frozenset({".py", ".c", ".h"})
    sources = tuple(
        path
        for path in REPOSITORY.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "tests" not in path.relative_to(REPOSITORY).parts
    )
    tests = tuple(
        path
        for path in REPOSITORY.rglob("*")
        if path.is_file() and path.suffix in suffixes and "tests" in path.parts
    )
    loc = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in (*sources, *tests)
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="forge-a45-baseline-") as name:
        copy = copy_repository(REPOSITORY, Path(name).resolve() / "workspace")
        outcome = run_oracle(copy, (CONFIGURE, BUILD, TEST))
    return RepositorySnapshot(
        "service-engine-v1",
        identity,
        "Python+C17",
        len(sources),
        len(tests),
        loc,
        BUILD,
        TEST,
        outcome,
        time.perf_counter() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds requires unique comma-separated integers")
    definitions = realistic_semantic_tasks(seeds)
    integrity = validate_integrity(definitions)
    if not all(item.semantic_score_eligible for item in integrity):
        raise RuntimeError("benchmark integrity failed before model execution")
    snapshot = _snapshot()
    if snapshot.baseline_outcome is not EvaluationOutcome.PASS:
        raise RuntimeError("canonical realistic benchmark baseline failed")

    canonical_before = hash_workspace(REPOSITORY)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.model)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A45 requires the unchanged llama.cpp profiles")
    artifact = _artifact_identity(profile.backend_config.model_path)
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError(
            f"realistic-semantic-v1 requires context 8192, got {model.context_capacity}"
        )
    try:
        run = run_realistic_semantic_v1(
            args.model,
            model,
            REPOSITORY,
            snapshot,
            tuple(item.metadata for item in definitions),
            integrity,
            model_artifact=artifact,
            mutation_representation=profile.mutation_representation,
        )
    finally:
        model.close()
    if canonical_before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical realistic semantic repository changed")
    write_realistic_semantic_json(run, args.output)
    print(render_realistic_semantic(run))
    print(f"Canonical unchanged: {run.canonical_unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

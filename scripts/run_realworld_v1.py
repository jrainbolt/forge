"""Opt-in local realworld-v1 baseline runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from forge.embedding_config import load_embedding_profile
from forge.evaluation import (
    RealWorldEvaluationRunner,
    foundation_realworld_tasks,
    inspect_repository,
    render_realworld_report,
    write_realworld_json,
)
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="qwen-small")
    parser.add_argument("--embedding-config", type=Path, required=True)
    parser.add_argument("--embedding-profile", required=True)
    parser.add_argument(
        "--skip-semantic-index",
        action="store_true",
        help="measure the lexical fallback after separately observing cold-index cost",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configure = ("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON")
    build = ("cmake", "--build", "build")
    test = ("ctest", "--test-dir", "build", "--output-on-failure")
    snapshot = inspect_repository(
        args.repository,
        name=args.repository.name,
        language="C17",
        source_suffixes=frozenset({".c", ".h"}),
        setup_commands=(configure,),
        build_command=build,
        test_command=test,
    )
    if snapshot.baseline_outcome.value != "PASS":
        raise RuntimeError("benchmark baseline build/tests failed")
    embedding_started = time.perf_counter()
    embedding = load_embedding_profile(args.embedding_config, args.embedding_profile)
    embedding_load = time.perf_counter() - embedding_started
    catalog = load_model_catalog(args.config, default_backend_registry())
    model_started = time.perf_counter()
    model = catalog.create(args.model)
    model_load = time.perf_counter() - model_started
    try:
        with model, embedding:
            run = RealWorldEvaluationRunner(
                args.model,
                model,
                args.repository,
                embedding_model=None if args.skip_semantic_index else embedding,
            ).run(foundation_realworld_tasks(), snapshot)
        write_realworld_json(run, args.output)
        print(render_realworld_report(run))
        print(f"Model load: {model_load:.3f}s")
        print(f"Embedding load: {embedding_load:.3f}s")
        print(f"Canonical unchanged: {run.canonical_unchanged}")
    finally:
        model.close()
        embedding.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

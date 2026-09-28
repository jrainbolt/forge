"""Run the bounded two-task A61 Foundation discovery smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.real_repository_discovery_boundary_v1.suite import (
    SUITE,
    AuthorityMode,
    production_inputs,
    score_acquired_source,
    tasks,
)
from benchmarks.real_repository_pilot_v1.suite import (
    create_snapshot,
    repository_identity,
    snapshot_description,
)
from forge.evaluation.realworld import RealWorldEvaluationRunner
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


@dataclass(frozen=True, slots=True)
class SmokeResult:
    task_id: str
    production_inputs: dict[str, object]
    discovery_calls: int
    search_calls: int
    index_queries: int
    index_builds: int
    index_refreshes: int
    index_seconds: float
    source_reads: int
    context_files: tuple[str, ...]
    expected_area_reached: bool
    source_sufficient: bool
    discovery_score: str
    mutation_ready: bool
    changed_paths: tuple[str, ...]
    final_status: str


def _metric(result, name: str, default=0):  # type: ignore[no-untyped-def]
    return getattr(result.metrics, name, default)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", default="qwen-large")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    canonical = args.repository.resolve(strict=True)
    before = repository_identity(canonical)
    definitions = tuple(
        item
        for item in tasks()
        if item.task_id in {"F01", "F03"}
        and item.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
    )
    if len(definitions) != 2:
        raise RuntimeError("A61 requires exactly two discovery controls")

    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if (
        not isinstance(profile.backend_config, LlamaCppConfig)
        or profile.backend_config.context_size != 8192
    ):
        raise RuntimeError("A61 smoke requires an unchanged 8192-token profile")

    cells: list[SmokeResult] = []
    with tempfile.TemporaryDirectory(prefix="forge-a61-") as name:
        snapshot = create_snapshot(canonical, Path(name) / "foundation")
        model = catalog.create(args.profile)
        try:
            for definition in definitions:
                raw = (
                    RealWorldEvaluationRunner(
                        args.profile,
                        model,
                        snapshot,
                        mutation_representation=profile.mutation_representation,
                    )
                    .run(
                        (definition.production_task,),
                        snapshot_description(snapshot),
                    )
                    .results[0]
                )
                inspected = tuple(raw.metrics.files_inspected)
                reached = bool(set(inspected).intersection(definition.expected_values))
                sufficient = reached and raw.metrics.mutation_ready_reached
                cells.append(
                    SmokeResult(
                        definition.task_id,
                        production_inputs(definition),
                        raw.metrics.discovery_calls,
                        _metric(raw, "lexical_search_queries"),
                        _metric(raw, "lexical_index_queries"),
                        raw.metrics.lexical_index_builds,
                        raw.metrics.lexical_index_refreshes,
                        raw.metrics.lexical_index_duration_seconds,
                        raw.metrics.source_reads,
                        inspected,
                        reached,
                        sufficient,
                        score_acquired_source(
                            definition, inspected, sufficient=sufficient
                        ).value,
                        raw.metrics.mutation_ready_reached,
                        raw.changed_paths,
                        raw.final_status,
                    )
                )
                if repository_identity(canonical) != before:
                    raise RuntimeError("canonical Foundation changed during A61")
        finally:
            model.close()

    payload = {
        "suite": SUITE,
        "profile": args.profile,
        "canonical_pre": before,
        "canonical_post": repository_identity(canonical),
        "canonical_unchanged": repository_identity(canonical) == before,
        "results": [asdict(cell) for cell in cells],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

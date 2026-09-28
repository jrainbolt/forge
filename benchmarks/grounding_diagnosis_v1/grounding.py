"""Model-free reconstruction of bounded A63 discovery variants."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.grounding_diagnosis_v1.suite import HistoricalCase
from benchmarks.real_repository_pilot_v1.suite import create_snapshot
from benchmarks.real_repository_pilot_v2.suite import PilotV2Task
from forge.evidence_coverage import EvidenceCoverageState, decompose_evidence_plan
from forge.lexical_index import RepositoryLexicalIndex
from forge.retrieval import SourceKind


@dataclass(frozen=True, slots=True)
class GroundingDiagnostic:
    case_id: str
    profile: str
    task_id: str
    task_text: str
    search_terms: tuple[str, ...]
    retrieval_results: tuple[str, ...]
    trusted_reads_g0: tuple[str, ...]
    selected_context_g0: tuple[str, ...]
    coverage_complete_g0: bool
    unresolved_goals_g0: tuple[str, ...]
    remaining_context_budget: int
    stopping_cause: str
    g0_sufficient: bool
    g1_additional_reads: tuple[str, ...]
    g1_sufficient: bool
    g2_triggered: bool
    g2_additional_reads: tuple[str, ...]
    g2_sufficient: bool
    maximum_additional_rounds: int


def diagnose_grounding(
    case: HistoricalCase, definition: PilotV2Task, canonical: Path
) -> GroundingDiagnostic:
    expected = set(definition.edit_paths)
    with tempfile.TemporaryDirectory(prefix=f"forge-a63-{case.case_id}-") as name:
        snapshot = create_snapshot(canonical, Path(name) / "foundation")
        plan = decompose_evidence_plan(definition.production_task.prompt)
        coverage = EvidenceCoverageState(plan)
        active = coverage.active_goal
        assert active is not None
        for index, path in enumerate(case.acquired_paths):
            coverage.register_source(active.goal_id, path, 0, f"g0-read-{index}")
        unresolved = tuple(
            item.goal_id
            for item in coverage.results()
            if item.status != "source_covered"
        )

        index = RepositoryLexicalIndex(snapshot, cache_root=Path(name) / "cache")
        matches = index.search(
            active.description,
            limit=8,
            preferred_source_kind=SourceKind.IMPLEMENTATION,
        )
        ranked = tuple(item.path for item in matches)
        extra = next((path for path in ranked if path not in case.acquired_paths), None)
        g1_reads = (extra,) if extra is not None else ()
        g0_sufficient = expected.issubset(case.acquired_paths)
        g1_sufficient = expected.issubset({*case.acquired_paths, *g1_reads})

        g2_triggered = not coverage.complete
        g2_reads = g1_reads if g2_triggered else ()
        g2_sufficient = expected.issubset({*case.acquired_paths, *g2_reads})
        if case.context_quality == "MISDIRECTED" and coverage.complete:
            cause = "MODEL_ACCEPTED_FIRST_PLAUSIBLE_SOURCE"
        elif not coverage.complete:
            cause = "INSUFFICIENT_EVIDENCE_COVERAGE"
        else:
            cause = "OTHER"
        source_bytes = sum(
            (snapshot / path).stat().st_size for path in case.acquired_paths
        )
        remaining = max(0, 8192 - 512 - source_bytes // 4)
        return GroundingDiagnostic(
            case.case_id,
            case.profile,
            case.task_id,
            definition.production_task.prompt,
            (active.description,),
            ranked,
            case.acquired_paths,
            case.acquired_paths,
            coverage.complete,
            unresolved,
            remaining,
            cause,
            g0_sufficient,
            g1_reads,
            g1_sufficient,
            g2_triggered,
            g2_reads,
            g2_sufficient,
            1,
        )


def source_free(value: GroundingDiagnostic) -> dict[str, object]:
    return asdict(value)

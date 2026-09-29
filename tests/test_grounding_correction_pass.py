from __future__ import annotations

from dataclasses import dataclass

import pytest

from forge.evidence_coverage import (
    EvidenceCoverageState,
    EvidenceGoal,
    EvidenceGoalKind,
    TaskEvidencePlan,
    decompose_evidence_plan,
)


def _relationship_plan(owner: str, relationship: str) -> TaskEvidencePlan:
    return TaskEvidencePlan(
        (
            EvidenceGoal("G1", owner),
            EvidenceGoal(
                "G2",
                relationship,
                EvidenceGoalKind.RELATIONSHIP,
                depends_on=("G1",),
            ),
        ),
        semantic_matching=True,
    )


def test_definition_reference_edge_satisfies_without_full_lexical_equality() -> None:
    state = EvidenceCoverageState(
        decompose_evidence_plan(
            "Correct the sensor boundary so measured intensity reaches zero"
        )
    )
    state.register_matching_source(
        "src/sensor.c",
        "int sensor_available(){return measure_intensity();}\n",
        0,
        "caller",
    )
    assert not state.complete
    state.register_matching_source(
        "src/measurement.c",
        "int measure_intensity(){return 0;}\n",
        0,
        "definition",
    )
    assert state.complete
    reasons = tuple(
        reason for item in state.results() for reason in item.evidence_reasons
    )
    assert any(reason.startswith("symbol_definition:") for reason in reasons)
    assert any(reason.startswith("symbol_reference:") for reason in reasons)


def test_import_edge_is_recorded_and_can_close_relationship() -> None:
    state = EvidenceCoverageState(
        _relationship_plan(
            "configuration loader", "settings module ownership connection"
        )
    )
    state.register_matching_source(
        "configuration_loader.py",
        "from settings import VALUE\ndef configuration_loader(): return VALUE\n",
        0,
        "loader",
    )
    assert not state.complete
    state.register_matching_source("settings.py", "VALUE = 1\n", 0, "settings")
    assert state.complete
    assert any(
        reason.startswith("include_or_import:")
        for reason in state.results()[1].evidence_reasons
    )


def test_declaration_implementation_reason_is_explainable() -> None:
    state = EvidenceCoverageState(
        TaskEvidencePlan(
            (EvidenceGoal("G1", "widget execution"),), semantic_matching=True
        )
    )
    state.register_matching_source(
        "src/widget.c", "int widget_execute(){return 1;}\n", 0, "definition"
    )
    state.register_matching_source(
        "include/widget.h", "int widget_execute(void);\n", 0, "declaration"
    )
    assert any(
        reason.startswith("declaration_implementation:")
        for reason in state.results()[0].evidence_reasons
    )


def test_two_unconnected_reads_do_not_close_relationship() -> None:
    plan = TaskEvidencePlan(
        (
            EvidenceGoal("G1", "alpha implementation"),
            EvidenceGoal("G2", "beta implementation"),
            EvidenceGoal(
                "G3",
                "coordination invariant",
                EvidenceGoalKind.RELATIONSHIP,
                depends_on=("G1", "G2"),
            ),
        ),
        semantic_matching=True,
    )
    state = EvidenceCoverageState(plan)
    state.register_matching_source("alpha.py", "ALPHA = 1\n", 0, "alpha")
    state.register_matching_source("beta.py", "BETA = 2\n", 0, "beta")
    assert not state.complete
    assert not state.results()[2].evidence_reasons


@dataclass(frozen=True, slots=True)
class HeldOutCase:
    case_id: str
    plan: TaskEvidencePlan
    sources: tuple[tuple[str, str], ...]
    expected_grounded: bool


HELD_OUT_CASES = (
    HeldOutCase(
        "H01-single-retry",
        decompose_evidence_plan("Fix retry boundary so attempts stop at the limit"),
        (
            (
                "retry.py",
                "def retry_boundary(): return attempts_stop_at_limit()\n"
                "def attempts_stop_at_limit(): return True\n"
                "attempts_stop_at_limit()\n",
            ),
        ),
        True,
    ),
    HeldOutCase(
        "H02-single-cache",
        decompose_evidence_plan("Ensure cache boundary so eviction reaches zero size"),
        (
            (
                "cache.py",
                "def cache_boundary(): return eviction_size()\n"
                "def eviction_size(): return 0\n"
                "eviction_size()\n",
            ),
        ),
        True,
    ),
    HeldOutCase(
        "H03-multi-source",
        decompose_evidence_plan(
            "Correct the meter boundary so sampled intensity reaches zero"
        ),
        (
            ("meter.c", "int meter_read(){return sample_intensity();}\n"),
            ("sample.c", "int sample_intensity(){return 0;}\n"),
        ),
        True,
    ),
    HeldOutCase(
        "H04-insufficient",
        decompose_evidence_plan(
            "Correct the meter boundary so sampled intensity reaches zero"
        ),
        (
            ("logging.c", "void write_log_message(){}\n"),
            ("metrics.c", "int request_count(){return 0;}\n"),
        ),
        False,
    ),
)


@pytest.mark.parametrize("case", HELD_OUT_CASES, ids=lambda case: case.case_id)
def test_frozen_held_out_grounding_controls(case: HeldOutCase) -> None:
    state = EvidenceCoverageState(case.plan)
    for index, (path, content) in enumerate(case.sources):
        state.register_matching_source(path, content, 0, f"read-{index}")
    assert state.complete is case.expected_grounded

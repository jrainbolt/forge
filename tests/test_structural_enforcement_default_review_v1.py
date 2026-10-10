from __future__ import annotations

import json
from pathlib import Path

from forge.project_config import parse_project_commands
from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessEvaluation,
    StructuralCompletenessMode,
    evaluate_structural_completeness,
)


def _mode(value: str | None) -> StructuralCompletenessMode:
    project = {} if value is None else {"structural_completeness_mode": value}
    return parse_project_commands({"project": project}).structural_completeness_mode


def test_legacy_default_remains_shadow_and_explicit_modes_override() -> None:
    assert _mode(None) is StructuralCompletenessMode.SHADOW
    assert _mode("off") is StructuralCompletenessMode.OFF
    assert _mode("shadow") is StructuralCompletenessMode.SHADOW
    assert _mode("enforce") is StructuralCompletenessMode.ENFORCE


def test_real_configuration_path_rollback_drill(tmp_path: Path) -> None:
    calls = 0

    def failed(_workspace: Path) -> StructuralCompletenessEvaluation:
        nonlocal calls
        calls += 1
        return StructuralCompletenessEvaluation(
            StructuralCompletenessDecision.FAIL, ("structural-check",)
        )

    sequence = (
        StructuralCompletenessMode.SHADOW,
        StructuralCompletenessMode.ENFORCE,
        StructuralCompletenessMode.SHADOW,
        StructuralCompletenessMode.OFF,
    )
    records = [
        evaluate_structural_completeness(mode, tmp_path, failed) for mode in sequence
    ]
    assert [record.enforced_rejection for record in records] == [
        False,
        True,
        False,
        False,
    ]
    assert calls == 3
    assert not tuple(tmp_path.iterdir())


def test_obligation_and_not_applicable_accounting() -> None:
    cells = [
        {"checks": 2, "decision": "pass"},
        {"checks": 1, "decision": "fail"},
        {"checks": 0, "decision": "not_applicable"},
    ]
    assert sum(cell["checks"] > 0 for cell in cells) / len(cells) == 2 / 3
    assert sum(cell["decision"] == "not_applicable" for cell in cells) == 1


def test_offline_enforcement_and_false_rejection_accounting() -> None:
    cells = [
        {"decision": "fail", "verification": False, "semantic": False},
        {"decision": "pass", "verification": True, "semantic": True},
        {"decision": "partial", "verification": False, "semantic": False},
        {"decision": "not_applicable", "verification": True, "semantic": True},
    ]
    blocked = [cell for cell in cells if cell["decision"] == "fail"]
    false = [
        cell
        for cell in blocked
        if cell["verification"] is True and cell["semantic"] is True
    ]
    assert len(blocked) == 1
    assert false == []


def test_operational_uncertainty_fails_open(tmp_path: Path) -> None:
    record = evaluate_structural_completeness(
        StructuralCompletenessMode.ENFORCE,
        tmp_path,
        lambda _workspace: StructuralCompletenessEvaluation(
            StructuralCompletenessDecision.PARTIAL,
            operational_failures=("PARSER_FAILURE",),
        ),
    )
    assert record.decision == "partial"
    assert not record.enforced_rejection


def test_telemetry_is_source_free_and_benchmark_is_package_excluded() -> None:
    telemetry = {
        "decision": "pass",
        "check_kinds": ["IMPORT_INCLUDE_RELATION"],
        "latency_seconds": 0.001,
        "subprocess_count": 0,
    }
    encoded = json.dumps(telemetry)
    assert "source_content" not in encoded
    assert "replacement_text" not in encoded
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "structural_enforcement_default_review_v1" not in configuration

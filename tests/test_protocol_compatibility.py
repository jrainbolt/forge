from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    PROTOCOL_COMPATIBILITY_V1,
    build_protocol_compatibility_run,
    protocol_compatibility_to_dict,
    render_protocol_compatibility,
    run_protocol_diagnostics,
)
from forge.models import MockModel


def _edit(path: str, old: str, new: str) -> str:
    return json.dumps(
        {"type": "structured_edit", "path": path, "old_text": old, "new_text": new}
    )


def _valid_responses() -> tuple[str, ...]:
    return (
        "Equality at the boundary limit must also be allowed.",
        json.dumps(
            {"old_text": "return value > limit;", "new_text": "return value >= limit;"}
        ),
        _edit("boundary.c", "return value > limit;", "return value >= limit;"),
        "True must enable and false must disable, so remove the inversion.",
        json.dumps({"old_text": "return not flag", "new_text": "return flag"}),
        _edit("feature.py", "return not flag", "return flag"),
        "The retry default is one but must permit three attempts.",
        json.dumps({"old_text": "return 1;", "new_text": "return 3;"}),
        _edit("retry.c", "return 1;", "return 3;"),
    )


def test_protocol_compatibility_layers_validate_mechanics_and_semantics(
    tmp_path: Path,
) -> None:
    results = run_protocol_diagnostics("mock", MockModel(_valid_responses()))
    run = build_protocol_compatibility_run(results)

    assert run.suite == PROTOCOL_COMPATIBILITY_V1
    assert len(results) == 9
    assert {result.layer for result in results} == {"L1", "L2", "L3"}
    assert all(result.representation_valid for result in results)
    assert all(
        result.oracle == "PASS" for result in results if result.layer in {"L2", "L3"}
    )
    aggregate = run.aggregates[0]
    assert aggregate.l1_concepts == aggregate.l2_valid_edits == 3
    assert aggregate.l3_schema_valid == aggregate.l3_valid_edits == 3
    assert aggregate.l1_to_l2_drop == aggregate.l2_to_l3_drop == 0
    l3 = next(
        result for result in results if result.task_id == "P01" and result.layer == "L3"
    )
    with_l4 = build_protocol_compatibility_run(
        (*results, replace(l3, layer="L4", representation_valid=False))
    ).aggregates[0]
    assert with_l4.l3_to_l4_drop == 1
    payload = protocol_compatibility_to_dict(run)
    assert payload["prompt_version"] == "protocol-diagnostic-v1"
    assert "mock" in render_protocol_compatibility(run)


def test_failure_layers_distinguish_concept_noop_and_schema_failure() -> None:
    bad = (
        "No relevant behavior described.",
        json.dumps(
            {"old_text": "return value > limit;", "new_text": "return value > limit;"}
        ),
        "not json",
    ) * 3
    results = run_protocol_diagnostics("bad", MockModel(bad))

    assert {
        result.representation_status for result in results if result.layer == "L1"
    } == {"concept_missing"}
    assert {
        result.representation_status for result in results if result.layer == "L2"
    } == {"no_op"}
    assert {
        result.representation_status for result in results if result.layer == "L3"
    } == {"schema_invalid"}
    aggregate = build_protocol_compatibility_run(results).aggregates[0]
    assert (
        aggregate.l1_concepts
        == aggregate.l2_valid_edits
        == aggregate.l3_valid_edits
        == 0
    )

from __future__ import annotations

import json

import pytest

from forge.evaluation import (
    EDIT_REPRESENTATION_V1,
    SYNTHETIC_FIXTURES,
    EditRepresentation,
    SourceSpan,
    build_edit_representation_run,
    construct_source_spans,
    edit_representation_to_dict,
    evaluate_representation_response,
    materialize_line_range,
    materialize_source_span,
    render_edit_representation,
    run_edit_representation_diagnostics,
)
from forge.models import MockModel, ModelUsage


def _responses() -> tuple[str, ...]:
    return (
        json.dumps(
            {"old_text": "return value > limit;", "new_text": "return value >= limit;"}
        ),
        json.dumps(
            {"start_line": 3, "end_line": 3, "new_text": "    return value >= limit;\n"}
        ),
        json.dumps(
            {
                "span_id": "S3",
                "new_text": "    return value >= limit;\n}\n",
            }
        ),
        json.dumps({"old_text": "return not flag", "new_text": "return flag"}),
        json.dumps({"start_line": 2, "end_line": 2, "new_text": "    return flag\n"}),
        json.dumps({"span_id": "S2", "new_text": "    return flag\n"}),
        json.dumps({"old_text": "return 1;", "new_text": "return 3;"}),
        json.dumps({"start_line": 2, "end_line": 2, "new_text": "    return 3;\n"}),
        json.dumps({"span_id": "S2", "new_text": "    return 3;\n"}),
    )


def test_all_representations_reuse_fixtures_oracles_and_profile_matrix() -> None:
    first = run_edit_representation_diagnostics("model-a", MockModel(_responses()))
    second = run_edit_representation_diagnostics("model-b", MockModel(_responses()))
    run = build_edit_representation_run((*first, *second))

    assert run.suite == EDIT_REPRESENTATION_V1
    assert run.context_capacity == 8192
    assert len(run.results) == 18
    assert {item.representation for item in run.results} == {
        representation.value for representation in EditRepresentation
    }
    assert all(item.response_valid for item in run.results)
    assert all(item.material_delta for item in run.results)
    assert all(item.oracle == "PASS" for item in run.results)
    assert all(
        item.target_region_correct
        for item in run.results
        if item.representation != EditRepresentation.EXACT_TEXT.value
    )
    assert len(run.aggregates) == 6
    assert all(item.oracle_passes == 3 for item in run.aggregates)
    assert all(gain.r2_oracle_gain == gain.r3_oracle_gain == 0 for gain in run.gains)
    payload = edit_representation_to_dict(run)
    assert payload["prompt_version"] == "edit-representation-diagnostic-v1"
    assert payload["context_capacity"] == 8192
    assert "model-a" in render_edit_representation(run)


def test_line_range_bounds_materialization_target_and_oracle() -> None:
    fixture = SYNTHETIC_FIXTURES[0]
    updated = materialize_line_range(
        fixture.source, 3, 3, "    return value >= limit;\n"
    )
    assert updated.startswith("#include <stdbool.h>\n")
    assert "return value >= limit;\n}" in updated

    for start, end in ((0, 1), (3, 2), (1, 99)):
        with pytest.raises(ValueError, match="invalid source line range"):
            materialize_line_range(fixture.source, start, end, "replacement\n")

    wrong = evaluate_representation_response(
        "mock",
        fixture,
        EditRepresentation.LINE_RANGE,
        json.dumps({"start_line": 1, "end_line": 1, "new_text": "/* changed */\n"}),
        0.1,
        ModelUsage(),
    )
    assert wrong.response_valid
    assert wrong.target_selectable
    assert wrong.target_region_correct is False
    assert wrong.material_delta
    assert wrong.oracle == "FAIL"
    assert wrong.failure == "TARGET_WRONG"

    no_delta = evaluate_representation_response(
        "mock",
        fixture,
        EditRepresentation.LINE_RANGE,
        json.dumps(
            {
                "start_line": 3,
                "end_line": 3,
                "new_text": "    return value > limit;\n",
            }
        ),
        0.1,
        ModelUsage(),
    )
    assert no_delta.response_valid
    assert no_delta.target_selectable
    assert no_delta.target_region_correct
    assert not no_delta.replacement_valid
    assert no_delta.failure == "NO_DELTA"


def test_span_validation_materialization_schema_and_aggregation() -> None:
    fixture = SYNTHETIC_FIXTURES[1]
    spans = construct_source_spans(fixture.source)
    assert spans == (SourceSpan("S1", 1, 1), SourceSpan("S2", 2, 2))
    updated, selected = materialize_source_span(
        fixture.source, spans, "S2", "    return flag\n"
    )
    assert selected.span_id == "S2"
    assert updated == "def enabled(flag: bool) -> bool:\n    return flag\n"
    with pytest.raises(ValueError, match="invalid source span"):
        materialize_source_span(fixture.source, spans, "S9", "change\n")

    invalid = evaluate_representation_response(
        "mock",
        fixture,
        EditRepresentation.SOURCE_SPAN,
        '{"span_id":"S2","new_text":"x","extra":true}',
        0.2,
        ModelUsage(),
        spans=spans,
    )
    assert invalid.failure == "SCHEMA_INVALID"
    assert not invalid.response_valid
    assert not invalid.material_delta

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.evaluation import run_line_range_mutation_v1, run_structured_mutation_v1
from forge.models import MockModel, MutationRepresentationPolicy
from forge.orchestration import (
    LineRangeEditProposal,
    MutationCandidate,
    RepositoryChatSession,
    RepositoryOrchestrationError,
    StructuredEditFailure,
    ToolCallOutcome,
    parse_model_output,
    validate_line_range_edit,
)
from forge.orchestration.protocol import build_mutation_ready_output
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def _candidate(
    path: Path, *, generation: int = 0, start: int = 1, end: int = 200
) -> MutationCandidate:
    return MutationCandidate(
        path.name,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        generation,
        "trusted-source",
        start,
        end,
    )


def test_line_range_mutation_v1_runs_eight_production_cases(tmp_path: Path) -> None:
    result = run_line_range_mutation_v1(tmp_path / "line-range")

    assert result.tasks_passed == result.tasks_total == 8
    assert tuple(task.task_id for task in result.tasks) == tuple(
        f"L{number:02d}" for number in range(1, 9)
    )
    assert run_structured_mutation_v1(tmp_path / "exact").tasks_passed == 8


def test_line_range_derives_exact_lf_and_preserves_prefix_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_bytes(b"prefix\nold one\nold two\nsuffix\n")
    result = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 2, 3, "new one\nnew two"),
        (_candidate(path, end=4),),
        tmp_path,
        0,
    )

    assert result.valid
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.arguments is not None
    assert result.arguments["edits"] == [
        {"old": "old one\nold two\n", "new": "new one\nnew two\n"}
    ]
    assert path.read_bytes() == b"prefix\nold one\nold two\nsuffix\n"


def test_line_range_preserves_crlf_unterminated_final_line_and_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_bytes(b"first\r\nsecond\r\nfinal")
    candidate = _candidate(path, end=3)
    crlf = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 2, 2, "changed"),
        (candidate,),
        tmp_path,
        0,
    )
    final = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 3, 3, "last"),
        (candidate,),
        tmp_path,
        0,
    )
    deletion = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 2, 2, ""),
        (candidate,),
        tmp_path,
        0,
    )

    assert crlf.arguments is not None
    assert crlf.arguments["edits"] == [{"old": "second\r\n", "new": "changed\r\n"}]
    assert final.arguments is not None
    assert final.arguments["edits"] == [{"old": "final", "new": "last"}]
    assert deletion.valid


def test_line_range_rejects_bounds_authority_boolean_staleness_and_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("one\ntwo\nthree\n")
    candidate = _candidate(path, start=2, end=2)
    proposals = (
        LineRangeEditProposal("source.txt", True, 2, "x"),
        LineRangeEditProposal("source.txt", 0, 1, "x"),
        LineRangeEditProposal("source.txt", 3, 2, "x"),
        LineRangeEditProposal("source.txt", 3, 3, "x"),
        LineRangeEditProposal("source.txt", 2, 99, "x"),
    )
    assert all(
        validate_line_range_edit(item, (candidate,), tmp_path, 0).failure
        is StructuredEditFailure.OUT_OF_RANGE
        for item in proposals
    )
    noop = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 2, 2, "two"),
        (candidate,),
        tmp_path,
        0,
    )
    assert noop.failure is StructuredEditFailure.NO_OP_EDIT
    path.write_text("one\nexternally changed\nthree\n")
    stale = validate_line_range_edit(
        LineRangeEditProposal("source.txt", 2, 2, "new"),
        (candidate,),
        tmp_path,
        0,
    )
    assert stale.failure is StructuredEditFailure.STALE_SOURCE


def test_line_range_protocol_is_strict_candidate_bound_and_explicit() -> None:
    output = build_mutation_ready_output(
        ("source.c",), representation=MutationRepresentationPolicy.LINE_RANGE
    )
    branch = output.schema["oneOf"][0]  # type: ignore[index]
    assert branch["properties"]["type"]["const"] == "line_range_edit"  # type: ignore[index]
    assert branch["properties"]["path"]["enum"] == ("source.c",)  # type: ignore[index]
    parsed = parse_model_output(
        json.dumps(
            {
                "type": "line_range_edit",
                "path": "source.c",
                "start_line": 2,
                "end_line": 3,
                "new_text": "changed",
            }
        )
    )
    assert parsed.outcome is ToolCallOutcome.LINE_RANGE_EDIT
    assert parsed.line_range_edit is not None


def test_mutation_ready_final_gets_line_range_specific_correction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.c"
    source.write_text("VALUE = 1\n")
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "read",
                    "tool": "repository.read_file",
                    "arguments": {"path": "main.c"},
                }
            ),
            json.dumps({"type": "final", "answer": "The change is needed."}),
            json.dumps(
                {
                    "type": "line_range_edit",
                    "path": "main.c",
                    "start_line": 1,
                    "end_line": 1,
                    "new_text": "VALUE = 2",
                }
            ),
            json.dumps({"type": "final", "answer": "Changed."}),
        )
    )
    session = RepositoryChatSession(
        "fixture",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    response = session.execute_task("Change VALUE to 2")

    assert source.read_text() == "VALUE = 2\n"
    assert response.coding_task is not None
    assert response.coding_task.transition_metrics.premature_finals == 1
    assert response.coding_task.structured_mutation_metrics.line_range_attempts == 1
    assert len(model.requests) == 4
    ready = model.requests[1]
    correction = model.requests[2]
    assert "line_range_edit" in str(ready.output.schema)
    assert "structured_edit" not in str(ready.output.schema)
    assert "  1 | VALUE = 1" in str(ready.messages)
    assert "matching the requested response schema" in ready.messages[0].content
    assert "Final example" not in ready.messages[0].content
    assert "line_range_edit" in correction.messages[-1].content
    assert "inclusive 1-based" in correction.messages[-1].content
    assert "repository.apply_patch" not in correction.messages[-1].content


def test_two_line_range_finals_stop_without_preview(tmp_path: Path) -> None:
    source = tmp_path / "main.c"
    source.write_text("VALUE = 1\n")
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "read",
                    "tool": "repository.read_file",
                    "arguments": {"path": "main.c"},
                }
            ),
            json.dumps({"type": "final", "answer": "Need a change."}),
            json.dumps({"type": "final", "answer": "Still no change."}),
        )
    )
    session = RepositoryChatSession(
        "fixture",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    with pytest.raises(RepositoryOrchestrationError, match="no mutation proposed"):
        session.execute_task("Change VALUE to 2")
    assert session.last_coding_task is not None
    assert session.last_coding_task.transition_metrics.premature_finals == 2
    assert session.last_coding_task.structured_mutation_metrics.line_range_attempts == 0
    assert source.read_text() == "VALUE = 1\n"

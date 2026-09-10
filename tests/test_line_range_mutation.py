from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.evaluation import run_line_range_mutation_v1, run_structured_mutation_v1
from forge.models import MutationRepresentationPolicy
from forge.orchestration import (
    LineRangeEditProposal,
    MutationCandidate,
    StructuredEditFailure,
    ToolCallOutcome,
    parse_model_output,
    validate_line_range_edit,
)
from forge.orchestration.protocol import build_mutation_ready_output


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

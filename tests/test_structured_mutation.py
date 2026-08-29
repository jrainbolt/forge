from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.evaluation import run_structured_mutation_v1
from forge.models import MockModel
from forge.orchestration import (
    MutationCandidate,
    RepositoryChatSession,
    StructuredEditFailure,
    StructuredEditProposal,
    validate_structured_edit,
)
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def _candidate(path: Path, start: int = 1, end: int = 20) -> MutationCandidate:
    return MutationCandidate(
        path.name,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        0,
        "trusted-read",
        start,
        end,
    )


def test_structured_mutation_v1_runs_eight_production_cases(tmp_path: Path) -> None:
    result = run_structured_mutation_v1(tmp_path / "suite")
    assert result.tasks_passed == result.tasks_total == 8
    assert tuple(task.task_id for task in result.tasks) == tuple(
        f"P{number:02d}" for number in range(1, 9)
    )


def test_exact_materialization_preserves_bytes_and_supports_text_edges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_bytes("α\told\r\nkeep\r\n".encode())
    proposal = StructuredEditProposal("source.txt", "α\told", "α\tnew")
    result = validate_structured_edit(proposal, (_candidate(path),), tmp_path, 0)
    assert result.valid
    assert result.arguments is not None
    assert result.arguments["edits"] == [{"old": "α\told", "new": "α\tnew"}]
    assert path.read_bytes() == "α\told\r\nkeep\r\n".encode()


def test_deletion_duplicate_range_and_bounds_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("first\nduplicate\nduplicate\noutside\n")
    candidate = _candidate(path, 1, 3)
    deletion = validate_structured_edit(
        StructuredEditProposal("source.txt", "first\n", ""),
        (candidate,),
        tmp_path,
        0,
    )
    duplicate = validate_structured_edit(
        StructuredEditProposal("source.txt", "duplicate", "changed"),
        (candidate,),
        tmp_path,
        0,
    )
    outside = validate_structured_edit(
        StructuredEditProposal("source.txt", "outside", "changed"),
        (candidate,),
        tmp_path,
        0,
    )
    oversized = validate_structured_edit(
        StructuredEditProposal("source.txt", "x" * 17000, "y"),
        (candidate,),
        tmp_path,
        0,
    )
    assert deletion.valid
    assert duplicate.failure is StructuredEditFailure.OLD_TEXT_AMBIGUOUS
    assert outside.failure is StructuredEditFailure.OUT_OF_RANGE
    assert oversized.failure is StructuredEditFailure.TOO_LARGE


def test_mutation_ready_schema_is_candidate_bound_and_hides_raw_patch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "main.py"
    path.write_text("VALUE = 1\n")

    def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": tool,
                "arguments": arguments,
            }
        )

    model = MockModel(
        (
            call("search", "repository.search_files", {"query": "VALUE"}),
            call("read", "repository.read_file", {"path": "main.py"}),
            json.dumps(
                {
                    "type": "structured_edit",
                    "path": "main.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ),
            json.dumps({"type": "final", "answer": "done"}),
        )
    )
    RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    ).execute_task("Change VALUE")
    schema = str(model.requests[2].output.schema)
    assert "structured_edit" in schema and "main.py" in schema
    assert "repository.apply_patch" not in schema
    assert "/etc/passwd" not in schema

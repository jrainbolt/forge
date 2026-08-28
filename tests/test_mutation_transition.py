import hashlib
import json
from pathlib import Path

import pytest

from forge.evaluation import run_mutation_transition_v1
from forge.models import MockModel
from forge.orchestration import (
    CodingTaskState,
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def test_mutation_transition_v1_runs_six_production_cases(tmp_path: Path) -> None:
    result = run_mutation_transition_v1(tmp_path / "suite")
    assert result.tasks_passed == result.tasks_total == 6
    assert tuple(task.task_id for task in result.tasks) == tuple(
        f"M{number:02d}" for number in range(1, 7)
    )


def test_candidates_require_trusted_hash_and_remain_bounded() -> None:
    state = CodingTaskState(0)
    assert not state.enter_mutation_ready()
    for number in range(5):
        state.consider_source(f"src/file{number}.py", "a" * 64, 0, f"read-{number}")
    assert state.mutation_candidate_paths == tuple(
        f"src/file{number}.py" for number in range(1, 5)
    )
    assert state.enter_mutation_ready()


def test_mutation_ready_schema_contains_patch_not_broad_discovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.py"
    source.write_text("VALUE = 1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = MockModel(
        (
            _call("search", "repository.search_files", {"query": "VALUE"}),
            _call("read", "repository.read_file", {"path": "main.py"}),
            _call(
                "patch",
                "repository.apply_patch",
                {
                    "path": "main.py",
                    "expected_sha256": digest,
                    "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
                },
            ),
            _final("Changed."),
        )
    )
    response = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    ).execute_task("Update the value")
    schema = str(model.requests[2].output.schema)
    assert "repository.apply_patch" in schema
    for name in (
        "repository.search_files",
        "repository.semantic_search",
        "repository.lexical_search",
        "repository.find_symbol",
        "repository.read_file",
    ):
        assert name not in schema
    assert response.coding_task.transition_metrics.entries == 1
    assert response.coding_task.transition_metrics.proposals == 1


def test_second_mutation_ready_final_fails_truthfully(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("VALUE = 1\n")
    model = MockModel(
        (
            _call("search", "repository.search_files", {"query": "VALUE"}),
            _call("read", "repository.read_file", {"path": "main.py"}),
            _final("I would update it."),
            _final("Still no patch."),
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    with pytest.raises(RepositoryOrchestrationError, match="no mutation proposed"):
        session.execute_task("Update the value")
    assert session.last_coding_task is not None
    assert session.last_coding_task.transition_metrics.premature_finals == 2
    assert source.read_text() == "VALUE = 1\n"


def test_read_mode_never_enters_mutation_ready(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("VALUE = 1\n")
    model = MockModel(
        (
            _call("search", "repository.search_files", {"query": "VALUE"}),
            _call("read", "repository.read_file", {"path": "main.py"}),
            _final("VALUE is one."),
        )
    )
    response = RepositoryChatSession(
        "test", model, tmp_path, require_relevant_source=False
    ).ask("What is VALUE?")
    assert response.coding_task is None
    assert all(
        "repository.apply_patch" not in str(item.output.schema)
        for item in model.requests
    )

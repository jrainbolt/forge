from __future__ import annotations

import json
from pathlib import Path

from forge.context_planner import (
    ContextAdmissionStatus,
    ContextPlanner,
    ObservationType,
    recommend_reference_window,
    recommend_symbol_window,
)
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.tools import (
    PermissionDecision,
    ToolEvidence,
    ToolExecutionMetadata,
    ToolResult,
    ToolResultStatus,
)


def result(name: str, output: object, *, invocation: str = "call") -> ToolResult:
    return ToolResult(
        invocation,
        name,
        ToolResultStatus.SUCCESS,
        ToolExecutionMetadata(PermissionDecision.ALLOW, 0.0),
        output=output,  # type: ignore[arg-type]
    )


def register(
    planner: ContextPlanner,
    item: ToolResult,
    evidence: ToolEvidence,
    arguments: dict[str, object],
    *,
    generation: int = 0,
) -> None:
    planner.register(
        assistant_text=(
            '{"type":"tool_call","id":"call","tool":"'
            + item.tool_name
            + '","arguments":{}}'
        ),
        rendered_result='{"type":"tool_result","output":"' + ("x" * 300) + '"}',
        result=item,
        evidence=evidence,
        arguments=arguments,
        generation=generation,
    )


def test_explicit_budget_accounts_for_every_component() -> None:
    planner = ContextPlanner(model_capacity=2000, reserved_output=200)

    budget = planner.budget(
        system_cost=100,
        tool_definition_cost=150,
        durable_conversation_cost=75,
        active_task_cost=50,
    )

    assert budget.remaining_estimated_tokens == 1361


def test_small_whole_file_fits_and_large_file_is_rejected(tmp_path: Path) -> None:
    small = tmp_path / "small.py"
    large = tmp_path / "large.py"
    small.write_text("VALUE = 1\n")
    large.write_text("value = 1\n" * 2000)
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)

    admitted = planner.preflight(
        "repository.read_file",
        {"path": "small.py"},
        tmp_path,
        remaining_tokens=500,
        generation=0,
    )
    rejected = planner.preflight(
        "repository.read_file",
        {"path": "large.py"},
        tmp_path,
        remaining_tokens=500,
        generation=0,
    )

    assert admitted.status is ContextAdmissionStatus.ADMIT
    assert rejected.status is ContextAdmissionStatus.REJECT_TOO_LARGE
    assert "read_range" in rejected.message


def test_range_fits_where_whole_file_does_not_and_large_range_rejects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.py"
    source.write_text("value = 1\n" * 2000)
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)

    whole = planner.preflight(
        "repository.read_file",
        {"path": "large.py"},
        tmp_path,
        remaining_tokens=300,
        generation=0,
    )
    narrow = planner.preflight(
        "repository.read_range",
        {"path": "large.py", "start_line": 20, "end_line": 30},
        tmp_path,
        remaining_tokens=300,
        generation=0,
    )
    broad = planner.preflight(
        "repository.read_range",
        {"path": "large.py", "start_line": 1, "end_line": 400},
        tmp_path,
        remaining_tokens=300,
        generation=0,
    )

    assert whole.status is ContextAdmissionStatus.REJECT_TOO_LARGE
    assert narrow.status is ContextAdmissionStatus.ADMIT
    assert broad.status is ContextAdmissionStatus.REJECT_TOO_LARGE


def test_symbol_and_reference_windows_clip_and_split() -> None:
    assert recommend_symbol_window("a.py", 100, 140).ranges == ((80, 160),)
    assert recommend_reference_window("a.py", 5).ranges == ((1, 25),)
    assert recommend_symbol_window("a.py", 10, 900, file_line_count=1000).ranges == (
        (1, 200),
        (721, 920),
    )
    assert recommend_symbol_window("a.py", 2, 5, file_line_count=8).ranges == ((1, 8),)


def test_targeted_source_compacts_broad_search_but_preserves_history() -> None:
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    search = result(
        "repository.search_files",
        {"matches": ({"path": "module.py", "line": 5},)},
        invocation="search",
    )
    register(planner, search, ToolEvidence.DISCOVERY, {"query": "target"})
    source = result(
        "repository.read_range",
        {"path": "module.py", "text": "def target(): pass"},
        invocation="range",
    )
    register(
        planner,
        source,
        ToolEvidence.SOURCE_CONTENT,
        {"path": "module.py", "start_line": 1, "end_line": 3},
    )

    assert len(planner.history) == 2
    assert planner.metrics.observations_compacted >= 1
    assert "x" * 100 not in "".join(
        message.content for message in planner.active_messages[:2]
    )


def test_mutation_drops_stale_source_but_metadata_remains() -> None:
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    source = result(
        "repository.read_file",
        {"path": "module.py", "text": "old"},
        invocation="read",
    )
    register(
        planner,
        source,
        ToolEvidence.SOURCE_CONTENT,
        {"path": "module.py"},
    )

    planner.mutation_succeeded(1)

    assert planner.history[0].observation_type is ObservationType.SOURCE_FILE
    assert not planner.active_messages


def test_repeated_read_is_generation_aware(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    source = result("repository.read_file", {"path": "module.py", "text": "VALUE = 1"})
    register(
        planner,
        source,
        ToolEvidence.SOURCE_CONTENT,
        {"path": "module.py"},
    )

    duplicate = planner.preflight(
        "repository.read_file",
        {"path": "module.py"},
        tmp_path,
        remaining_tokens=1000,
        generation=0,
    )
    after_mutation = planner.preflight(
        "repository.read_file",
        {"path": "module.py"},
        tmp_path,
        remaining_tokens=1000,
        generation=1,
    )

    assert duplicate.status is ContextAdmissionStatus.REJECT_REDUNDANT
    assert after_mutation.status is ContextAdmissionStatus.ADMIT


def test_consecutive_broad_search_rejected_but_search_after_source_allowed(
    tmp_path: Path,
) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    search = result(
        "repository.search_files",
        {"matches": ({"path": "module.py", "line": 1},)},
    )
    register(planner, search, ToolEvidence.DISCOVERY, {"query": "VALUE"})

    redundant = planner.preflight(
        "repository.search_files",
        {"query": "OTHER"},
        tmp_path,
        remaining_tokens=1000,
        generation=0,
    )
    source = result("repository.read_range", {"path": "module.py", "text": "VALUE = 1"})
    register(
        planner,
        source,
        ToolEvidence.SOURCE_CONTENT,
        {"path": "module.py", "start_line": 1, "end_line": 1},
    )
    later = planner.preflight(
        "repository.search_files",
        {"query": "OTHER"},
        tmp_path,
        remaining_tokens=1000,
        generation=0,
    )

    assert redundant.status is ContextAdmissionStatus.REJECT_REDUNDANT
    assert later.status is ContextAdmissionStatus.ADMIT


def test_budget_pressure_never_drops_required_source() -> None:
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    source = result("repository.read_range", {"path": "module.py", "text": "important"})
    register(
        planner,
        source,
        ToolEvidence.SOURCE_CONTENT,
        {"path": "module.py", "start_line": 1, "end_line": 2},
    )

    assert not planner.compact_to_fit(0)
    assert any("x" * 100 in message.content for message in planner.active_messages)


def test_successful_verification_is_bounded_and_old_generation_compacts() -> None:
    planner = ContextPlanner(model_capacity=4096, reserved_output=256)
    first = result("project.test", {"exit_code": 1}, invocation="failure")
    register(planner, first, ToolEvidence.TEST_RESULT, {}, generation=0)
    second = result("project.test", {"exit_code": 0}, invocation="success")
    register(planner, second, ToolEvidence.TEST_RESULT, {}, generation=1)

    assert len(planner.history) == 2
    assert planner.metrics.observations_compacted >= 1


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def test_read_session_rejects_whole_file_then_admits_targeted_range(
    tmp_path: Path,
) -> None:
    (tmp_path / "large.py").write_text("value = 1\n" * 2000)
    model = MockModel(
        (
            _call("whole", "repository.read_file", {"path": "large.py"}),
            _call(
                "range",
                "repository.read_range",
                {"path": "large.py", "start_line": 1, "end_line": 10},
            ),
            json.dumps({"type": "final", "answer": "The selected range sets value."}),
        ),
        context_capacity=4096,
    )

    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("Explain large.py")

    assert response.text == "The selected range sets value."
    assert response.context_metrics.context_rejections == 1
    assert response.context_metrics.whole_file_reads == 0
    assert response.context_metrics.range_reads == 1
    assert response.tool_activity[0].status == "failure"


def test_read_session_rejects_consecutive_search_after_candidate(
    tmp_path: Path,
) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    model = MockModel(
        (
            _call("search-1", "repository.search_files", {"query": "VALUE"}),
            _call("search-2", "repository.search_files", {"query": "VALUE"}),
            _call(
                "range",
                "repository.read_range",
                {"path": "module.py", "start_line": 1, "end_line": 1},
            ),
            json.dumps({"type": "final", "answer": "VALUE is assigned to one."}),
        )
    )

    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("Find VALUE")

    assert response.context_metrics.context_rejections == 1
    assert response.tool_activity[1].status == "failure"
    assert response.context_metrics.range_reads == 1

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.cli import _resolve_model_profile_interactive
from forge.interaction import AutonomyMode
from forge.models import MockModel, ModelSelectionError
from forge.orchestration import RepositoryChatSession
from forge.orchestration.repository_session import (
    RepositoryQueryScope,
    ToolActivity,
    classify_repository_query_scope,
    requires_repository_grounding,
    tool_budget_snapshot,
)
from forge.repl import run_repl


def _input(values: tuple[str, ...]):
    iterator = iter(values)

    def read(_prompt: str) -> str:
        return next(iterator)

    return read


def _read(identifier: str, path: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "id": identifier,
            "tool": "repository.read_file",
            "arguments": {"path": path},
        }
    )


def _final(answer: str = "done") -> str:
    return json.dumps({"type": "final", "answer": answer})


def _list(identifier: str, path: str = ".") -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "id": identifier,
            "tool": "repository.list_directory",
            "arguments": {"path": path},
        }
    )


def _workspace(root: Path) -> None:
    for path in ("input/control.py", "combat/attack.py", "actors/health.py"):
        target = root / path
        target.parent.mkdir(exist_ok=True)
        target.write_text(f"# {path}\ncombat input damage health\n")


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("How does MeleeAttackSystem work?", RepositoryQueryScope.NARROW),
        (
            "Trace combat from player input through damage",
            RepositoryQueryScope.MEDIUM,
        ),
        ("What parts of this project are most developed?", RepositoryQueryScope.BROAD),
    ),
)
def test_repository_scope_classification(
    question: str, expected: RepositoryQueryScope
) -> None:
    assert classify_repository_query_scope(question) is expected


def test_explicit_entire_project_language_is_broad() -> None:
    assert (
        classify_repository_query_scope(
            "Analyze the entire project and create a multiplayer plan."
        )
        is RepositoryQueryScope.BROAD
    )


def test_tool_budget_accounts_for_failures_and_categories() -> None:
    activities = [
        ToolActivity("d", "repository.search_files", "success", "discovery", False),
        ToolActivity("r", "repository.read_file", "failure", "source_content", False),
        ToolActivity("t", "project.test", "success", "test_result", False),
        ToolActivity("w", "repository.apply_patch", "success", "patch_success", False),
    ]
    budget = tool_budget_snapshot(activities, 8)
    assert (budget.used, budget.remaining, budget.failed) == (4, 4, 1)
    assert (budget.discovery, budget.source_reads) == (1, 0)
    assert (budget.verification, budget.mutation) == (1, 1)


def test_chat_brainstorming_stays_conversational_without_tools(tmp_path: Path) -> None:
    model = MockModel(("Paladin, rogue, or mage could each work.",))
    session = RepositoryChatSession("fixture", model, tmp_path)
    output: list[str] = []
    run_repl(
        session,
        input_fn=_input(
            (
                "/chat",
                "I'm thinking about another character archetype; what are options?",
                "/exit",
            )
        ),
        output_fn=output.append,
    )
    assert session.last_activity == ()
    assert model.requests[0].output.schema is None


def test_chat_explicit_repository_request_uses_repository_protocol(
    tmp_path: Path,
) -> None:
    (tmp_path / "combat.py").write_text("class Combat: pass\n")
    model = MockModel((_read("read", "combat.py"), _final("Combat is local.")))
    session = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    )
    run_repl(
        session,
        input_fn=_input(("/chat", "Inspect combat.py in this project", "/exit")),
        output_fn=lambda _line: None,
    )
    assert tuple(item.path for item in session.last_activity) == ("combat.py",)
    assert model.requests[0].output is not None


def test_medium_trace_requires_multiple_distinct_sources(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (
            _read("one", "input/control.py"),
            _read("two", "combat/attack.py"),
            _read("three", "actors/health.py"),
            _final("input to attack to health"),
        )
    )
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("Trace combat from player input through damage")
    assert {item.path for item in response.tool_activity if item.path} == {
        "input/control.py",
        "combat/attack.py",
        "actors/health.py",
    }


def test_narrow_question_can_finish_from_one_targeted_source(tmp_path: Path) -> None:
    (tmp_path / "melee.py").write_text("class MeleeAttackSystem: pass\n")
    model = MockModel((_read("one", "melee.py"), _final("targeted")))
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("How does MeleeAttackSystem work?")
    assert tuple(item.path for item in response.tool_activity) == ("melee.py",)


def test_narrow_early_stop_preserves_unused_budget(tmp_path: Path) -> None:
    (tmp_path / "melee.py").write_text("class MeleeAttackSystem: pass\n")
    (tmp_path / "other.py").write_text("unrelated = True\n")
    model = MockModel(
        (_read("one", "melee.py"), _read("unneeded", "other.py"), _final("targeted"))
    )
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("How does MeleeAttackSystem work?")
    assert tuple(item.path for item in response.tool_activity) == ("melee.py",)
    assert response.tool_budget.remaining == 7
    assert len(model.requests) == 3


def test_discovery_soft_budget_preserves_source_capacity(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (
            _list("discover-two", "combat"),
            _list("discover-three", "actors"),
            _read("source", "combat/attack.py"),
            _final("limited evidence"),
        )
    )
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("Tell me about this project")
    assert response.tool_budget.discovery == 2
    assert response.tool_budget.source_reads == 1
    assert response.tool_budget.remaining == 5
    assert response.text.startswith("Partial repository survey:")


def test_medium_one_source_returns_partial_trace(tmp_path: Path) -> None:
    _workspace(tmp_path)
    response = RepositoryChatSession(
        "fixture",
        MockModel((_read("one", "combat/attack.py"), _final("attack dispatch"))),
        tmp_path,
        require_relevant_source=False,
    ).ask("Trace melee combat from input through damage")
    assert response.text.startswith("Partial repository trace:")
    assert "remain unverified" in response.text


def test_broad_query_gathers_multiple_repository_areas(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (
            _read("one", "input/control.py"),
            _read("two", "combat/attack.py"),
            _read("three", "actors/health.py"),
            _final("bounded comparison"),
        )
    )
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("What parts of this project are most developed?")
    source_areas = {
        Path(item.path).parent
        for item in response.tool_activity
        if item.evidence == "source_content" and item.path
    }
    assert len(source_areas) == 3
    assert response.tool_activity[0].acquisition_origin == "orchestrator_broad_survey"


def test_broad_comparison_from_one_source_is_explicitly_partial(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (_read("one", "combat/attack.py"), _final("combat wins"), _final("still"))
    )
    response = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    ).ask("What parts of this project are most developed?")
    assert response.text.startswith("Partial repository survey:")
    assert response.tool_budget.source_reads == 1


def test_exhausted_broad_survey_is_disclosed_as_partial(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel((_read("one", "combat/attack.py"), _final("bounded observation")))
    response = RepositoryChatSession(
        "fixture",
        model,
        tmp_path,
        require_relevant_source=False,
        max_tool_executions=2,
    ).ask("Analyze the entire project and propose a plan")
    assert response.text.startswith("Partial repository survey:")
    assert response.query_scope is RepositoryQueryScope.BROAD
    assert response.tool_budget.used == response.tool_budget.total == 2


def test_follow_up_topic_shift_adds_fresh_retrieval_guidance(tmp_path: Path) -> None:
    (tmp_path / "melee.py").write_text("melee attack\n")
    (tmp_path / "ranged.py").write_text("ranged attack\n")
    model = MockModel(
        (
            _read("melee", "melee.py"),
            _final("melee"),
            _read("ranged", "ranged.py"),
            _final("ranged"),
        )
    )
    session = RepositoryChatSession(
        "fixture", model, tmp_path, require_relevant_source=False
    )
    session.ask("Explain the melee attack")
    second = session.ask("What about the ranged weapon system?")
    assert tuple(item.path for item in second.tool_activity) == ("ranged.py",)
    assert any(
        "fresh retrieval" in message.content for message in model.requests[2].messages
    )


def test_display_levels_do_not_change_repository_behavior(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text("VALUE = 1\n")
    requests = []
    paths = []
    for display in ("minimal", "normal", "trace"):
        model = MockModel((_read("read", "value.py"), _final()))
        session = RepositoryChatSession(
            "fixture", model, tmp_path, require_relevant_source=False
        )
        run_repl(
            session,
            input_fn=_input(("Inspect value.py", "/exit")),
            output_fn=lambda _line: None,
            display_level=display,
        )
        requests.append(tuple(request.output for request in model.requests))
        paths.append(tuple(item.path for item in session.last_activity))
    assert requests[0] == requests[1] == requests[2]
    assert paths == [("value.py",)] * 3


def test_slash_aliases_and_state_queries(tmp_path: Path) -> None:
    output: list[str] = []
    run_repl(
        RepositoryChatSession("fixture", MockModel(("unused",)), tmp_path),
        input_fn=_input(
            ("/plan", "/mode", "/chat", "/mode", "/display", "/trace", "/exit")
        ),
        output_fn=output.append,
    )
    rendered = "\n".join(output)
    assert "[PLAN] Mode: PLAN" in rendered
    assert "Mode: CHAT" in rendered
    assert "Display: NORMAL" in rendered
    assert "Display: TRACE" in rendered


def test_trace_checklist_omits_unreachable_plan_mutation_phases(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.py").write_text("VALUE = 1\n")
    output: list[str] = []
    run_repl(
        RepositoryChatSession(
            "fixture",
            MockModel((_read("read", "value.py"), _final())),
            tmp_path,
            require_relevant_source=False,
        ),
        input_fn=_input(("Inspect value.py", "/exit")),
        output_fn=output.append,
        display_level="trace",
    )
    rendered = "\n".join(output)
    assert "Synthesize answer or plan" in rendered
    assert "Apply approved change" not in rendered
    assert "○ Verify" not in rendered


def test_ambiguous_protocol_failure_is_presented_as_clarification(
    tmp_path: Path,
) -> None:
    output: list[str] = []
    run_repl(
        RepositoryChatSession(
            "fixture", MockModel(("nonsense", "still nonsense")), tmp_path
        ),
        input_fn=_input(("unclear words", "/exit")),
        output_fn=output.append,
    )
    rendered = "\n".join(output)
    assert "Please clarify what you want to inspect or change" in rendered
    assert "not one valid JSON object" not in rendered


def test_chat_grounding_intent_is_semantic_not_a_phrase_list() -> None:
    assert not requires_repository_grounding(
        "What character archetypes could work in an action game?"
    )
    assert requires_repository_grounding(
        "Based on the current implementation, would multiplayer be difficult?"
    )
    assert requires_repository_grounding("How does MeleeAttackSystem work?")


def test_interactive_missing_default_requires_explicit_profile_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Catalog:
        profile_names = ("small", "large")

        def resolve_profile(self, _explicit: None, _role: object) -> str:
            raise ModelSelectionError("no repository default")

    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    assert (
        _resolve_model_profile_interactive(
            Catalog(), None, AutonomyMode.READ, allow_prompt=True
        )
        == "large"
    )
    with pytest.raises(ModelSelectionError):
        _resolve_model_profile_interactive(
            Catalog(), None, AutonomyMode.READ, allow_prompt=False
        )

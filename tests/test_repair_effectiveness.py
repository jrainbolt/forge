"""A47 verification-grounded repair effectiveness acceptance scenarios."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    REPAIR_EFFECTIVENESS_V1,
    RealisticSemanticResult,
    RealisticSemanticRun,
    RepairCaseType,
    build_repair_case_corpus,
)
from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import (
    CodingTaskPhase,
    LineRangeEditProposal,
    MutationCandidate,
    RepositoryChatSession,
    StructuredEditFailure,
    validate_line_range_edit,
)
from forge.orchestration.repository_session import _bound_repair_mutation_diff
from forge.orchestration.verification_attribution import bounded_failure_lines
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def _call(identifier: str, path: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "id": identifier,
            "tool": "repository.read_file",
            "arguments": {"path": path},
        }
    )


def _edit(path: str, old: str, new: str) -> str:
    return json.dumps(
        {
            "type": "structured_edit",
            "path": path,
            "old_text": old,
            "new_text": new,
        }
    )


def _repair_run(
    workspace: Path,
    *,
    diagnostic: str = "test_value FAILED: expected 3, actual 2",
    repaired_value: int = 3,
    include_history: bool = True,
    verification_plan: VerificationPlan | None = None,
) -> tuple[object, MockModel]:
    workspace.mkdir()
    (workspace / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    test = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "ok='VALUE = 3' in Path('value.py').read_text(); "
            f"print({diagnostic!r}, file=sys.stderr) if not ok else None; "
            "raise SystemExit(0 if ok else 7)",
        ),
        5,
    )
    passing = ProjectCommand((sys.executable, "-c", "pass"), 5)
    model = MockModel(
        (
            _call("read-primary", "value.py"),
            _edit("value.py", "VALUE = 1", "VALUE = 2"),
            _edit("value.py", "VALUE = 2", f"VALUE = {repaired_value}"),
            json.dumps({"type": "final", "answer": "Repair complete."}),
        ),
        context_capacity=8192,
    )
    commands = ProjectCommands(
        configure=passing if verification_plan is not None else None,
        build=passing if verification_plan is not None else None,
        test=test,
    )
    response = RepositoryChatSession(
        REPAIR_EFFECTIVENESS_V1,
        model,
        workspace,
        mode=AutonomyMode.REPAIR,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        verification_plan=verification_plan,
        include_repair_mutation_history=include_history,
    ).run_agent_task("Correct VALUE to 3 and verify the repository.")
    return response, model


def _prompt(model: MockModel) -> str:
    return "\n".join(message.content for message in model.requests[2].messages)


def test_e01_compile_failure_evidence_is_bounded_and_retained(tmp_path: Path) -> None:
    _, model = _repair_run(
        tmp_path / "e01", diagnostic="value.py:1: error: expected expression"
    )
    prompt = _prompt(model)
    assert "value.py:1: error: expected expression" in prompt


def test_e02_test_failure_evidence_retains_assertion(tmp_path: Path) -> None:
    _, model = _repair_run(
        tmp_path / "e02", diagnostic="test_value FAILED: AssertionError: 2 != 3"
    )
    assert "test_value FAILED: AssertionError: 2 != 3" in _prompt(model)
    assert bounded_failure_lines(
        {"stdout": "noise\n", "stderr": "test_value FAILED\nAssertionError: 2 != 3"}
    ) == ("test_value FAILED", "AssertionError: 2 != 3")


def test_e03_first_mutation_diff_is_ordered_and_bounded(tmp_path: Path) -> None:
    _, model = _repair_run(tmp_path / "e03")
    prompt = _prompt(model)
    assert "-VALUE = 1" in prompt and "+VALUE = 2" in prompt
    assert (
        prompt.index("Requested code change")
        < prompt.index("test_value FAILED")
        < prompt.index("Accepted first mutation")
        < prompt.index("END FILE")
    )
    bounded, truncated = _bound_repair_mutation_diff("x" * 10_000)
    assert truncated and len(bounded) == 4096
    assert "truncated by Forge" in bounded
    _, baseline_model = _repair_run(tmp_path / "e03-r0", include_history=False)
    assert "Accepted first mutation" not in _prompt(baseline_model)


def test_e04_repair_source_is_fresh_post_mutation_content(tmp_path: Path) -> None:
    response, model = _repair_run(tmp_path / "e04")
    prompt = _prompt(model)
    assert "VALUE = 2" in prompt
    assert "current workspace" in prompt
    assert response.coding_task.repair_evidence.generation == 1


def test_e05_stale_source_is_rejected_before_write(tmp_path: Path) -> None:
    source = tmp_path / "value.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    old_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = MutationCandidate("value.py", old_hash, 1, "read", 1, 1)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    result = validate_line_range_edit(
        LineRangeEditProposal("value.py", 1, 1, "VALUE = 3"),
        (candidate,),
        tmp_path,
        1,
    )
    assert result.failure is StructuredEditFailure.STALE_SOURCE


def test_e06_single_file_repair_reuses_production_schema(tmp_path: Path) -> None:
    response, model = _repair_run(tmp_path / "e06")
    assert response.coding_task.mutation_count == 2
    assert "structured_edit" in str(model.requests[2].output.schema)


def test_e07_grouped_repair_reuses_grouped_production_schema(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")

    def group(old: int, new: int) -> str:
        return json.dumps(
            {
                "type": "multi_file_structured_edit",
                "edits": [
                    {
                        "path": "a.py",
                        "old_text": f"A = {old}",
                        "new_text": f"A = {new}",
                    },
                    {
                        "path": "b.py",
                        "old_text": f"B = {old}",
                        "new_text": f"B = {new}",
                    },
                ],
            }
        )

    model = MockModel(
        (
            _call("read-a", "a.py"),
            _call("read-b", "b.py"),
            group(1, 2),
            _call("repair-a", "a.py"),
            _call("repair-b", "b.py"),
            group(2, 3),
            json.dumps({"type": "final", "answer": "Done."}),
        )
    )
    test = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "ok='A = 3' in Path('a.py').read_text() and "
            "'B = 3' in Path('b.py').read_text(); sys.exit(0 if ok else 1)",
        ),
        5,
    )
    response = RepositoryChatSession(
        REPAIR_EFFECTIVENESS_V1,
        model,
        tmp_path,
        mode=AutonomyMode.REPAIR,
        registry=create_assist_repository_registry(ProjectCommands(test=test)),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        minimum_source_files=2,
    ).run_agent_task("Change both files to 3 and verify.")
    assert response.coding_task.mutation_count == 2
    assert "multi_file_structured_edit" in str(model.requests[5].output.schema)


def test_e08_unauthorized_repair_path_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "e08"
    workspace.mkdir()
    (workspace / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    command = ProjectCommand((sys.executable, "-c", "raise SystemExit(1)"), 5)
    model = MockModel(
        (
            _call("read", "value.py"),
            _edit("value.py", "VALUE = 1", "VALUE = 2"),
            _edit("other.py", "OTHER = 1", "OTHER = 2"),
            _edit("value.py", "VALUE = 2", "VALUE = 3"),
            json.dumps({"type": "final", "answer": "Stopped."}),
        )
    )
    response = RepositoryChatSession(
        REPAIR_EFFECTIVENESS_V1,
        model,
        workspace,
        mode=AutonomyMode.REPAIR,
        registry=create_assist_repository_registry(ProjectCommands(test=command)),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    ).run_agent_task("Change only value.py.")
    assert (workspace / "other.py").read_text(encoding="utf-8") == "OTHER = 1\n"
    assert response.coding_task.structured_mutation_metrics.corrections == 1


def test_e09_preexisting_failure_blocks_repair() -> None:
    from forge.orchestration import CodingTaskState
    from forge.orchestration.verification_attribution import (
        AttributionResult,
        VerificationAttribution,
    )

    state = CodingTaskState(0, repair_enabled=True)
    state.mutation_count = 1
    state.phase = CodingTaskPhase.MUTATED
    state.verification_finished(
        "test",
        "failure",
        {"outcome": "nonzero_exit", "exit_code": 1},
        attribution=VerificationAttribution(
            AttributionResult.PREEXISTING_OR_UNRELATED, True
        ),
    )
    assert not state.repair_eligible and not state.repair_ready


def test_e10_successful_repair_restarts_full_plan(tmp_path: Path) -> None:
    plan = VerificationPlan(
        "full", ("project.configure", "project.build", "project.test")
    )
    response, _ = _repair_run(tmp_path / "e10", verification_plan=plan)
    runs = response.coding_task.verification_plan_runs
    assert len(runs) == 2
    assert tuple(name for name, _ in runs[0].steps) == plan.steps
    assert tuple(name for name, _ in runs[1].steps) == plan.steps
    assert runs[1].outcome == "pass"


def test_e11_failed_repair_respects_two_mutation_ceiling(tmp_path: Path) -> None:
    response, _ = _repair_run(tmp_path / "e11", repaired_value=4)
    assert response.coding_task.mutation_count == 2
    assert response.coding_task.status.value == "repair_verification_failed"


def test_e12_hidden_evaluator_data_never_enters_repair_context(tmp_path: Path) -> None:
    _, model = _repair_run(tmp_path / "e12")
    prompt = _prompt(model)
    for forbidden in ("hidden oracle", "reference mutation", "expected source"):
        assert forbidden not in prompt.casefold()


def test_repair_case_corpus_records_eligibility_and_disagreement() -> None:
    result = RealisticSemanticResult(
        "R03",
        1,
        "repo",
        "qwen-small",
        "model:sha",
        42,
        8192,
        512,
        0.0,
        "discovery_required",
        "single_file",
        "PASS",
        "PASS",
        True,
        True,
        True,
        True,
        "step_failed",
        "failed",
        "FAIL",
        False,
        False,
        True,
        False,
        "VERIFICATION_FAILED",
        1,
        2,
        0,
        0,
        2,
        9,
        4,
        100,
        20,
        5000,
        10.0,
    )
    oracle_only = replace(
        result,
        task_id="R04",
        verification_status="pass",
        build_test_status="passed",
        semantic_oracle="FAIL",
        repair_used=False,
        failure_layer="SEMANTIC_ORACLE_FAILED",
    )
    run = RealisticSemanticRun(
        "realistic-semantic-v1",
        1,
        1,
        "A46",
        "repo",
        "qwen-small",
        "model:sha",
        8192,
        512,
        0.0,
        (),
        (result, oracle_only),
        (),
        True,
    )
    corpus = build_repair_case_corpus((run,))
    assert corpus[0].eligible and corpus[0].language == "C17"
    assert corpus[0].case_type is RepairCaseType.COMPILE_FAILURE
    assert corpus[1].case_type is RepairCaseType.VERIFICATION_PASS_ORACLE_FAIL
    assert not corpus[1].eligible

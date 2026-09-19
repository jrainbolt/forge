"""human-gated-acceptance-v1 model-free H01-H16 trust-boundary scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.ephemeral_acceptance import (
    EphemeralAcceptanceCandidate,
    EphemeralAcceptanceGate,
    EphemeralAcceptanceMode,
    EphemeralAcceptanceState,
)
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.models import GenerationConfig, MockModel
from forge.orchestration import RepositoryChatSession, RepositoryOrchestrationError
from forge.orchestration.coding_task import (
    CodingTaskState,
    CodingTaskStatus,
    VerificationRecord,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    InvocationApproval,
    PermissionDecision,
    ToolCapability,
    create_assist_repository_policy,
    create_assist_repository_registry,
)

TASK = "In pkg/state.py, make FAILED terminal and add a regression test."
BAD = "def can_transition(state: str) -> bool:\n    return True\n"
GOOD = "def can_transition(state: str) -> bool:\n    return False\n"
SOURCE = "from pkg.state import can_transition\nassert not can_transition('FAILED')\n"


def _workspace(root: Path) -> Path:
    workspace = root / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "pkg" / "state.py").write_text(BAD, encoding="utf-8")
    (workspace / "tests" / "test_state.py").write_text(
        "from pkg.state import can_transition\n", encoding="utf-8"
    )
    return workspace


def _gate(
    mode: EphemeralAcceptanceMode = EphemeralAcceptanceMode.OPTIONAL,
) -> EphemeralAcceptanceGate:
    return EphemeralAcceptanceGate(
        mode,
        context_paths=("pkg/state.py", "tests/test_state.py"),
        import_root="pkg",
    )


def _candidate(source: str = SOURCE) -> EphemeralAcceptanceCandidate:
    return EphemeralAcceptanceCandidate("failed_terminal", source)


def test_h01_safe_baseline_failure_requires_review(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    observed = []
    gate = _gate()

    def review(preview):
        assert gate.state is EphemeralAcceptanceState.REVIEW_REQUIRED
        assert not gate.approved
        observed.append(preview)
        return False

    state = gate.prepare_candidate(_candidate(), TASK, workspace, 0, review)
    assert observed and observed[0].baseline_result == "FAIL"
    assert "ephemeral_test_review_required" in gate.events
    assert state is EphemeralAcceptanceState.REJECTED


def test_h02_baseline_pass_not_reviewable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    calls = []
    gate = _gate()
    state = gate.prepare_candidate(
        _candidate("assert True\n"),
        TASK,
        workspace,
        0,
        lambda preview: calls.append(preview) or True,
    )
    assert state is EphemeralAcceptanceState.BASELINE_NOT_DETECTED
    assert not calls


def test_h03_unsafe_candidate_rejected_before_execution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    state = gate.prepare_candidate(
        _candidate("import socket\nassert False\n"),
        TASK,
        workspace,
        0,
        lambda _preview: True,
    )
    assert state is EphemeralAcceptanceState.UNAVAILABLE
    assert gate.metrics.baseline_outcome == "not_run"


def test_runtime_import_failure_is_not_a_reviewable_baseline_failure(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    seen = []
    gate = _gate()
    outcome = gate.prepare_candidate(
        _candidate("from pkg.missing import value\nassert value == 1\n"),
        TASK,
        workspace,
        0,
        lambda preview: seen.append(preview) or True,
    )
    assert outcome is EphemeralAcceptanceState.UNAVAILABLE
    assert gate.metrics.baseline_outcome == "error"
    assert not seen


def test_dunder_builtin_escape_is_rejected_before_execution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    outcome = gate.prepare_candidate(
        _candidate("assert not __builtins__['__import__']('socket')\n"),
        TASK,
        workspace,
        0,
        lambda _preview: True,
    )
    assert outcome is EphemeralAcceptanceState.UNAVAILABLE
    assert gate.metrics.baseline_outcome == "not_run"


def test_h04_preview_contains_exact_candidate_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seen = []
    _gate().prepare_candidate(
        _candidate(), TASK, workspace, 0, lambda preview: seen.append(preview) or False
    )
    assert seen[0].test_source == SOURCE


def test_h05_preview_shows_failure_and_trusted_execution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seen = []
    _gate().prepare_candidate(
        _candidate(), TASK, workspace, 0, lambda preview: seen.append(preview) or False
    )
    preview = seen[0]
    assert preview.baseline_result == "FAIL"
    assert "AssertionError" in preview.baseline_failure_output
    assert "trusted interpreter" in preview.execution_description
    assert "SEMANTICALLY UNVERIFIED" in preview.warning


def test_h06_approval_binds_candidate_sha(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert gate.metrics.candidate_sha256 == _candidate().sha256
    assert gate._approval is not None
    assert gate._approval.candidate_sha256 == _candidate().sha256


def test_h07_candidate_change_invalidates_approval(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    gate._candidate = _candidate("assert False\n")
    assert not gate.before_mutation(TASK, workspace, 0)
    assert gate.state is EphemeralAcceptanceState.INVALIDATED


def test_h08_source_change_invalidates_approval(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    (workspace / "pkg" / "state.py").write_text(GOOD, encoding="utf-8")
    assert not gate.before_mutation(TASK, workspace, 0)


def test_h09_semantic_approval_grants_no_write_or_execute_authority(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert gate._approval is not None
    assert not isinstance(gate._approval, InvocationApproval)
    policy = resolve_interaction_policy(AutonomyMode.AGENT, "safe")
    assert all(
        policy.decision_for(capability) is PermissionDecision.DENY
        for capability in (
            ToolCapability.WRITE,
            ToolCapability.BUILD,
            ToolCapability.TEST,
            ToolCapability.CONFIGURE,
        )
    )


def test_h10_rejection_has_no_mutation_or_regeneration(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: False)
    assert (workspace / "pkg" / "state.py").read_text() == BAD
    with pytest.raises(RuntimeError):
        gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert gate.state is EphemeralAcceptanceState.REJECTED


def test_h11_acceptance_pass_and_full_pass_can_verify(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert gate.before_mutation(TASK, workspace, 0)
    (workspace / "pkg" / "state.py").write_text(GOOD, encoding="utf-8")
    assert gate.postmutation(workspace, 1) is EphemeralAcceptanceState.POSTMUTATION_PASS
    state = CodingTaskState(0)
    state.mutation_count = 1
    state.generation = 1
    state.ephemeral_acceptance_metrics = gate.metrics
    state.test = VerificationRecord(True, "passed", generation=1)
    assert state.finish("done").status is CodingTaskStatus.COMPLETED_VERIFIED


def test_h12_acceptance_fail_blocks_verified_and_repair(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert gate.postmutation(workspace, 1) is EphemeralAcceptanceState.POSTMUTATION_FAIL
    state = CodingTaskState(0, repair_enabled=True)
    state.mutation_count = 1
    state.ephemeral_acceptance_failed(gate.metrics)
    result = state.finish("failed")
    assert result.status is CodingTaskStatus.EPHEMERAL_ACCEPTANCE_FAILED
    assert not result.repair_eligible and not result.repair_attempted
    assert result.test.status == "not_run"


def test_h13_full_verification_fail_remains_authoritative(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    (workspace / "pkg" / "state.py").write_text(GOOD, encoding="utf-8")
    gate.postmutation(workspace, 1)
    state = CodingTaskState(0)
    state.mutation_count = 1
    state.generation = 1
    state.ephemeral_acceptance_metrics = gate.metrics
    state.test = VerificationRecord(True, "failed", generation=1)
    assert state.finish("failed").status is not CodingTaskStatus.COMPLETED_VERIFIED


def test_h14_no_human_channel_cannot_auto_approve(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate(EphemeralAcceptanceMode.REQUIRED)
    model = MockModel((json.dumps({"test_name": "x", "test_source": SOURCE}),))
    state = gate.generate(model, TASK, workspace, 0, GenerationConfig(), None)
    assert state is EphemeralAcceptanceState.UNAVAILABLE
    assert not model.requests


def test_h15_trusted_exec_cannot_bypass_semantic_review(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    policy = resolve_interaction_policy(AutonomyMode.AGENT, "trusted-exec")
    assert policy.decision_for(ToolCapability.TEST) is PermissionDecision.ALLOW
    gate = _gate(EphemeralAcceptanceMode.REQUIRED)
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, None)
    assert not gate.approved


def test_h16_failure_not_in_repair_context(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    gate.postmutation(workspace, 1)
    state = CodingTaskState(0, repair_enabled=True)
    state.mutation_count = 1
    state.ephemeral_acceptance_failed(gate.metrics)
    result = state.finish("failed")
    assert result.repair_evidence is None
    assert SOURCE not in repr(result)
    assert "AssertionError" not in repr(result)


def test_generation_is_one_call_at_existing_output_budget(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    model = MockModel((json.dumps({"test_name": "x", "test_source": SOURCE}),))
    gate = _gate()
    gate.generate(
        model,
        TASK,
        workspace,
        0,
        GenerationConfig(max_tokens=256, temperature=0.0, seed=42),
        lambda _preview: False,
    )
    assert len(model.requests) == 1
    assert model.requests[0].generation.max_tokens == 256
    assert gate.metrics.generation_calls == 1


def test_approved_source_not_written_to_project(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert not list(workspace.rglob("candidate.py"))
    assert not list(workspace.rglob("__pycache__"))
    gate.discard_source()
    assert gate.preview is None


def test_approval_is_task_and_generation_local(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    gate = _gate()
    gate.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert not gate.before_mutation(TASK + " extra", workspace, 0)
    second = _gate()
    second.prepare_candidate(_candidate(), TASK, workspace, 0, lambda _preview: True)
    assert not second.before_mutation(TASK, workspace, 1)


def _session(workspace: Path, answers: tuple[str, ...], mode: EphemeralAcceptanceMode):
    commands = ProjectCommands(
        test=ProjectCommand(
            (
                "python3",
                "-c",
                "from pathlib import Path; "
                "assert Path('src/value.py').read_text() == 'VALUE = 2\\n'",
            ),
            5,
        )
    )
    return RepositoryChatSession(
        "fixture",
        MockModel(answers),
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        ephemeral_acceptance_mode=mode,
        ephemeral_acceptance_paths=("src/value.py", "tests/test_value.py"),
        ephemeral_acceptance_import_root="src",
        ephemeral_review_callback=lambda _preview: True,
        minimum_source_files=1,
        require_relevant_source=False,
    )


def _session_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "session-workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src" / "value.py").write_text("VALUE = 1\n")
    (workspace / "tests" / "test_value.py").write_text("assert True\n")
    return workspace


def _tool_call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def test_production_session_acceptance_pass_then_full_test_pass(tmp_path: Path) -> None:
    workspace = _session_workspace(tmp_path)
    candidate = json.dumps(
        {
            "test_name": "value_changes",
            "test_source": "from src.value import VALUE\nassert VALUE == 2\n",
        }
    )
    answers = (
        candidate,
        _tool_call("read", "repository.read_file", {"path": "src/value.py"}),
        json.dumps(
            {
                "type": "structured_edit",
                "path": "src/value.py",
                "old_text": "VALUE = 1",
                "new_text": "VALUE = 2",
            }
        ),
        _tool_call("test", "project.test", {}),
        json.dumps({"type": "final", "answer": "Verified."}),
    )
    session = _session(workspace, answers, EphemeralAcceptanceMode.REQUIRED)
    response = session.ask("Change VALUE to 2")
    assert response.coding_task is not None
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert (
        response.coding_task.ephemeral_acceptance_metrics.postmutation_outcome == "pass"
    )
    assert response.coding_task.test.status == "passed"
    assert len(session._model.requests) == 5


def test_production_optional_rejection_continues_without_acceptance(
    tmp_path: Path,
) -> None:
    workspace = _session_workspace(tmp_path)
    answers = (
        json.dumps(
            {
                "test_name": "value_changes",
                "test_source": "from src.value import VALUE\nassert VALUE == 2\n",
            }
        ),
        _tool_call("read", "repository.read_file", {"path": "src/value.py"}),
        json.dumps(
            {
                "type": "structured_edit",
                "path": "src/value.py",
                "old_text": "VALUE = 1",
                "new_text": "VALUE = 2",
            }
        ),
        _tool_call("test", "project.test", {}),
        json.dumps({"type": "final", "answer": "Verified."}),
    )
    session = _session(workspace, answers, EphemeralAcceptanceMode.OPTIONAL)
    session.set_ephemeral_review_callback(lambda _preview: False)
    response = session.ask("Change VALUE to 2")
    assert response.coding_task is not None
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert (
        response.coding_task.ephemeral_acceptance_metrics.approval_outcome == "rejected"
    )
    assert (
        response.coding_task.ephemeral_acceptance_metrics.postmutation_outcome
        == "not_run"
    )


def test_production_acceptance_failure_stops_full_verification(
    tmp_path: Path,
) -> None:
    workspace = _session_workspace(tmp_path)
    answers = (
        json.dumps(
            {
                "test_name": "value_changes",
                "test_source": "from src.value import VALUE\nassert VALUE == 2\n",
            }
        ),
        _tool_call("read", "repository.read_file", {"path": "src/value.py"}),
        json.dumps(
            {
                "type": "structured_edit",
                "path": "src/value.py",
                "old_text": "VALUE = 1",
                "new_text": "VALUE = 3",
            }
        ),
        *(
            json.dumps({"type": "final", "answer": "Acceptance failed."})
            for _ in range(8)
        ),
    )
    session = _session(workspace, answers, EphemeralAcceptanceMode.REQUIRED)
    response = session.ask("Change VALUE to 2")
    assert response.coding_task is not None
    assert response.coding_task.status is CodingTaskStatus.EPHEMERAL_ACCEPTANCE_FAILED
    assert response.coding_task.test.status == "not_run"
    assert (
        response.coding_task.ephemeral_acceptance_metrics.postmutation_outcome == "fail"
    )
    assert not response.coding_task.repair_eligible


def test_production_required_mode_blocks_without_human(tmp_path: Path) -> None:
    workspace = _session_workspace(tmp_path)
    session = _session(
        workspace,
        (json.dumps({"type": "final", "answer": "unused"}),),
        EphemeralAcceptanceMode.REQUIRED,
    )
    session.set_ephemeral_review_callback(None)
    with pytest.raises(RepositoryOrchestrationError, match="required ephemeral"):
        session.ask("Change VALUE to 2")
    assert session.last_coding_task is not None
    assert (workspace / "src" / "value.py").read_text() == "VALUE = 1\n"

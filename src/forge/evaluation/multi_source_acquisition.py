"""Deterministic A42 required-source acquisition and protocol evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import (
    CodingTaskState,
    DuplicateToolCallIdError,
    RepositoryChatSession,
)
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_policy,
    create_assist_repository_registry,
    create_readonly_repository_policy,
    create_readonly_repository_registry,
)

MULTI_SOURCE_ACQUISITION_V1 = "multi-source-acquisition-v1"


@dataclass(frozen=True, slots=True)
class MultiSourceAcquisitionCase:
    case_id: str
    passed: bool
    outcome: str


@dataclass(frozen=True, slots=True)
class MultiSourceAcquisitionRun:
    cases: tuple[MultiSourceAcquisitionCase, ...]

    @property
    def tasks_passed(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def tasks_total(self) -> int:
        return len(self.cases)


def _read(identifier: str, path: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "id": identifier,
            "tool": "repository.read_file",
            "arguments": {"path": path},
        }
    )


def _group() -> str:
    return json.dumps(
        {
            "type": "multi_file_structured_edit",
            "edits": [
                {"path": "a.py", "old_text": "A = 1", "new_text": "A = 2"},
                {"path": "b.py", "old_text": "B = 1", "new_text": "B = 2"},
            ],
        }
    )


def _fixture(root: Path, name: str, *, include_b: bool = True) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_text("A = 1\n")
    if include_b:
        (workspace / "b.py").write_text("B = 1\n")
    return workspace


def _required_session(
    workspace: Path,
    model: MockModel,
    *,
    policy=None,  # type: ignore[no-untyped-def]
    max_tools: int | None = None,
) -> RepositoryChatSession:
    return RepositoryChatSession(
        "multi-source-v1",
        model,
        workspace,
        mode=AutonomyMode.AGENT,
        registry=create_assist_repository_registry(),
        policy=policy or create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        minimum_source_files=2,
        required_candidate_paths=("b.py", "a.py"),
        require_relevant_source=False,
        skip_verification=True,
        max_tool_executions=max_tools,
    )


def run_multi_source_acquisition_v1(root: Path) -> MultiSourceAcquisitionRun:
    """Run the fixed Q01-Q10 deterministic acceptance ladder."""
    root.mkdir(parents=True, exist_ok=True)
    cases: list[MultiSourceAcquisitionCase] = []

    q01 = _fixture(root, "Q01")
    response = _required_session(
        q01,
        MockModel((_group(), json.dumps({"type": "final", "answer": "done"}))),
    ).ask("Change both files")
    acquisition = response.coding_task.source_acquisition_metrics  # type: ignore[union-attr]
    cases.append(
        MultiSourceAcquisitionCase(
            "Q01",
            acquisition.deterministic_source_reads == 2
            and response.coding_task.transition_metrics.entries == 1,  # type: ignore[union-attr]
            "two_required_candidates_acquired",
        )
    )

    state = CodingTaskState(0)
    state.establish_required_candidates(("a.py", "b.py"))
    state.note_source_acquired(
        "a.py", "a" * 64, start_line=1, end_line=1, deterministic=False
    )
    cases.append(
        MultiSourceAcquisitionCase(
            "Q02",
            tuple(item.path for item in state.missing_required_candidates) == ("b.py",),
            "one_existing_one_missing",
        )
    )

    order = CodingTaskState(0)
    order.establish_required_candidates(("z.py", "a.py", "m.py"))
    cases.append(
        MultiSourceAcquisitionCase(
            "Q03",
            order.required_candidate_paths == ("a.py", "m.py", "z.py"),
            "deterministic_path_order",
        )
    )

    q04 = _fixture(root, "Q04")
    denied_model = MockModel((json.dumps({"type": "final", "answer": "unused"}),))
    denied_policy = RuleBasedPolicy(
        {
            "repository.read_file": PermissionDecision.DENY,
            "repository.apply_patch": PermissionDecision.ASK,
            "repository.apply_multi_patch": PermissionDecision.ASK,
        }
    )
    denied_session = _required_session(q04, denied_model, policy=denied_policy)
    denied = False
    try:
        denied_session.ask("Change both files")
    except Exception:
        denied = len(denied_model.requests) == 0
    cases.append(MultiSourceAcquisitionCase("Q04", denied, "read_denied"))

    q05 = _fixture(root, "Q05", include_b=False)
    missing_model = MockModel((json.dumps({"type": "final", "answer": "unused"}),))
    missing_session = _required_session(q05, missing_model)
    missing = False
    try:
        missing_session.ask("Change both files")
    except Exception:
        result = missing_session.last_coding_task
        missing = (
            len(missing_model.requests) == 0
            and result is not None
            and result.source_acquisition_metrics.source_acquisition_failures == 1
        )
    cases.append(MultiSourceAcquisitionCase("Q05", missing, "missing_path_bounded"))

    q06 = _fixture(root, "Q06")
    budget_model = MockModel((_group(),))
    budget_session = _required_session(q06, budget_model, max_tools=2)
    budget = False
    try:
        budget_session.ask("Change both files")
    except Exception as error:
        budget = "reserved tool budget" in str(error) and not budget_model.requests
    cases.append(MultiSourceAcquisitionCase("Q06", budget, "budget_reserved"))

    generation = CodingTaskState(0)
    generation.establish_required_candidates(("a.py", "b.py"))
    for path in generation.required_candidate_paths:
        generation.note_source_acquired(
            path, path[0] * 64, start_line=1, end_line=1, deterministic=True
        )
    generation.invalidate_mutation_ready(1)
    cases.append(
        MultiSourceAcquisitionCase(
            "Q07",
            len(generation.missing_required_candidates) == 2,
            "generation_requires_reacquisition",
        )
    )

    for path in generation.required_candidate_paths:
        assert generation.begin_source_acquisition(path)
        generation.note_source_acquired(
            path, path[0] * 64, start_line=1, end_line=1, deterministic=True
        )
    cases.append(
        MultiSourceAcquisitionCase(
            "Q08",
            generation.required_sources_ready
            and generation.source_acquisition_metrics.deterministic_source_reads == 4,
            "grouped_repair_sources_reacquired",
        )
    )

    q09 = _fixture(root, "Q09")
    corrected_model = MockModel(
        (
            _read("same", "a.py"),
            _read("same", "b.py"),
            _read("new", "b.py"),
            json.dumps({"type": "final", "answer": "done"}),
        )
    )
    corrected = RepositoryChatSession(
        "duplicate-corrected",
        corrected_model,
        q09,
        registry=create_readonly_repository_registry(),
        policy=create_readonly_repository_policy(),
        require_relevant_source=False,
        minimum_source_files=2,
    ).ask("Inspect both files")
    cases.append(
        MultiSourceAcquisitionCase(
            "Q09",
            corrected.protocol_corrections == 1
            and tuple(item.invocation_id for item in corrected.tool_activity)
            == ("same", "new"),
            "duplicate_corrected_once",
        )
    )

    q10 = _fixture(root, "Q10")
    terminal_model = MockModel(
        (_read("same", "a.py"), _read("same", "b.py"), _read("same", "b.py"))
    )
    terminal_session = RepositoryChatSession(
        "duplicate-terminal",
        terminal_model,
        q10,
        registry=create_readonly_repository_registry(),
        policy=create_readonly_repository_policy(),
        require_relevant_source=False,
        minimum_source_files=2,
    )
    terminal = False
    try:
        terminal_session.ask("Inspect both files")
    except DuplicateToolCallIdError as error:
        terminal = error.classification == "DUPLICATE_TOOL_CALL_ID"
    cases.append(
        MultiSourceAcquisitionCase("Q10", terminal, "second_duplicate_terminal")
    )
    return MultiSourceAcquisitionRun(tuple(cases))

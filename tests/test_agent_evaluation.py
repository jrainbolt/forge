from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from forge.evaluation import (
    AGENT_V1,
    AGENT_V1_TASKS,
    AgentEvaluationRunner,
    load_agent_suite,
)
from forge.models import MockModel
from forge.project_config import ProjectCommand, ProjectCommands

FIXTURE = Path(__file__).parent / "fixtures" / "eval_repo"


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def retry_patch(call_id: str) -> str:
    path = "src/tinyqueue/retry.py"
    data = (FIXTURE / path).read_bytes()
    return call(
        call_id,
        "repository.apply_patch",
        {
            "path": path,
            "expected_sha256": hashlib.sha256(data).hexdigest(),
            "edits": [
                {
                    "old": "task.attempts <= self.max_attempts",
                    "new": "task.attempts < self.max_attempts",
                }
            ],
        },
    )


def test_agent_suite_defines_four_frozen_behaviors() -> None:
    assert load_agent_suite(AGENT_V1) is AGENT_V1_TASKS
    assert [task.task_id for task in AGENT_V1_TASKS] == ["G01", "G02", "G03", "G04"]


def test_agent_evaluation_isolated_fixtures_and_deterministic_scores() -> None:
    responses = (
        call("g01-search", "repository.search_files", {"query": "RetryPolicy"}),
        call("g01-retry", "repository.read_file", {"path": "src/tinyqueue/retry.py"}),
        call(
            "g01-service", "repository.read_file", {"path": "src/tinyqueue/service.py"}
        ),
        final("RetryPolicy controls QueueService retry decisions."),
        call("g02-retry", "repository.read_file", {"path": "src/tinyqueue/retry.py"}),
        call("g02-model", "repository.read_file", {"path": "src/tinyqueue/models.py"}),
        retry_patch("g02-patch"),
        final("Boundary fixed."),
        final("Boundary fixed; verification was not run."),
        call("g03-empty", "repository.search_files", {"query": "not_a_symbol"}),
        call("g03-search", "repository.search_files", {"query": "next_attempt"}),
        call("g03-read", "repository.read_file", {"path": "src/tinyqueue/retry.py"}),
        final("next_attempt advances attempts."),
        call("g04-read", "repository.read_file", {"path": "src/tinyqueue/retry.py"}),
        retry_patch("g04-patch"),
        call("g04-test", "project.test", {}),
        call("g04-inspect", "repository.read_file", {"path": "tests/test_retry.py"}),
        final("The boundary changed, but verification failed."),
    )
    fixture_before = {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    runner = AgentEvaluationRunner(
        "fixture",
        MockModel(responses, context_capacity=8192),
        FIXTURE,
        approval_callback=lambda *_args: True,
        commands=ProjectCommands(
            test=ProjectCommand((sys.executable, "-c", "raise SystemExit(4)"), 5)
        ),
    )
    results = runner.run(AGENT_V1_TASKS)
    assert [result.score.total for result in results] == [6, 6, 6, 6]
    assert [result.stop_reason for result in results] == [
        "completed",
        "completed",
        "completed",
        "verification_failed",
    ]
    assert [result.mutation_count for result in results] == [0, 1, 0, 1]
    assert fixture_before == {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }

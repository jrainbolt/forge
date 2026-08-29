from __future__ import annotations

import json
from pathlib import Path

from forge.evaluation import (
    CODING_WRITE_V1,
    CODING_WRITE_V1_TASKS,
    CodingWriteEvaluationRunner,
    load_write_suite,
)
from forge.models import MockModel

FIXTURE = Path(__file__).parent / "fixtures" / "eval_repo"


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def flow(path: str, old: str, new: str, suffix: str) -> tuple[str, ...]:
    return (
        call(f"search-{suffix}", "repository.search_files", {"query": old.split()[0]}),
        call(f"read-{suffix}", "repository.read_file", {"path": path}),
        json.dumps(
            {"type": "structured_edit", "path": path, "old_text": old, "new_text": new}
        ),
        json.dumps({"type": "final", "answer": f"Completed {suffix}."}),
    )


def test_coding_write_suite_has_three_single_step_tasks() -> None:
    tasks = load_write_suite(CODING_WRITE_V1)
    assert tasks == CODING_WRITE_V1_TASKS
    assert [task.task_id for task in tasks] == ["W01", "W02", "W03"]


def test_write_evaluation_uses_fresh_fixtures_and_deterministic_scoring() -> None:
    tests = (FIXTURE / "tests/test_retry.py").read_text()
    responses = (
        *flow(
            "src/tinyqueue/retry.py",
            "task.attempts <= self.max_attempts",
            "task.attempts < self.max_attempts",
            "W01",
        ),
        *flow(
            "tests/test_retry.py",
            tests,
            tests
            + "\n\ndef test_zero_attempt_policy() -> None:\n"
            + '    assert not RetryPolicy(0).should_retry(Task("x", "work"))\n',
            "W02",
        ),
        *flow(
            "src/tinyqueue/retry.py",
            "    def next_attempt(self, task: Task) -> Task:\n",
            "    def is_exhausted(self, task: Task) -> bool:\n"
            "        return task.attempts >= self.max_attempts\n\n"
            "    def next_attempt(self, task: Task) -> Task:\n",
            "W03",
        ),
    )
    fixture_before = {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    runner = CodingWriteEvaluationRunner(
        "fixture",
        MockModel(responses, context_capacity=8192),
        FIXTURE,
        approval_callback=lambda *_args: True,
    )
    results = runner.run(CODING_WRITE_V1_TASKS)
    assert [result.score.total for result in results] == [5, 5, 5]
    assert all(result.mutation_count == 1 for result in results)
    assert all(result.score.unexpected_files_unchanged for result in results)
    assert all(result.status == "completed_unverified" for result in results)
    assert fixture_before == {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }

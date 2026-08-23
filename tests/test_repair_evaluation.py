from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from forge.evaluation import (
    REPAIR_V1,
    REPAIR_V1_TASKS,
    RepairEvaluationRunner,
    load_repair_suite,
)
from forge.models import MockModel
from forge.project_config import ProjectCommand, ProjectCommands

FIXTURE = Path(__file__).parent / "fixtures" / "eval_repo"
PATH = "src/tinyqueue/retry.py"
ORIGINAL = (FIXTURE / PATH).read_text()


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def patch(call_id: str, before: str, old: str, new: str) -> str:
    return call(
        call_id,
        "repository.apply_patch",
        {
            "path": PATH,
            "expected_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "edits": [{"old": old, "new": new}],
        },
    )


def repair_flow(prefix: str, operation: str, bad: str, fixed: str) -> tuple[str, ...]:
    bad_text = ORIGINAL.replace("return task.attempts <= self.max_attempts", bad)
    return (
        call(f"{prefix}-read-0", "repository.read_file", {"path": PATH}),
        patch(
            f"{prefix}-patch-1",
            ORIGINAL,
            "return task.attempts <= self.max_attempts",
            bad,
        ),
        call(f"{prefix}-{operation}-1", f"project.{operation}", {}),
        call(f"{prefix}-read-1", "repository.read_file", {"path": PATH}),
        patch(f"{prefix}-patch-2", bad_text, bad, fixed),
        call(f"{prefix}-{operation}-2", f"project.{operation}", {}),
        final(f"{prefix} complete."),
    )


def test_repair_suite_has_four_controlled_scenarios() -> None:
    assert load_repair_suite(REPAIR_V1) is REPAIR_V1_TASKS
    assert [task.task_id for task in REPAIR_V1_TASKS] == ["R01", "R02", "R03", "R04"]


def test_repair_evaluation_scores_scripted_failures_in_fresh_fixtures() -> None:
    fixed = "return task.attempts < self.max_attempts"
    responses = (
        *repair_flow(
            "r01",
            "test",
            "return task.attempts != self.max_attempts",
            fixed,
        ),
        *repair_flow("r02", "build", "return task.attempts <", fixed),
        *repair_flow(
            "r03",
            "test",
            "return task.attempts != self.max_attempts",
            fixed,
        ),
        call("r04-read-0", "repository.read_file", {"path": PATH}),
        patch(
            "r04-patch-1",
            ORIGINAL,
            "return task.attempts <= self.max_attempts",
            fixed,
        ),
        call("r04-test-1", "project.test", {}),
        call("r04-read-1", "repository.read_file", {"path": PATH}),
        patch(
            "r04-patch-2",
            ORIGINAL.replace("<=", "<"),
            fixed,
            "return task.attempts >= self.max_attempts",
        ),
        final("The process could not start; no repair was made."),
    )
    commands = {
        "R01": ProjectCommands(
            test=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert "
                    "'return task.attempts < self.max_attempts' in "
                    "Path('src/tinyqueue/retry.py').read_text()",
                ),
                5,
            )
        ),
        "R02": ProjectCommands(
            build=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; source = "
                    "Path('src/tinyqueue/retry.py').read_text(); "
                    "compile(source, 'retry.py', 'exec')",
                ),
                5,
            )
        ),
        "R03": ProjectCommands(
            test=ProjectCommand((sys.executable, "-c", "raise SystemExit(8)"), 5)
        ),
        "R04": ProjectCommands(
            test=ProjectCommand(("/definitely/missing/forge-test",), 5)
        ),
    }
    before = {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    runner = RepairEvaluationRunner(
        "fixture",
        MockModel(responses, context_capacity=16384),
        FIXTURE,
        approval_callback=lambda *_args: True,
        commands=commands,
    )
    results = runner.run(REPAIR_V1_TASKS)
    assert [result.score.total for result in results] == [7, 7, 7, 7]
    assert [result.mutation_count for result in results] == [2, 2, 2, 1]
    assert [result.verification_attempts for result in results] == [2, 2, 2, 1]
    assert before == {
        path.relative_to(FIXTURE): path.read_bytes()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }

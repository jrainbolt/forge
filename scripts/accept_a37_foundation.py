"""Reproduce the accepted E04 delta through Forge's production line-range gate."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    RepositorySnapshot,
    copy_repository,
    hash_workspace,
)
from forge.evaluation.realworld_tasks import foundation_realworld_tasks
from forge.models import MockModel, MutationRepresentationPolicy
from forge.process_isolation import ExecutionIsolationMode, ExecutionIsolationPolicy
from forge.project_config import VerificationPlan


def main(*, milestone: str = "a37") -> int:
    canonical = Path(sys.argv[1]).resolve(strict=True)
    task = next(task for task in foundation_realworld_tasks() if task.task_id == "E04")
    if milestone == "a37":
        task = replace(
            task,
            setup_commands=(("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON"),),
            configure_command=None,
            verification_plan=VerificationPlan(
                "foundation-build-test", ("project.build", "project.test")
            ),
            oracle_commands=(
                ("cmake", "--build", "build"),
                ("ctest", "--test-dir", "build", "--output-on-failure"),
            ),
        )
    elif milestone in {"a38", "a39", "a39strict"}:
        if task.setup_commands or task.configure_command is None:
            raise RuntimeError("A38 E04 must have no evaluator configure setup")
        if milestone != "a38":
            task = replace(
                task,
                execution_isolation=ExecutionIsolationPolicy(
                    ExecutionIsolationMode.STRICT
                    if milestone == "a39strict"
                    else ExecutionIsolationMode.CONTROLLED_ENV
                ),
            )
        with tempfile.TemporaryDirectory(prefix="forge-a38-clean-check-") as name:
            copy = copy_repository(canonical, Path(name) / "workspace")
            if (copy / "build").exists():
                raise RuntimeError("A38 disposable workspace is already configured")
    else:
        raise ValueError("unknown acceptance milestone")
    if len(sys.argv) > 2 and sys.argv[2] == "--diagnose":
        logging.basicConfig(level=logging.DEBUG)
        task = replace(task, oracle_commands=())
    source = canonical / "src/clock.c"
    old = "return clock != NULL && clock->tick != UINT64_MAX;"
    lines = source.read_text().splitlines()
    matches = [index for index, line in enumerate(lines, 1) if old in line]
    if len(matches) != 1:
        raise RuntimeError("canonical E04 source precondition not met")
    target_line = matches[0]
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "source",
                    "tool": "repository.read_range",
                    "arguments": {
                        "path": "src/clock.c",
                        "start_line": max(1, target_line - 5),
                        "end_line": target_line + 5,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "line_range_edit",
                    "path": "src/clock.c",
                    "start_line": target_line,
                    "end_line": target_line,
                    "new_text": lines[target_line - 1],
                }
            ),
            json.dumps({"type": "final", "answer": "Applied and verified."}),
        )
    )
    before = hash_workspace(canonical)
    snapshot = RepositorySnapshot(
        "Foundation",
        hashlib.sha256(repr(before).encode()).hexdigest(),
        "C",
        0,
        0,
        0,
        task.build_command or (),
        task.test_command or (),
        EvaluationOutcome.NOT_RUN,
        0.0,
    )
    result = RealWorldEvaluationRunner(
        f"{milestone}-accepted-e04-delta",
        model,
        canonical,
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
    ).run((task,), snapshot)
    attempt = result.results[0]
    assert hash_workspace(canonical) == before
    print(
        json.dumps(
            {
                "canonical_unchanged": result.canonical_unchanged,
                "initial_configured_state": milestone == "a37",
                "repository_identity": snapshot.identity,
                "task_id": attempt.task_id,
                "seed": attempt.seed,
                "status": attempt.status.value,
                "forge_status": attempt.final_status,
                "oracle": attempt.oracle.value,
                "failure": attempt.failure.value
                if attempt.failure is not None
                else None,
                "failure_message": attempt.failure_message,
                "metrics": {
                    "mutation": attempt.metrics.mutations,
                    "line_range_valid": attempt.metrics.line_range_valid,
                    "mutation_ready": attempt.metrics.mutation_ready_reached,
                    "expected_source": attempt.metrics.expected_implementation_acquired,
                    "bootstrap": attempt.metrics.bootstrap_provider,
                    "preview_created": attempt.metrics.preview_created,
                    "verification_tools": attempt.metrics.verification_tools,
                    "verification_result": attempt.metrics.verification_result,
                    "verification_plan_id": attempt.metrics.verification_plan_id,
                    "verification_plan_required_steps": (
                        attempt.metrics.verification_plan_required_steps
                    ),
                    "verification_plan_steps_started": (
                        attempt.metrics.verification_plan_steps_started
                    ),
                    "verification_plan_steps_passed": (
                        attempt.metrics.verification_plan_steps_passed
                    ),
                    "verification_plan_result": (
                        attempt.metrics.verification_plan_result
                    ),
                    "verification_plan_duration": (
                        attempt.metrics.verification_plan_duration
                    ),
                    "verification_build_duration": (
                        attempt.metrics.verification_build_duration
                    ),
                    "configure_required": attempt.metrics.configure_required,
                    "configure_executed": attempt.metrics.configure_executed,
                    "configure_result": attempt.metrics.configure_result,
                    "configure_duration": attempt.metrics.configure_duration,
                    "execution_isolation_mode": (
                        attempt.metrics.execution_isolation_mode
                    ),
                    "execution_sandbox_adapter": (
                        attempt.metrics.execution_sandbox_adapter
                    ),
                    "execution_sandbox_available": (
                        attempt.metrics.execution_sandbox_available
                    ),
                    "execution_environment_hardened": (
                        attempt.metrics.execution_environment_hardened
                    ),
                    "execution_home_redirected": (
                        attempt.metrics.execution_home_redirected
                    ),
                    "execution_tmp_redirected": (
                        attempt.metrics.execution_tmp_redirected
                    ),
                    "execution_isolation_failure": (
                        attempt.metrics.execution_isolation_failure
                    ),
                    "verification_test_duration": (
                        attempt.metrics.verification_test_duration
                    ),
                    "verification_approval_requested": (
                        attempt.metrics.verification_approval_requested
                    ),
                    "verification_approved": attempt.metrics.verification_approved,
                    "model_calls": attempt.metrics.model_calls,
                    "tool_executions": attempt.metrics.tool_executions,
                    "elapsed_seconds": attempt.elapsed_seconds,
                },
            },
            sort_keys=True,
        )
    )
    return (
        0
        if attempt.final_status == "completed_verified"
        and attempt.oracle is EvaluationOutcome.PASS
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

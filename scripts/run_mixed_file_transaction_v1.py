"""Disposable A54 behavioral task and optional qwen-small production attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from forge.models import (
    GenerationConfig,
    Model,
    ModelRequest,
    ModelResponse,
    default_backend_registry,
    load_model_catalog,
)
from forge.orchestration import RepositoryChatSession
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)
from forge.tools.controlled_creation import authorize_create_candidate
from forge.tools.mixed_transaction import MixedFileTransactionTool, mixed_group_id
from forge.tools.tool import ToolError
from forge.tools.types import ExecutionContext

TASK = (
    "Update app.py so adjusted(value) clamps the input to 0 through 10 using "
    "a new helper.py module. Create helper.py with clamp(value, minimum, maximum), "
    "which returns the lower bound below range, upper bound above range, and "
    "the unchanged value inside range. Both app.py and helper.py must change."
)
REFERENCE_APP = (
    "from helper import clamp\n\ndef adjusted(value):\n    return clamp(value, 0, 10)\n"
)
REFERENCE_HELPER = (
    "def clamp(value, minimum, maximum):\n"
    "    return min(max(value, minimum), maximum)\n"
)


class _RecordingModel(Model):
    """Record envelope shape only; never persist generated source text."""

    def __init__(self, backend: Model) -> None:
        self.backend = backend
        self.envelopes: list[dict[str, object]] = []

    @property
    def identity(self):  # type: ignore[no-untyped-def]
        return self.backend.identity

    @property
    def capabilities(self):  # type: ignore[no-untyped-def]
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = self.backend.generate(request)
        try:
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("non-object")
            operations = payload.get("operations")
            self.envelopes.append(
                {
                    "type": payload.get("type"),
                    "children": tuple(
                        {"type": item.get("type"), "path": item.get("path")}
                        for item in operations
                        if isinstance(item, dict)
                    )
                    if isinstance(operations, list)
                    else (),
                }
            )
        except (ValueError, TypeError):
            self.envelopes.append({"type": "invalid_json"})
        return response

    def close(self) -> None:
        self.backend.close()


def _fixture(workspace: Path) -> None:
    (workspace / "tests").mkdir()
    (workspace / "app.py").write_text("def adjusted(value):\n    return value\n")
    (workspace / "tests" / "test_app.py").write_text(
        "import unittest\nfrom app import adjusted\n\n"
        "class AppTest(unittest.TestCase):\n"
        "    def test_bounds(self):\n"
        "        self.assertEqual(adjusted(-3), 0)\n"
        "        self.assertEqual(adjusted(12), 10)\n"
        "        self.assertEqual(adjusted(5), 5)\n"
    )


def _oracle(workspace: Path) -> bool:
    result = subprocess.run(
        (sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"),
        cwd=workspace,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(workspace),
        },
        shell=False,
        capture_output=True,
        timeout=20,
        check=False,
    )
    return result.returncode == 0


def _performance() -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="forge-a54-perf-") as name:
        workspace = Path(name)
        for path in ("a.py", "c.py"):
            (workspace / path).write_text("VALUE = 1\n")
        candidate = authorize_create_candidate(
            workspace, "b.py", 0, "evaluator_fixture"
        )
        context = ExecutionContext(
            workspace,
            create_candidates=(candidate,),
            edit_candidates=("a.py", "c.py"),
        )
        operations: tuple[dict[str, object], ...] = tuple(
            {"type": "create", "path": path, "content": "NEW = 1\n"}
            if path == "b.py"
            else {
                "type": "edit",
                "path": path,
                "expected_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
                "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
            }
            for path in ("a.py", "b.py", "c.py")
        )
        arguments = {
            "group_id": mixed_group_id(operations, (candidate,), 0, context.workspace),
            "workspace_generation": 0,
            "operations": operations,
        }

        def fail_last(source: Path, target: Path) -> None:
            if target.name == "c.py":
                raise OSError("injected failure")
            os.replace(source, target)

        try:
            MixedFileTransactionTool(replace_operation=fail_last).execute(
                arguments, context
            )
        except ToolError as error:
            rollback = float(error.output["rollback_seconds"])
        else:
            raise AssertionError("rollback sample did not fail")
        output = MixedFileTransactionTool().execute(arguments, context)
        return {
            key: float(output[key])
            for key in (
                "normalization_seconds",
                "materialization_seconds",
                "validation_seconds",
                "staging_seconds",
                "application_seconds",
            )
        } | {"rollback_seconds": rollback}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge-a54-mixed-") as name:
        workspace = Path(name)
        _fixture(workspace)
        baseline = _oracle(workspace)
        (workspace / "app.py").write_text(REFERENCE_APP)
        (workspace / "helper.py").write_text(REFERENCE_HELPER)
        reference = _oracle(workspace)
        (workspace / "app.py").write_text("def adjusted(value):\n    return value\n")
        (workspace / "helper.py").unlink()
        result: dict[str, object] = {
            "suite": "mixed-file-transaction-v1",
            "baseline_pass": baseline,
            "reference_pass": reference,
            "model": "not_run" if args.config is None else "qwen-small",
            "performance": _performance(),
        }
        if args.config is not None:
            catalog = load_model_catalog(args.config, default_backend_registry())
            model = _RecordingModel(catalog.create("qwen-small"))
            previews = []
            commands = ProjectCommands(
                test=ProjectCommand(
                    (sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"),
                    20,
                )
            )
            with model:
                session = RepositoryChatSession(
                    "qwen-small",
                    model,
                    workspace,
                    generation=GenerationConfig(
                        max_tokens=512, temperature=0.0, seed=42
                    ),
                    registry=create_assist_repository_registry(
                        commands, include_creation=True
                    ),
                    policy=create_assist_repository_policy(),
                    required_candidate_paths=("app.py",),
                    create_candidate_paths=("helper.py",),
                    mixed_file_operations=True,
                    approval_callback=lambda _invocation, preview: (
                        previews.append(preview) or True
                    ),
                    require_relevant_source=False,
                )
                try:
                    response = session.execute_task(TASK)
                    result.update(
                        {
                            "readiness": any(
                                item.tool_name == "repository.read_file"
                                and item.status == "success"
                                for item in response.tool_activity
                            ),
                            "proposal": any(
                                item.tool_name == "repository.apply_file_operations"
                                for item in response.tool_activity
                            ),
                            "preview": bool(previews),
                            "transaction": (workspace / "helper.py").exists(),
                            "generation": session._mutation_generation,
                            "verification": response.coding_task.test.status
                            if response.coding_task is not None
                            else "not_run",
                            "oracle_pass": _oracle(workspace),
                            "tool_count": len(response.tool_activity),
                            "model_calls": response.orchestration_steps,
                        }
                    )
                except Exception as error:
                    result["error_class"] = type(error).__name__
                    result["error"] = str(error)[:200]
                    result["envelopes"] = model.envelopes
                    result["tool_count"] = len(session.last_activity)
                    result["generation"] = session._mutation_generation
        print(json.dumps(result, sort_keys=True))
        return 0 if not baseline and reference else 1


if __name__ == "__main__":
    raise SystemExit(main())

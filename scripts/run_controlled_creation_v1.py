"""A53 disposable behavioral fixture and optional qwen-small production run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from forge.models import GenerationConfig, default_backend_registry, load_model_catalog
from forge.orchestration import RepositoryChatSession
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)
from forge.tools.controlled_creation import (
    CreateTextFilesTool,
    authorize_create_candidate,
    create_group_id,
)
from forge.tools.tool import ToolError
from forge.tools.types import ExecutionContext

TASK = (
    "Create helper.py with a clamp(value, minimum, maximum) function. The existing "
    "app.py imports it. clamp returns minimum below the range, maximum above it, "
    "and the original value inside it. Do not modify existing files."
)
REFERENCE = (
    "def clamp(value, minimum, maximum):\n"
    "    return min(max(value, minimum), maximum)\n"
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


def _fixture(workspace: Path) -> None:
    (workspace / "tests").mkdir()
    (workspace / "app.py").write_text(
        "from helper import clamp\n\n"
        "def adjusted(value):\n"
        "    return clamp(value, 0, 10)\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_app.py").write_text(
        "import unittest\nfrom app import adjusted\n\n"
        "class AppTest(unittest.TestCase):\n"
        "    def test_bounds(self):\n"
        "        self.assertEqual(adjusted(-3), 0)\n"
        "        self.assertEqual(adjusted(12), 10)\n"
        "        self.assertEqual(adjusted(5), 5)\n",
        encoding="utf-8",
    )


def _tool_microbenchmark() -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="forge-a53-perf-") as name:
        workspace = Path(name)
        candidate = authorize_create_candidate(
            workspace, "sample.py", 0, "evaluator_fixture"
        )
        creates = ({"path": "sample.py", "content": "VALUE = 1\n"},)
        output = CreateTextFilesTool().execute(
            {
                "group_id": create_group_id(creates, (candidate,), 0),
                "workspace_generation": 0,
                "creates": creates,
            },
            ExecutionContext(workspace, create_candidates=(candidate,)),
        )
        metrics = {
            key: float(output[key])
            for key in (
                "validation_seconds",
                "staging_seconds",
                "application_seconds",
                "rollback_seconds",
            )
        }
    with tempfile.TemporaryDirectory(prefix="forge-a53-rollback-perf-") as name:
        workspace = Path(name)
        candidates = tuple(
            authorize_create_candidate(workspace, path, 0, "evaluator_fixture")
            for path in ("a.py", "b.py")
        )
        creates = (
            {"path": "a.py", "content": "A = 1\n"},
            {"path": "b.py", "content": "B = 2\n"},
        )
        normal_create = CreateTextFilesTool()._create_operation

        def fail_second(path: Path, data: bytes) -> tuple[int, int]:
            if path.name == "b.py":
                raise OSError("evaluator-injected later-child failure")
            return normal_create(path, data)

        try:
            CreateTextFilesTool(create_operation=fail_second).execute(
                {
                    "group_id": create_group_id(creates, candidates, 0),
                    "workspace_generation": 0,
                    "creates": creates,
                },
                ExecutionContext(workspace, create_candidates=candidates),
            )
        except ToolError as error:
            metrics["handled_rollback_seconds"] = float(
                error.output["rollback_seconds"]
            )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", default="qwen-small")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge-a53-create-") as name:
        workspace = Path(name)
        _fixture(workspace)
        baseline = _oracle(workspace)
        reference = workspace / "helper.py"
        reference.write_text(REFERENCE, encoding="utf-8")
        reference_pass = _oracle(workspace)
        reference.unlink()
        result: dict[str, object] = {
            "suite": "controlled-file-creation-v1",
            "baseline_pass": baseline,
            "reference_pass": reference_pass,
            "model": args.model if args.config else "not_run",
            "tool_microbenchmark": _tool_microbenchmark(),
        }
        if args.config is not None:
            catalog = load_model_catalog(args.config, default_backend_registry())
            model = catalog.create(args.model)
            previews = []
            commands = ProjectCommands(
                test=ProjectCommand(
                    (sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"),
                    20,
                )
            )
            with model:
                session = RepositoryChatSession(
                    args.model,
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
                                item.tool_name == "repository.create_text_files"
                                for item in response.tool_activity
                            ),
                            "preview": bool(previews),
                            "transaction": reference.exists(),
                            "generation": session._mutation_generation,
                            "verification": (
                                response.coding_task.test.status
                                if response.coding_task is not None
                                else "not_run"
                            ),
                            "oracle_pass": _oracle(workspace)
                            if reference.exists()
                            else False,
                        }
                    )
                except Exception as error:
                    result["error_class"] = type(error).__name__
        print(json.dumps(result, sort_keys=True))
        return 0 if not baseline and reference_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

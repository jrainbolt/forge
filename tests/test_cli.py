from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge import __version__
from forge.cli import (
    _detect_current_project,
    _interactive_startup,
    _normalize_primary_command,
    main,
)
from forge.evaluation import EvaluationRun
from forge.models import MockModel, ModelSelectionError


def test_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Forge local-first AI coding assistant" in capsys.readouterr().out


@pytest.mark.parametrize(
    "source,expected",
    (
        (
            ["repo", "/tmp/project", "--model", "qwen-small"],
            [
                "chat",
                "--mode",
                "read",
                "--workspace",
                "/tmp/project",
                "--model",
                "qwen-small",
            ],
        ),
        (
            ["assist", "."],
            ["chat", "--mode", "assist", "--workspace", "."],
        ),
        (
            ["agent"],
            ["chat", "--mode", "agent", "--workspace", "."],
        ),
        (
            ["--verbose", "assist", "."],
            ["--verbose", "chat", "--mode", "assist", "--workspace", "."],
        ),
    ),
)
def test_primary_command_aliases_use_existing_chat_path(
    source: list[str], expected: list[str]
) -> None:
    assert _normalize_primary_command(source) == expected


def test_bare_noninteractive_forge_shows_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("forge.cli.sys.stdin.isatty", lambda: False)
    assert main([]) == 2
    output = capsys.readouterr().out
    assert "usage: forge" in output
    assert "forge assist [PATH]" in output


def test_current_project_detection_uses_markers_without_git_commands(
    tmp_path: Path,
) -> None:
    assert _detect_current_project(tmp_path) is None
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    assert _detect_current_project(tmp_path) == tmp_path.resolve()
    assert _interactive_startup(cwd=tmp_path) == [
        "chat",
        "--mode",
        "plan",
        "--workspace",
        str(tmp_path.resolve()),
    ]


def test_version_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"forge {__version__}"


def test_invalid_argument_is_controlled(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--not-a-forge-option"])

    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_python_module_entry_point_succeeds() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_path, existing_pythonpath))
        if existing_pythonpath
        else source_path
    )

    result = subprocess.run(
        [sys.executable, "-m", "forge", "--help"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: forge" in result.stdout


def test_chat_help_succeeds_without_constructing_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "forge.cli.load_model_catalog",
        lambda *_args: pytest.fail("help must not load configuration"),
    )
    with pytest.raises(SystemExit) as exit_info:
        main(["chat", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--model" in output
    assert "--workspace" in output
    assert "--assist" in output
    assert "--agent" in output
    assert "--mode" in output
    assert "--permissions" in output
    assert "trusted-exec" in output


def test_assist_and_agent_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                "forge.toml",
                "--assist",
                "--agent",
            ]
        )
    assert exit_info.value.code == 2
    assert "not allowed with argument --assist" in capsys.readouterr().err


def test_repair_requires_agent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert (
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                str(tmp_path / "forge.toml"),
                "--workspace",
                str(tmp_path),
                "--repair",
            ]
        )
        == 2
    )
    assert "--repair requires --agent" in caplog.text


def test_explicit_mode_conflicts_with_legacy_flags(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    result = main(
        [
            "chat",
            "--model",
            "fixture",
            "--config",
            str(tmp_path / "forge.toml"),
            "--workspace",
            str(tmp_path),
            "--mode",
            "agent",
            "--assist",
        ]
    )
    assert result == 2
    assert "--mode cannot be combined" in caplog.text


@pytest.mark.parametrize("option", ("--mode", "--permissions"))
def test_invalid_mode_or_permission_is_rejected_by_parser(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["chat", "--model", "fixture", option, "unknown"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dry_run_fails_closed_until_preview_execution_is_isolated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert (
        main(
            [
                "chat",
                "--config",
                str(tmp_path / "forge.toml"),
                "--workspace",
                str(tmp_path),
                "--mode",
                "dry-run",
            ]
        )
        == 2
    )
    assert "DRY_RUN is deferred" in caplog.text


def test_eval_help_succeeds_without_constructing_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "forge.cli.load_model_catalog",
        lambda *_args: pytest.fail("help must not load configuration"),
    )
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--model" in output
    assert "--suite" in output
    assert "--output" in output


def test_eval_requires_configuration_path(caplog: pytest.LogCaptureFixture) -> None:
    assert main(["eval", "--model", "fixture"]) == 2
    assert "eval requires --config or FORGE_CONFIG" in caplog.text


def test_eval_loads_model_once_writes_only_explicit_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MockModel(("unused",))
    workspace = tmp_path / "fixture"
    workspace.mkdir()
    output = tmp_path / "result.json"

    class Catalog:
        def create(self, name: str) -> MockModel:
            assert name == "fixture"
            return model

    class Runner:
        def __init__(self, profile: str, received: MockModel, root: Path) -> None:
            assert (profile, received, root) == ("fixture", model, workspace)

        def run(
            self, suite: str, tasks: object, *, suite_version: int
        ) -> EvaluationRun:
            assert suite == "coding-v1"
            assert tasks == ()
            assert suite_version == 1
            return EvaluationRun("coding-v1", 1, "fixture", (), 0.25)

    monkeypatch.setattr("forge.cli.load_suite", lambda _name: ())
    monkeypatch.setattr("forge.cli.fixture_workspace", lambda: workspace)
    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    monkeypatch.setattr("forge.cli.EvaluationRunner", Runner)
    assert (
        main(
            [
                "eval",
                "--model",
                "fixture",
                "--config",
                str(tmp_path / "forge.toml"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert model.closed
    assert json.loads(output.read_text())["schema_version"] == 1
    assert "Evaluation workspace:" in capsys.readouterr().out


def test_normal_chat_still_requires_explicit_model(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Catalog:
        def resolve_profile(self, _explicit: None, role: object) -> str:
            raise ModelSelectionError(
                f"no default model profile is configured for {role.value}"
            )

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    assert main(["chat", "--config", str(tmp_path / "forge.toml")]) == 2
    assert "no default model profile is configured for chat" in caplog.text


def test_chat_requires_configuration_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert main(["chat", "--model", "fixture"]) == 2
    assert "requires --config or FORGE_CONFIG" in caplog.text


def test_unknown_chat_profile_is_controlled(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingCatalog:
        def create(self, name: str) -> MockModel:
            raise ModelSelectionError(f"unknown model profile {name!r}")

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: MissingCatalog())
    assert (
        main(["chat", "--model", "absent", "--config", str(tmp_path / "forge.toml")])
        == 2
    )
    assert "unknown model profile 'absent'" in caplog.text


def test_chat_constructs_once_and_closes_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = MockModel(("unused",))
    construction_count = 0

    class Catalog:
        def create(self, name: str) -> MockModel:
            nonlocal construction_count
            construction_count += 1
            assert name == "fixture"
            return model

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    monkeypatch.setattr("forge.cli.run_repl", lambda _session, **_kwargs: 0)
    assert (
        main(["chat", "--model", "fixture", "--config", str(tmp_path / "forge.toml")])
        == 0
    )
    assert construction_count == 1
    assert model.closed


def test_chat_workspace_constructs_repository_session_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = MockModel(("unused",))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Catalog:
        def create(self, _name: str) -> MockModel:
            return model

    observed: dict[str, object] = {}

    def inspect_session(session: object, **_kwargs: object) -> int:
        observed["session"] = session
        observed["workspace"] = session.info.workspace  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    monkeypatch.setattr("forge.cli.run_repl", inspect_session)
    result = main(
        [
            "chat",
            "--model",
            "fixture",
            "--config",
            str(tmp_path / "forge.toml"),
            "--workspace",
            str(workspace),
        ]
    )
    assert result == 0
    assert observed["workspace"] == workspace.resolve()
    assert observed["session"].info.repository_mode is True  # type: ignore[attr-defined]
    assert model.closed


def test_repository_chat_rejects_missing_workspace_and_no_system(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = MockModel(("unused",))

    class Catalog:
        def create(self, _name: str) -> MockModel:
            return model

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    base = ["chat", "--model", "fixture", "--config", str(tmp_path / "forge.toml")]
    assert main([*base, "--workspace", str(tmp_path / "missing")]) == 2
    assert "workspace does not exist" in caplog.text
    caplog.clear()
    assert main([*base, "--workspace", str(tmp_path), "--no-system"]) == 2
    assert "cannot be used" in caplog.text


def test_assist_requires_workspace_and_is_explicitly_write_capable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MockModel(("unused",))

    class Catalog:
        def create(self, _name: str) -> MockModel:
            return model

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    config = str(tmp_path / "forge.toml")
    assert main(["chat", "--model", "fixture", "--config", config, "--assist"]) == 2
    assert "--assist requires --workspace" in caplog.text

    observed: dict[str, object] = {}

    def inspect(session: object, **_kwargs: object) -> int:
        observed["assist"] = session.info.assist_mode  # type: ignore[attr-defined]
        observed["tools"] = session.info.available_tools  # type: ignore[attr-defined]
        return 0

    second = MockModel(("unused",))

    class SecondCatalog:
        def create(self, _name: str) -> MockModel:
            return second

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: SecondCatalog())
    monkeypatch.setattr("forge.cli.run_repl", inspect)
    assert (
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                config,
                "--workspace",
                str(tmp_path),
                "--assist",
            ]
        )
        == 0
    )
    assert observed == {"assist": True, "tools": 16}
    assert second.closed


def test_agent_requires_workspace_and_selects_bounded_agent_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = iter((MockModel(("unused",)), MockModel(("unused",))))

    class Catalog:
        def create(self, _name: str) -> MockModel:
            return next(models)

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    config = str(tmp_path / "forge.toml")
    assert main(["chat", "--model", "fixture", "--config", config, "--agent"]) == 2
    assert "--agent requires --workspace" in caplog.text

    observed: dict[str, object] = {}

    def inspect(session: object, **_kwargs: object) -> int:
        info = session.info  # type: ignore[attr-defined]
        observed.update(
            agent=info.agent_mode,
            mode=info.autonomy_mode.value,
            tools=info.available_tools,
            iterations=info.iteration_limit,
            tool_limit=info.tool_limit,
        )
        return 0

    monkeypatch.setattr("forge.cli.run_repl", inspect)
    assert (
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                config,
                "--workspace",
                str(tmp_path),
                "--agent",
            ]
        )
        == 0
    )
    assert observed == {
        "agent": True,
        "mode": "agent",
        "tools": 16,
        "iterations": 16,
        "tool_limit": 12,
    }


def test_repair_flag_selects_explicit_larger_bounded_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = MockModel(("unused",))

    class Catalog:
        def create(self, _name: str) -> MockModel:
            return model

    observed: dict[str, object] = {}

    def inspect(session: object, **_kwargs: object) -> int:
        info = session.info  # type: ignore[attr-defined]
        observed.update(
            repair=info.repair_enabled,
            iterations=info.iteration_limit,
            tools=info.tool_limit,
        )
        return 0

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    monkeypatch.setattr("forge.cli.run_repl", inspect)
    assert (
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                str(tmp_path / "forge.toml"),
                "--workspace",
                str(tmp_path),
                "--agent",
                "--repair",
            ]
        )
        == 0
    )
    assert observed == {"repair": True, "iterations": 24, "tools": 18}


def test_explicit_mode_and_permission_profile_reach_session_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = MockModel(("unused",))

    class Catalog:
        project_commands = None

        def create(self, _name: str) -> MockModel:
            return model

    observed: dict[str, object] = {}

    def inspect(session: object, **_kwargs: object) -> int:
        info = session.info  # type: ignore[attr-defined]
        observed.update(
            mode=info.autonomy_mode.value,
            profile=info.permission_profile,
            read=info.read_permission.value,
            write=info.write_permission.value,
            test=info.test_permission.value,
        )
        return 0

    monkeypatch.setattr("forge.cli.load_model_catalog", lambda *_args: Catalog())
    monkeypatch.setattr("forge.cli.run_repl", inspect)
    assert (
        main(
            [
                "chat",
                "--model",
                "fixture",
                "--config",
                str(tmp_path / "forge.toml"),
                "--workspace",
                str(tmp_path),
                "--mode",
                "agent",
                "--permissions",
                "trusted-exec",
            ]
        )
        == 0
    )
    assert observed == {
        "mode": "agent",
        "profile": "trusted-exec",
        "read": "allow",
        "write": "ask",
        "test": "allow",
    }

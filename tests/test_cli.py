from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge import __version__
from forge.cli import main
from forge.evaluation import EvaluationRun
from forge.models import MockModel, ModelSelectionError


def test_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Forge local-first AI coding assistant" in capsys.readouterr().out


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


def test_chat_requires_explicit_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["chat", "--config", "forge.toml"])
    assert exit_info.value.code == 2
    assert "--model" in capsys.readouterr().err


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
    monkeypatch.setattr("forge.cli.run_repl", lambda _session: 0)
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

    def inspect_session(session: object) -> int:
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

    def inspect(session: object) -> int:
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
    assert observed == {"assist": True, "tools": 9}
    assert second.closed

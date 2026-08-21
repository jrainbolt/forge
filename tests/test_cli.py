from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge import __version__
from forge.cli import main
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
    assert "--model" in capsys.readouterr().out


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

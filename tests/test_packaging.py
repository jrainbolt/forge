from __future__ import annotations

import tomllib
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import forge
from forge.evaluation import fixture_workspace


def test_src_layout_discovery_and_extras_are_explicit() -> None:
    project = Path(__file__).resolve().parents[1]
    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert config["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert config["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert config["project"]["scripts"]["forge"] == "forge.cli:main"
    assert config["project"]["dynamic"] == ["version"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "forge.__version__"
    }
    assert config["project"]["dependencies"] == []
    assert any(
        item.startswith("llama-cpp-python")
        for item in config["project"]["optional-dependencies"]["llama"]
    )


def test_packaged_evaluation_fixture_is_a_package_resource() -> None:
    resource = files("forge.evaluation").joinpath(
        "fixtures", "eval_repo", "src", "tinyqueue", "retry.py"
    )
    assert resource.is_file()
    assert "class RetryPolicy" in resource.read_text(encoding="utf-8")
    workspace = fixture_workspace()
    assert workspace.is_dir()
    assert (workspace / "tests/test_retry.py").is_file()


def test_packaged_fixture_matches_historical_test_fixture() -> None:
    project = Path(__file__).resolve().parents[1]
    historical = project / "tests/fixtures/eval_repo"
    packaged = fixture_workspace()
    historical_files = {
        path.relative_to(historical): path.read_bytes()
        for path in historical.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    packaged_files = {
        path.relative_to(packaged): path.read_bytes()
        for path in packaged.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert packaged_files == historical_files


def test_package_and_distribution_version_authority_agree() -> None:
    assert version("forge-coding-assistant") == forge.__version__

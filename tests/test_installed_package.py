from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_editable_and_wheel_install_without_pythonpath(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    editable = tmp_path / "editable"
    _run((sys.executable, "-m", "venv", "--system-site-packages", str(editable)))
    editable_python = editable / "bin" / "python"
    _run(
        (
            str(editable_python),
            "-m",
            "pip",
            "install",
            "-e",
            str(project),
        ),
        environment=environment,
    )
    _assert_installed(editable, tmp_path, environment)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _run(
        (
            str(editable_python),
            "-m",
            "pip",
            "wheel",
            "--wheel-dir",
            str(wheelhouse),
            str(project),
        ),
        environment=environment,
    )
    wheel = next(wheelhouse.glob("forge_coding_assistant-*.whl"))
    _assert_wheel_contents(wheel)

    installed = tmp_path / "wheel-installed"
    _run((sys.executable, "-m", "venv", "--system-site-packages", str(installed)))
    installed_python = installed / "bin" / "python"
    _run(
        (
            str(installed_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            str(wheel),
        ),
        environment=environment,
    )
    _assert_installed(installed, tmp_path, environment)


def _assert_installed(
    environment_root: Path, working_directory: Path, environment: dict[str, str]
) -> None:
    python = environment_root / "bin" / "python"
    console = environment_root / "bin" / "forge"
    imported = _run(
        (
            str(python),
            "-c",
            (
                "import forge, forge.cli; "
                "print(forge.__file__); print(forge.cli.__file__)"
            ),
        ),
        cwd=working_directory,
        environment=environment,
    )
    assert "site-packages" in imported.stdout or "/src/forge/" in imported.stdout
    help_result = _run(
        (str(console), "--help"),
        cwd=working_directory,
        environment=environment,
    )
    assert "Forge local-first AI coding assistant" in help_result.stdout
    entry_point = _run(
        (
            str(python),
            "-c",
            (
                "import importlib.metadata as m; "
                "print(next(e.value for e in m.entry_points(group='console_scripts') "
                "if e.name == 'forge'))"
            ),
        ),
        cwd=working_directory,
        environment=environment,
    )
    assert entry_point.stdout.strip() == "forge.cli:main"


def _assert_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "forge/__init__.py" in names
    assert "forge/cli.py" in names
    assert "forge/evaluation/fixtures/eval_repo/src/tinyqueue/__init__.py" in names
    excluded = ("eval-results/", "benchmarks/", "checkpoints/", ".gguf")
    assert not any(any(marker in name for marker in excluded) for name in names)
    assert not any("Foundation" in name for name in names)


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

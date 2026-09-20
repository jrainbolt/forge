"""Inspect Forge wheels and sdists against the explicit runtime-resource policy."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

RUNTIME_EVAL_RESOURCE_FILES = frozenset(
    {
        "forge/evaluation/fixtures/eval_repo/README.md",
        "forge/evaluation/fixtures/eval_repo/src/tinyqueue/__init__.py",
        "forge/evaluation/fixtures/eval_repo/src/tinyqueue/models.py",
        "forge/evaluation/fixtures/eval_repo/src/tinyqueue/retry.py",
        "forge/evaluation/fixtures/eval_repo/src/tinyqueue/service.py",
        "forge/evaluation/fixtures/eval_repo/src/tinyqueue/storage.py",
        "forge/evaluation/fixtures/eval_repo/tests/test_retry.py",
        "forge/evaluation/fixtures/eval_repo/tests/test_service.py",
        "forge/evaluation/fixtures/eval_repo/tests/test_storage.py",
    }
)

REQUIRED = (
    frozenset(
        {
            "forge/__init__.py",
            "forge/__main__.py",
            "forge/cli.py",
            "forge/models/catalog.py",
            "forge/tools/executor.py",
            "forge/orchestration/repository_session.py",
            "forge/evaluation/runner.py",
            "forge/repository_analysis.py",
        }
    )
    | RUNTIME_EVAL_RESOURCE_FILES
)

FORBIDDEN_COMPONENTS = frozenset(
    {
        ".git",
        ".venv",
        ".forge-exec",
        "__pycache__",
        "benchmarks",
        "benchmark-repo",
        "benchmark_repo",
        "eval-results",
        "replay-data",
        "replay_data",
        "replay-results",
        "replays",
        "generated-tests",
        "generated-test",
        "generated_tests",
        "generated_test",
        "transactions",
    }
)


def archive_files(archive_path: Path) -> frozenset[str]:
    """Return distribution-relative file names from a wheel or source archive."""
    if archive_path.suffix == ".whl":
        with zipfile.ZipFile(archive_path) as archive:
            return frozenset(
                name for name in archive.namelist() if not name.endswith("/")
            )
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            names = set()
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                _, _, relative = member.name.partition("/")
                names.add(relative.removeprefix("src/"))
            return frozenset(names)
    raise ValueError("archive must be a .whl or .tar.gz distribution")


def content_violations(
    names: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Allow only the nine documented installed-evaluation resource files."""
    missing = tuple(sorted(REQUIRED - names))
    forbidden = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            name.startswith("/")
            or "\\" in name
            or ".." in parts
            or PurePosixPath(name).as_posix() != name
            or any(part in FORBIDDEN_COMPONENTS for part in parts)
            or ("fixtures" in parts and name not in RUNTIME_EVAL_RESOURCE_FILES)
            or ("eval_repo" in parts and name not in RUNTIME_EVAL_RESOURCE_FILES)
            or name.endswith(".gguf")
            or name.endswith("forge.toml")
            or name.startswith("tests/fixtures/")
            or (
                name.startswith("forge/evaluation/fixtures/")
                and name not in RUNTIME_EVAL_RESOURCE_FILES
            )
        ):
            forbidden.append(name)
    return missing, tuple(sorted(forbidden))


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: verify_wheel.py WHEEL_OR_SDIST [...]")
    failed = False
    for archive_path in map(Path, sys.argv[1:]):
        names = archive_files(archive_path)
        missing, forbidden = content_violations(names)
        if missing or forbidden:
            failed = True
            if missing:
                print(
                    f"{archive_path.name} missing: {', '.join(missing)}",
                    file=sys.stderr,
                )
            if forbidden:
                print(
                    f"{archive_path.name} forbidden: {', '.join(forbidden)}",
                    file=sys.stderr,
                )
        else:
            print(
                f"verified {archive_path.name}: {len(names)} files, "
                f"{archive_path.stat().st_size} bytes"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

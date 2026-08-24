"""Inspect one Forge wheel for required modules and forbidden local artifacts."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REQUIRED = {
    "forge/__init__.py",
    "forge/__main__.py",
    "forge/cli.py",
    "forge/models/catalog.py",
    "forge/tools/executor.py",
    "forge/orchestration/repository_session.py",
    "forge/evaluation/runner.py",
    "forge/repository_analysis.py",
    "forge/evaluation/fixtures/eval_repo/src/tinyqueue/retry.py",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel.py WHEEL")
    wheel = Path(sys.argv[1])
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED - names)
    forbidden = sorted(
        name
        for name in names
        if name.endswith(".gguf")
        or name.endswith("forge.toml")
        or "eval-results" in name
        or "/.git/" in name
        or "/.venv/" in name
        or "__pycache__" in name
    )
    if missing or forbidden:
        if missing:
            print(f"missing: {', '.join(missing)}", file=sys.stderr)
        if forbidden:
            print(f"forbidden: {', '.join(forbidden)}", file=sys.stderr)
        return 1
    print(f"verified {wheel.name}: {len(names)} files, {wheel.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

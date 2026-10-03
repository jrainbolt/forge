"""Independent focused oracles for the four additional A68 tasks."""

from __future__ import annotations

import subprocess
import sys

PATTERNS = {
    "F09": "factory_(steam_engine|steam_turbine)_tests",
    "F10": "factory_(assembler|refinery)_tests",
    "F11": "factory_(extractor|burner)_tests",
    "F12": "factory_(fluid|fluid_machine)_tests",
}


def main() -> int:
    pattern = PATTERNS.get(sys.argv[1]) if len(sys.argv) == 2 else None
    if pattern is None:
        return 2
    return subprocess.run(
        (
            "ctest",
            "--test-dir",
            "build",
            "--output-on-failure",
            "--timeout",
            "10",
            "-R",
            pattern,
        ),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

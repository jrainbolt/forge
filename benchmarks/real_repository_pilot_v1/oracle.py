"""Task-specific behavioral oracle over already-built Foundation workspaces."""

from __future__ import annotations

import subprocess
import sys

PATTERNS = {
    "F01": "factory_solar_tests",
    "F02": "factory_solar_tests",
    "F03": "factory_entity_tests",
    "F04": "factory_refinery_tests",
    "F05": "factory_(belt|storage|demolition)_tests",
    "F06": "factory_(splitter|inserter)_tests",
    "F07": "factory_(storage|telemetry|advanced_science)_tests",
    "F08": "factory_(snapshot|presentation)_tests",
}


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    pattern = PATTERNS.get(sys.argv[1])
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
        shell=False,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

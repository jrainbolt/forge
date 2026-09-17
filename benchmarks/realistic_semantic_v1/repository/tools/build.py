import subprocess
from pathlib import Path

BUILD = Path("build")
BUILD.mkdir(exist_ok=True)
cases = {
    "parser": "parser.c",
    "quota": "quota.c",
    "window": "window.c",
    "status": "status.c",
}
for name, source in cases.items():
    completed = subprocess.run(
        (
            "/usr/bin/cc",
            "-std=c17",
            "-Wall",
            "-Werror",
            f"cengine/{source}",
            f"cengine/tests/test_{name}.c",
            "-o",
            str(BUILD / f"test_{name}"),
        ),
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)

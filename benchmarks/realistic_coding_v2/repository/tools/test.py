import os
import subprocess
import sys
from pathlib import Path

environment = dict(os.environ)
environment["PYTHONPATH"] = str(Path.cwd())

for test in sorted(Path("tests").glob("test_*.py")):
    completed = subprocess.run(
        (sys.executable, str(test)), env=environment, check=False, shell=False
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
for test in sorted(Path("build").glob("test_*")):
    completed = subprocess.run((str(test.resolve()),), check=False, shell=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

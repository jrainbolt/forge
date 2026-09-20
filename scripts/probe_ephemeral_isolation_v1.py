"""Host-only real Seatbelt probes for ephemeral-acceptance-isolation-v1."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from forge.ephemeral_acceptance import (
    EphemeralAcceptanceCandidate,
    EphemeralExecutionClass,
    _candidate_valid,
    _execute_python_candidate,
)
from forge.process_isolation import platform_ephemeral_test_sandbox


def main() -> int:
    sandbox = platform_ephemeral_test_sandbox()
    if not sandbox.available():
        print(
            json.dumps(
                {"suite": "ephemeral-acceptance-isolation-v1", "host": "unavailable"}
            )
        )
        return 2
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="forge-a52-probe-") as name:
        root = Path(name)
        workspace = root / "workspace"
        package = workspace / "pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        source = package / "value.py"
        source.write_text("def value():\n    return 1\n")
        sibling = root / "sibling"
        sibling.mkdir()
        home_like = root / "home-like"
        home_like.mkdir()
        secret = home_like / "secret"
        secret.write_text("probe-only")
        real_home_read_target = Path.cwd() / "README.md"
        if not real_home_read_target.is_file():
            raise RuntimeError("real-home read target is unavailable")
        os.environ["FORGE_TEST_SECRET"] = "fake-ambient-secret"
        modules = {
            "python": "assert True\n",
            "benign": "from pkg.value import value\nVALUE = value() == 1\n",
            "temp_write": (
                "import os\nfrom pathlib import Path\n"
                "Path(os.environ['TMPDIR'], 'allowed').write_text('ok')\n"
                "VALUE = 1\n"
            ),
            "project_write": (
                "from pathlib import Path\n"
                "Path('pkg/value.py').write_text('changed')\nVALUE = 1\n"
            ),
            "sibling_write": (
                "from pathlib import Path\n"
                f"Path({str(sibling / 'blocked')!r}).write_text('bad')\nVALUE = 1\n"
            ),
            "home_write": (
                "from pathlib import Path\n"
                f"Path({str(home_like / 'blocked')!r}).write_text('bad')\nVALUE = 1\n"
            ),
            "home_read": (
                "from pathlib import Path\n"
                f"Path({str(secret)!r}).read_text()\nVALUE = 1\n"
            ),
            "real_home_read": (
                "from pathlib import Path\n"
                f"Path({str(real_home_read_target)!r}).read_text()\nVALUE = 1\n"
            ),
            "real_home_write_open": (
                f"with open({str(real_home_read_target)!r}, 'r+') as stream:\n"
                "    stream.read(1)\nVALUE = 1\n"
            ),
            "network": (
                "import socket\n"
                "s = socket.socket()\ns.bind(('127.0.0.1', 0))\nVALUE = 1\n"
            ),
            "subprocess": (
                "import subprocess\nsubprocess.run(('/usr/bin/true',))\nVALUE = 1\n"
            ),
            "subprocess_python": (
                "import subprocess, sys\n"
                "subprocess.run((sys.executable, '-S', '-c', 'pass'), check=True)\n"
                "VALUE = 1\n"
            ),
            "environment": (
                "import os\nVALUE = os.environ.get('FORGE_TEST_SECRET') is None\n"
            ),
            "synthetic_home": (
                f"import os\nVALUE = os.environ['HOME'] != {str(Path.home())!r}\n"
            ),
            "symlink_escape": (
                "import os\nfrom pathlib import Path\n"
                f"os.symlink({str(sibling)!r}, Path(os.environ['TMPDIR'], 'link'))\n"
                "Path(os.environ['TMPDIR'], 'link', 'blocked').write_text('bad')\n"
                "VALUE = 1\n"
            ),
        }
        for name, body in modules.items():
            if name != "python":
                (package / f"{name}.py").write_text(body)
            candidate = EphemeralAcceptanceCandidate(
                name,
                "assert True\n"
                if name == "python"
                else f"from pkg.{name} import VALUE\nassert VALUE\n",
            )
            if not _candidate_valid(candidate, "pkg"):
                raise RuntimeError(f"probe candidate failed static validation: {name}")
            outcome = _execute_python_candidate(
                candidate, workspace, sandbox, ("pkg/value.py",)
            )
            results[name] = outcome.classification.value
            if name == "python":
                results["python_diagnostic"] = outcome.output[:500]
                results["python_exit_code"] = str(outcome.exit_code)
                results["python_pid"] = str(outcome.pid)
                results["preparation_seconds"] = f"{outcome.preparation_seconds:.4f}"
                results["python_total_seconds"] = f"{outcome.duration_seconds:.4f}"
        results["source_unchanged"] = str(
            source.read_text() == "def value():\n    return 1\n"
        )
        results["sibling_unchanged"] = str(not (sibling / "blocked").exists())
        results["home_unchanged"] = str(not (home_like / "blocked").exists())
        results["symlink_unchanged"] = str(not (sibling / "blocked").exists())
        results["policy_id"] = sandbox.identity
    os.environ.pop("FORGE_TEST_SECRET", None)
    print(json.dumps(results, indent=2, sort_keys=True))
    passing = {"python", "benign", "temp_write", "environment", "synthetic_home"}
    denied = {
        "project_write",
        "sibling_write",
        "home_write",
        "home_read",
        "real_home_read",
        "real_home_write_open",
        "network",
        "subprocess",
        "subprocess_python",
        "symlink_escape",
    }
    if any(results[item] != EphemeralExecutionClass.PASS.value for item in passing):
        return 1
    if any(
        results[item] != EphemeralExecutionClass.SANDBOX_VIOLATION.value
        for item in denied
    ):
        return 1
    if any(
        results[key] != "True"
        for key in (
            "source_unchanged",
            "sibling_unchanged",
            "home_unchanged",
            "symlink_unchanged",
        )
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

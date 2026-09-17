"""Evaluator-only behavioral and regression-sensitivity oracle for A45."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "repository"

DEFECTS = {
    "R05": (
        "pyservice/state.py",
        "JobState.FAILED: frozenset(),",
        "JobState.FAILED: frozenset({JobState.RUNNING}),",
        "tests/test_state.py",
    ),
    "R06": (
        "pyservice/headers.py",
        're.compile(r"^[A-Za-z0-9-]+$")',
        're.compile(r"^[A-Za-z0-9_-]+$")',
        "tests/test_headers.py",
    ),
    "R07": (
        "cengine/window.c",
        "return limit > 0 && events < limit;",
        "return limit > 0 && events <= limit;",
        "cengine/tests/test_window.c",
    ),
    "R08": (
        "cengine/status.c",
        "return status;\n    }\n    return ENGINE_INVALID;",
        "return status == ENGINE_IO_ERROR ? ENGINE_RETRY : status;\n"
        "    }\n    return ENGINE_INVALID;",
        "cengine/tests/test_status.c",
    ),
}


def _run(
    command: tuple[str, ...], cwd: Path, *, pythonpath: Path | None = None
) -> bool:
    environment = dict(os.environ)
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
        shell=False,
    )
    return result.returncode == 0


def _python_hidden(task_id: str, workspace: Path) -> bool:
    checks = {
        "R01": (
            "from pyservice.retry import RetryPolicy,should_retry; p=RetryPolicy(3); "
            "assert should_retry(0,429,p); assert should_retry(1,503,p); "
            "assert not should_retry(2,500,p); assert not should_retry(0,409,p)"
        ),
        "R02": (
            "from pyservice.config import parse_enabled; "
            "assert parse_enabled({'X':'TRUE'},'X',default=False); "
            "assert not parse_enabled({'X':' no '},'X',default=True); "
            "assert not parse_enabled({'X':''},'X',default=True); "
            "assert parse_enabled({},'X',default=True); "
            "\ntry: parse_enabled({'X':'sometimes'},'X',default=False); assert False\n"
            "except ValueError: pass"
        ),
        "R05": (
            "from pyservice.state import JobState,can_transition; "
            "assert can_transition(JobState.QUEUED,JobState.RUNNING); "
            "assert not can_transition(JobState.FAILED,JobState.RUNNING); "
            "assert not can_transition(JobState.SUCCEEDED,JobState.CANCELLED)"
        ),
        "R06": (
            "from pyservice.headers import normalize_header_name; "
            "assert normalize_header_name(' X-9 ')=='x-9'; "
            "\nfor value in ('x_header','bad header','x:y'):\n"
            "  try: normalize_header_name(value); assert False\n"
            "  except ValueError: pass"
        ),
    }
    return _run(
        (sys.executable, "-c", checks[task_id]), workspace, pythonpath=workspace
    )


def _c_hidden(task_id: str, workspace: Path) -> bool:
    sources = {
        "R03": (
            "cengine/parser.c",
            '#include "parser.h"\n#include <assert.h>\nint main(void){int p=0;'
            'assert(parse_port("443",&p)&&p==443);'
            'assert(!parse_port("443x",&p));assert(!parse_port("  ",&p));'
            'assert(!parse_port("65536",&p));return 0;}\n',
        ),
        "R04": (
            "cengine/quota.c",
            '#include "quota.h"\n#include <assert.h>\nint main(void){quota q={2,9};'
            "quota_release(&q,7);assert(q.used==0);"
            "assert(quota_reserve(&q,9));quota_release(&q,9);"
            "assert(q.used==0);return 0;}\n",
        ),
        "R07": (
            "cengine/window.c",
            '#include "window.h"\n#include <assert.h>\nint main(void){'
            "assert(window_accepts(4,5));assert(!window_accepts(5,5));"
            "assert(!window_accepts(6,5));assert(!window_accepts(0,0));return 0;}\n",
        ),
        "R08": (
            "cengine/status.c",
            '#include "status.h"\n#include <assert.h>\nint main(void){'
            "assert(normalize_dependency_status(ENGINE_IO_ERROR)==ENGINE_IO_ERROR);"
            "assert(normalize_dependency_status(ENGINE_RETRY)==ENGINE_RETRY);"
            "assert(normalize_dependency_status((engine_status)77)==ENGINE_INVALID);"
            "return 0;}\n",
        ),
    }
    implementation, hidden_source = sources[task_id]
    with tempfile.TemporaryDirectory(prefix="forge-a45-hidden-c-") as name:
        temporary = Path(name).resolve()
        hidden = temporary / "hidden.c"
        hidden.write_text(hidden_source, encoding="utf-8")
        executable = temporary / "hidden"
        include = workspace / Path(implementation).parent
        if not _run(
            (
                "/usr/bin/cc",
                "-std=c17",
                "-Wall",
                "-Werror",
                "-I",
                str(include),
                str(workspace / implementation),
                str(hidden),
                "-o",
                str(executable),
            ),
            workspace,
        ):
            return False
        return _run((str(executable),), workspace)


def _regression_detects_defect(task_id: str, workspace: Path) -> bool:
    implementation, correct, broken, test_path = DEFECTS[task_id]
    with tempfile.TemporaryDirectory(prefix="forge-a45-mutant-") as name:
        mutant = Path(name).resolve() / "repository"
        shutil.copytree(CANONICAL, mutant)
        source = (mutant / implementation).read_text(encoding="utf-8")
        if source.count(correct) != 1:
            return False
        (mutant / implementation).write_text(
            source.replace(correct, broken, 1), encoding="utf-8"
        )
        (mutant / test_path).write_bytes((workspace / test_path).read_bytes())
        if task_id in {"R05", "R06"}:
            # A good submitted regression must fail against the known defect.
            return not _run((sys.executable, test_path), mutant, pythonpath=mutant)
        executable = mutant / "mutant-test"
        compiled = _run(
            (
                "/usr/bin/cc",
                "-std=c17",
                "-Wall",
                "-Werror",
                implementation,
                test_path,
                "-o",
                str(executable),
            ),
            mutant,
        )
        return compiled and not _run((str(executable),), mutant)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {f"R{i:02d}" for i in range(1, 9)}:
        return 2
    task_id = sys.argv[1]
    workspace = Path.cwd().resolve()
    hidden = (
        _python_hidden(task_id, workspace)
        if task_id in {"R01", "R02", "R05", "R06"}
        else _c_hidden(task_id, workspace)
    )
    if not hidden:
        return 1
    if task_id in DEFECTS and not _regression_detects_defect(task_id, workspace):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

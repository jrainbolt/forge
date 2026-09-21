"""Evaluator-held behavioral oracle for realistic-coding-v2."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A45_ORACLE = ROOT.parent / "realistic_semantic_v1" / "oracle.py"
LEGACY = {
    "C01": "R01",
    "C02": "R03",
    "C03": "R04",
    "C04": "R05",
    "C05": "R07",
    "C06": "R08",
}

PYTHON_CHECKS = {
    "C07": (
        "from pyservice.scheduler import schedule; "
        "from pyservice.priority import priority_rank; "
        "assert priority_rank(' high ') == 0; "
        "assert schedule([('a','low'),('b','HIGH'),('c','normal'),"
        "('d','high')]) == ['b','d','c','a']; "
        "\ntry: schedule([('z','urgent')]); assert False\n"
        "except ValueError: pass"
    ),
    "C08": (
        "from pyservice.models import Response; "
        "from pyservice.response_wire import wire_response; "
        "from pyservice.serialization import encode_response; "
        "assert encode_response(201,b'abc') == b'201:3:abc'; "
        "assert wire_response(Response(200,b'\\x00A')) == b'200:2:\\x00A'; "
        "assert wire_response(Response(204,b'')) == b'204:0:'; "
        "\ntry: wire_response(Response(600,b'')); assert False\n"
        "except ValueError: pass"
    ),
    "C10": (
        "from pyservice.dispatch import dispatch; "
        "from pyservice.dispatch_rules import choose_handler; "
        "assert choose_handler('read') == 'reader'; "
        "assert dispatch(' READ ') == 'reader'; "
        "assert dispatch('Write') == 'writer'; "
        "assert dispatch('delete') == 'remover'; "
        "\nfor value in (None, '\\t', 'unknown'):\n"
        "  try: dispatch(value); assert False\n"
        "  except ValueError: pass"
    ),
    "C11": (
        "from pyservice.billing import invoice_line; "
        "from pyservice.currency import format_cents; "
        "assert format_cents(105) == '$1.05'; "
        "assert invoice_line(0) == 'TOTAL $0.00'; "
        "assert invoice_line(101) == 'TOTAL $1.01'; "
        "assert invoice_line(9999) == 'TOTAL $99.99'; "
        "\ntry: invoice_line(-1); assert False\n"
        "except ValueError: pass"
    ),
}

C_CHECKS = {
    "C09": (
        ("backoff.c", "retry_policy.c"),
        '#include "backoff.h"\n#include <assert.h>\n'
        "unsigned retry_wait(unsigned);\nint main(void){"
        "assert(retry_wait(0)==2);assert(retry_wait(1)==4);"
        "assert(retry_wait(4)==30);assert(retry_wait(20)==30);"
        "assert(backoff_delay(3,1,20)==8);"
        "assert(backoff_delay(2,40,30)==30);return 0;}\n",
    ),
    "C12": (
        ("health.c", "health_policy.c"),
        '#include "health_policy.h"\n#include <assert.h>\n'
        "int health_status(unsigned,int);\nint main(void){"
        "assert(health_status(0,0)==-1);assert(health_status(0,-1)==-1);"
        "assert(health_status(0,3)==1);assert(health_status(3,3)==0);"
        "assert(health_status(4,3)==0);"
        "assert(health_is_healthy(2,3)==1);return 0;}\n",
    ),
}


def _run(argv: tuple[str, ...], workspace: Path, *, python: bool = False) -> bool:
    environment = dict(os.environ)
    if python:
        environment["PYTHONPATH"] = str(workspace)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            argv,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def check(task_id: str, workspace: Path) -> bool:
    if task_id in LEGACY:
        return _run((sys.executable, str(A45_ORACLE), LEGACY[task_id]), workspace)
    if task_id in PYTHON_CHECKS:
        return _run(
            (sys.executable, "-B", "-c", PYTHON_CHECKS[task_id]), workspace, python=True
        )
    if task_id in C_CHECKS:
        sources, source = C_CHECKS[task_id]
        with tempfile.TemporaryDirectory(prefix="forge-a56-hidden-c-") as name:
            temporary = Path(name).resolve()
            probe = temporary / "probe.c"
            probe.write_text(source, encoding="utf-8")
            executable = temporary / "probe"
            if not _run(
                (
                    "/usr/bin/cc",
                    "-std=c17",
                    "-Wall",
                    "-Werror",
                    "-I",
                    str(workspace / "cengine"),
                    *(str(workspace / "cengine" / path) for path in sources),
                    str(probe),
                    "-o",
                    str(executable),
                ),
                workspace,
            ):
                return False
            return _run((str(executable),), workspace)
    raise ValueError(f"unknown A56 task: {task_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(2)
    raise SystemExit(0 if check(sys.argv[1], Path.cwd().resolve()) else 1)

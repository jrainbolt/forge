"""Evaluator-held behavioral oracle for the frozen A59 task corpus."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON_CHECKS = {
    "Q01": (
        "from pyservice.retry import *; p=RetryPolicy(3,25); "
        "assert should_retry(0,500,p); assert should_retry(1,429,p); "
        "assert not should_retry(2,500,p); assert not should_retry(0,499,p); "
        "assert retry_delay_ms(3,p)==200"
    ),
    "Q03": (
        "from pyservice.headers import normalize_header_name; "
        "from pyservice.models import Request; from pyservice.service import handle; "
        "assert normalize_header_name(' X-Id ')=='x-id'; "
        "assert handle(Request('',b'x')).status==400; "
        "assert handle(Request('r',b'')).status==204"
    ),
    "Q05": (
        "from pyservice.billing import invoice_line; "
        "from pyservice.currency import format_cents; "
        "assert format_cents(5)=='$0.05'; assert format_cents(1205)=='$12.05'; "
        "assert invoice_line(101)=='TOTAL $1.01'"
    ),
    "Q07": (
        "from pyservice.dispatch import dispatch; "
        "from pyservice.dispatch_rules import choose_handler; "
        "assert dispatch(' READ ')=='reader'; assert dispatch('Write')=='writer'; "
        "assert choose_handler('delete')=='remover'; "
        "\ntry: dispatch('unknown'); assert False\nexcept ValueError: pass"
    ),
}

C_CHECKS = {
    "Q02": (
        ("quota.c",),
        '#include "quota.h"\n#include <assert.h>\nint main(void){quota q={2,5};'
        "assert(quota_reserve(&q,3));assert(q.used==5);"
        "assert(!quota_reserve(&q,1));quota_release(&q,9);"
        "assert(q.used==0);return 0;}\n",
    ),
    "Q04": (
        ("window.c", "status.c"),
        '#include "window.h"\n#include "status.h"\n#include <assert.h>\nint main(void){'
        "assert(window_accepts(2,3));assert(!window_accepts(3,3));"
        "assert(normalize_dependency_status(ENGINE_RETRY)==ENGINE_RETRY);"
        "assert(normalize_dependency_status((engine_status)99)==ENGINE_INVALID);"
        "return 0;}\n",
    ),
    "Q06": (
        ("health_policy.c",),
        '#include "health_policy.h"\n#include <assert.h>\nint main(void){'
        "assert(health_is_healthy(2,3)==1);"
        "assert(health_is_healthy(3,3)==0);return 0;}\n",
    ),
    "Q08": (
        ("backoff.c", "retry_policy.c"),
        '#include "backoff.h"\n#include <assert.h>\nunsigned retry_wait(unsigned);'
        "int main(void){assert(retry_wait(0)==2);assert(retry_wait(1)==4);"
        "assert(retry_wait(4)==30);assert(backoff_delay(3,1,20)==8);return 0;}\n",
    ),
}


def _run(argv: tuple[str, ...], workspace: Path, *, python: bool = False) -> bool:
    environment = dict(os.environ)
    if python:
        environment["PYTHONPATH"] = str(workspace)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
    return result.returncode == 0


def check(task_id: str, workspace: Path) -> bool:
    if task_id in PYTHON_CHECKS:
        return _run(
            (sys.executable, "-B", "-c", PYTHON_CHECKS[task_id]),
            workspace,
            python=True,
        )
    sources, source = C_CHECKS[task_id]
    with tempfile.TemporaryDirectory(prefix="forge-a59-oracle-") as name:
        probe = Path(name) / "probe.c"
        probe.write_text(source, encoding="utf-8")
        executable = Path(name) / "probe"
        if not _run(
            (
                "/usr/bin/cc",
                "-std=c17",
                "-Wall",
                "-Werror",
                "-I",
                str(workspace / "cengine"),
                *(str(workspace / "cengine" / item) for item in sources),
                str(probe),
                "-o",
                str(executable),
            ),
            workspace,
        ):
            return False
        return _run((str(executable),), workspace)


if __name__ == "__main__":
    raise SystemExit(0 if check(sys.argv[1], Path.cwd().resolve()) else 1)

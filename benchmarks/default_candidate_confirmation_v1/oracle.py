"""Evaluator-held semantic oracle for the A78 held-out corpus."""
# ruff: noqa: E501

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = {
    "metrics": "from pyservice.metrics import Counter; c=Counter(); assert c.value==0; c.increment(0); c.increment(); c.increment(4); assert c.value==5;\ntry: c.increment(-1); assert False\nexcept ValueError: pass",
    "storage": "from pyservice.storage import ResponseStore; from pyservice.models import Response; s=ResponseStore(); assert s.get('x') is None; s.put('x',Response(200,b'a')); s.put('x',Response(201,b'b')); assert s.get('x')==Response(201,b'b')",
    "config": "from pyservice.config import parse_enabled,parse_timeout; assert parse_enabled({},'X',default=True); assert parse_enabled({'X':' YES '},'X',default=False); assert not parse_enabled({'X':'off'},'X',default=True); assert parse_timeout({'T':'2'},'T',1)==2;\nfor v in ('0','-1'):\n try: parse_timeout({'T':v},'T',1); assert False\n except ValueError: pass\ntry: parse_enabled({'X':'maybe'},'X',default=False); assert False\nexcept ValueError: pass",
    "headers": "from pyservice.headers import normalize_header_name; assert normalize_header_name(' X-Trace ')=='x-trace';\nfor v in ('','bad name'):\n try: normalize_header_name(v); assert False\n except ValueError: pass",
    "queue_service": "from pyservice.queue import RequestQueue; from pyservice.models import Request; from pyservice.service import handle; q=RequestQueue(); a=Request('a',b'1'); b=Request('b',b'2'); q.push(a);q.push(b);assert q.pop()==a;assert q.pop()==b;assert q.pop() is None; assert handle(Request('',b'')).status==400; assert handle(Request('x',b'')).status==204",
    "state": "from pyservice.state import JobState,can_transition; assert can_transition(JobState.QUEUED,JobState.RUNNING); assert can_transition(JobState.QUEUED,JobState.CANCELLED); assert not can_transition(JobState.SUCCEEDED,JobState.RUNNING)",
}


def _run(argv: tuple[str, ...], workspace: Path, *, python: bool = False) -> bool:
    env = dict(os.environ)
    if python:
        env.update(PYTHONPATH=str(workspace), PYTHONDONTWRITEBYTECODE="1")
    try:
        return (
            subprocess.run(
                argv,
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _python(workspace: Path, *names: str) -> bool:
    return _run(
        (sys.executable, "-B", "-c", "\n".join(PYTHON[name] for name in names)),
        workspace,
        python=True,
    )


def _c(workspace: Path, sources: tuple[str, ...], probe_source: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="forge-a78-oracle-") as name:
        probe = Path(name) / "probe.c"
        probe.write_text(probe_source)
        output = Path(name) / "probe"
        command = (
            "/usr/bin/cc",
            "-std=c17",
            "-Wall",
            "-Werror",
            "-I",
            str(workspace / "cengine"),
            *(str(workspace / "cengine" / source) for source in sources),
            str(probe),
            "-o",
            str(output),
        )
        return _run(command, workspace) and _run((str(output),), workspace)


def check(task: str, workspace: Path) -> bool:
    if task in {"H01", "H07"}:
        return _python(workspace, "metrics")
    if task == "H02":
        return _python(workspace, "storage")
    if task == "H04":
        return _python(workspace, "config", "headers")
    if task == "H05":
        return _python(workspace, "queue_service")
    if task == "H09":
        return _python(workspace, "config")
    if task == "H10":
        return _python(workspace, "state", "metrics")
    if task == "H12":
        return _python(workspace, "storage", "config")
    if task == "H03":
        return _c(
            workspace,
            ("clock.c",),
            '#include "clock.h"\n#include <assert.h>\nint main(void){assert(clock_elapsed(5,9)==4);assert(clock_elapsed(9,5)==0);return 0;}',
        )
    if task == "H06":
        return _c(
            workspace,
            ("parser.c", "quota.c"),
            '#include "parser.h"\n#include "quota.h"\n#include <assert.h>\nint main(void){int p=7;assert(!parse_port("0",&p));assert(parse_port("65535",&p)&&p==65535);quota q={2,5};assert(quota_reserve(&q,3)&&q.used==5);return 0;}',
        )
    checksum = '#include "checksum.h"\n#include <assert.h>\nint main(void){const unsigned char d[]={1,2,3};assert(checksum_bytes(d,3)==((1U*33U^2U)*33U^3U));assert(checksum_bytes(d,0)==0);return 0;}'
    if task == "H08":
        return _c(workspace, ("checksum.c",), checksum)
    if task == "H11":
        return _c(
            workspace,
            ("window.c", "checksum.c"),
            '#include "window.h"\n#include <assert.h>\nint main(void){assert(!window_accepts(0,0));assert(window_accepts(2,3));assert(!window_accepts(3,3));return 0;}',
        ) and _c(workspace, ("checksum.c",), checksum)
    raise ValueError(f"unknown A78 task: {task}")


if __name__ == "__main__":
    raise SystemExit(0 if len(sys.argv) == 2 and check(sys.argv[1], Path.cwd()) else 1)

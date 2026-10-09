"""Transient per-obligation checks derived only from stated task behavior."""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.oracle import _c, _run


def _python(workspace: Path, source: str) -> bool:
    import sys

    return _run((sys.executable, "-B", "-c", source), workspace, python=True)


PYTHON_CHECKS = {
    "initial-zero": "from pyservice.metrics import Counter; assert Counter().value == 0",
    "accumulate": "from pyservice.metrics import Counter; c=Counter(); c.increment(0); c.increment(); c.increment(4); assert c.value==5",
    "reject-negative": "from pyservice.metrics import Counter; c=Counter();\ntry: c.increment(-1); assert False\nexcept ValueError: pass",
    "boolean-forms": "from pyservice.config import parse_enabled; assert parse_enabled({'X':' YES '},'X',default=False); assert not parse_enabled({'X':'off'},'X',default=True);\ntry: parse_enabled({'X':'maybe'},'X',default=False); assert False\nexcept ValueError: pass",
    "absent-default": "from pyservice.config import parse_enabled; assert parse_enabled({},'X',default=True); assert not parse_enabled({},'X',default=False)",
    "positive-timeout": "from pyservice.config import parse_timeout; assert parse_timeout({'T':'2'},'T',1)==2",
    "reject-nonpositive": "from pyservice.config import parse_timeout;\nfor v in ('0','-1'):\n try: parse_timeout({'T':v},'T',1); assert False\n except ValueError: pass",
    "queued-running": "from pyservice.state import JobState,can_transition; assert can_transition(JobState.QUEUED,JobState.RUNNING)",
    "queued-cancelled": "from pyservice.state import JobState,can_transition; assert can_transition(JobState.QUEUED,JobState.CANCELLED)",
    "counter-accumulate": "from pyservice.metrics import Counter; c=Counter(); c.increment(); c.increment(4); assert c.value==5",
    "counter-negative": "from pyservice.metrics import Counter; c=Counter();\ntry: c.increment(-1); assert False\nexcept ValueError: pass",
    "storage-replace": "from pyservice.storage import ResponseStore; from pyservice.models import Response; s=ResponseStore(); s.put('x',Response(200,b'a')); s.put('x',Response(201,b'b')); assert s.get('x')==Response(201,b'b')",
    "accept-zero": "from pyservice.metrics import Counter; c=Counter(); c.increment(0); assert c.value==0",
    "header-normalize": "from pyservice.headers import normalize_header_name; assert normalize_header_name(' X-Trace ')=='x-trace'",
    "header-validate": "from pyservice.headers import normalize_header_name;\ntry: normalize_header_name('bad name'); assert False\nexcept ValueError: pass",
}


def check(check_id: str, workspace: Path) -> bool | None:
    name = check_id.split("-", 1)[1]
    if name in PYTHON_CHECKS:
        return _python(workspace, PYTHON_CHECKS[name])
    if name in {
        "header-interface",
        "rolling-algorithm",
        "all-bytes",
        "checksum-interface",
        "checksum-algorithm",
    }:
        if name in {"header-interface", "checksum-interface"}:
            probe = '#include "checksum.h"\nint main(void){const unsigned char d[]={1};(void)checksum_bytes(d,1);return 0;}'
        elif name == "all-bytes":
            probe = '#include "checksum.h"\n#include <assert.h>\nint main(void){const unsigned char a[]={1,2};const unsigned char b[]={1,3};assert(checksum_bytes(a,2)!=checksum_bytes(b,2));return 0;}'
        else:
            probe = '#include "checksum.h"\n#include <assert.h>\nint main(void){const unsigned char d[]={1,2,3};assert(checksum_bytes(d,3)==((1U*33U^2U)*33U^3U));assert(checksum_bytes(d,0)==0);return 0;}'
        return _c(workspace, ("checksum.c",), probe)
    if name in {"positive-limit", "strictly-below"}:
        probe = (
            '#include "window.h"\n#include <assert.h>\nint main(void){assert(!window_accepts(0,0));return 0;}'
            if name == "positive-limit"
            else '#include "window.h"\n#include <assert.h>\nint main(void){assert(window_accepts(2,3));assert(!window_accepts(3,3));return 0;}'
        )
        return _c(workspace, ("window.c",), probe)
    return None

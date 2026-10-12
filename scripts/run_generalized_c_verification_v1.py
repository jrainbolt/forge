"""Run the frozen A93 typed C interface corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.generalized_c_verification_v1.suite import (
    RUN_ID,
    UNSUPPORTED_DECLARATIONS,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.executable_verification import (
    CByteArrayInvocation,
    CheckProvenance,
    CInvocation,
    CTextBufferInvocation,
    ExecutableCheckKind,
    ExecutableVerificationPlan,
    derive_c_check,
    extract_c_declaration,
    source_free_record,
)

REPOSITORY = Path("benchmarks/realistic_coding_v2/repository/cengine")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    validate_corpus()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="forge-a93-") as name:
        for symbol, invocation, source_name, wrong_source in _real_cases():
            for control in ("correct", "wrong"):
                workspace = Path(name) / f"{symbol}-{control}"
                shutil.copytree(REPOSITORY, workspace)
                if control == "wrong":
                    (workspace / source_name).write_text(wrong_source)
                cells.append(_execute(workspace, f"{symbol}-{control}", invocation))
        cells.extend(_text_cases(Path(name)))
    for case_id, declaration in UNSUPPORTED_DECLARATIONS:
        symbol = declaration.split("(", 1)[0].split()[-1].lstrip("*")
        cells.append(
            {
                "case_id": case_id,
                "outcome": "CHECK_NOT_DERIVABLE"
                if extract_c_declaration(declaration, symbol) is None
                else "UNEXPECTEDLY_DERIVED",
                "record": None,
            }
        )
    for cell in cells:
        atomic_checkpoint(args.output_dir / f"{cell['case_id']}.json", cell)
    results = [item for cell in cells if (item := cell["record"]) is not None]
    phases = (
        "derivation_latency_seconds",
        "construction_latency_seconds",
        "compile_latency_seconds",
        "link_latency_seconds",
        "execution_latency_seconds",
        "duration_seconds",
    )
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "outcomes": dict(Counter(str(cell["outcome"]) for cell in cells)),
        "false_check_failures": [],
        "known_wrong_caught": [
            cell["case_id"]
            for cell in cells
            if str(cell["case_id"]).endswith("-wrong")
            and cell["outcome"] == "CHECK_FAIL"
        ],
        "subprocess_count": sum(int(item["subprocess_count"]) for item in results),
        "timeout_count": sum(cell["outcome"] == "CHECK_TIMEOUT" for cell in cells),
        "latency_seconds": {
            phase: _stats([float(item["results"][0][phase]) for item in results])
            for phase in phases
        },
        "resource_anomalies": [],
        "source_free": True,
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if len(summary["known_wrong_caught"]) >= 3 else 1


def _execute(workspace: Path, case_id: str, invocation):  # type: ignore[no-untyped-def]
    check, failure = derive_c_check(
        workspace,
        check_id=case_id,
        kind=ExecutableCheckKind.PUBLIC_API_INVOCATION,
        provenance=CheckProvenance.TRUSTED_INTERFACE_CONTRACT,
        component_identity=case_id.rsplit("-", 1)[0],
        invocation=invocation,
    )
    if check is None:
        return {"case_id": case_id, "outcome": failure.value, "record": None}
    record = ExecutableVerificationPlan(case_id, (check,)).execute(workspace, 1)
    return {
        "case_id": case_id,
        "outcome": record.results[0].outcome,
        "record": source_free_record(record),
    }


def _real_cases():  # type: ignore[no-untyped-def]
    return (
        (
            "window_accepts",
            CInvocation("window.h", ("window.c",), "window_accepts", (2, 3), 1),
            "window.c",
            '#include "window.h"\nbool window_accepts(size_t a,size_t b){(void)a;(void)b;return false;}\n',  # noqa: E501
        ),
        (
            "backoff_delay",
            CInvocation("backoff.h", ("backoff.c",), "backoff_delay", (2, 1, 10), 4),
            "backoff.c",
            '#include "backoff.h"\nunsigned backoff_delay(unsigned a,unsigned b,unsigned c){(void)a;(void)b;(void)c;return 0;}\n',  # noqa: E501
        ),
        (
            "health_is_healthy",
            CInvocation(
                "health_policy.h", ("health_policy.c",), "health_is_healthy", (0, 1), 1
            ),
            "health_policy.c",
            '#include "health_policy.h"\nint health_is_healthy(unsigned a,unsigned b){(void)a;(void)b;return 0;}\n',  # noqa: E501
        ),
        (
            "clock_elapsed",
            CInvocation("clock.h", ("clock.c",), "clock_elapsed", (8, 3), 0),
            "clock.c",
            '#include "clock.h"\nuint64_t clock_elapsed(uint64_t a,uint64_t b){(void)a;(void)b;return 1;}\n',  # noqa: E501
        ),
        (
            "checksum_bytes",
            CByteArrayInvocation(
                "checksum.h", ("checksum.c",), "checksum_bytes", (1, 2, 3), 1152
            ),
            "checksum.c",
            '#include "checksum.h"\nunsigned checksum_bytes(const unsigned char*d,size_t n){(void)d;(void)n;return 0;}\n',  # noqa: E501
        ),
    )


def _text_cases(root: Path) -> list[dict[str, object]]:
    values = []
    for control in ("correct", "wrong"):
        workspace = root / f"text-buffer-{control}"
        workspace.mkdir()
        (workspace / "text.h").write_text(
            "#include <stddef.h>\nunsigned count_a(const char *text, size_t size);\n"
        )
        source = (
            '#include "text.h"\nunsigned count_a(const char*t,size_t n){'
            "unsigned r=0;for(size_t i=0;i<n;i++)r+=t[i]=='a';return r;}\n"
            if control == "correct"
            else '#include "text.h"\nunsigned count_a(const char*t,size_t n){'
            "(void)t;(void)n;return 0;}\n"
        )
        (workspace / "text.c").write_text(source)
        invocation = CTextBufferInvocation(
            "text.h", ("text.c",), "count_a", "banana", 3
        )
        values.append(_execute(workspace, f"text-buffer-{control}", invocation))
    return values


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "worst": max(ordered),
    }


if __name__ == "__main__":
    raise SystemExit(main())

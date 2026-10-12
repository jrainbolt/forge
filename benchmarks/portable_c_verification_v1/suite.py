"""Frozen deterministic corpus for the A92 typed C harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from forge.executable_verification import ExecutableCheckKind

RUN_ID = "a92-portable-c-executable-verification-v1"


@dataclass(frozen=True, slots=True)
class CCase:
    case_id: str
    kind: ExecutableCheckKind
    header: str
    source: str
    invocation: str
    expected_outcome: str
    semantic_correct: bool | None


CASES = (
    CCase(
        "C01-scalar-correct",
        ExecutableCheckKind.PUBLIC_API_INVOCATION,
        "int add(int left, int right);\n",
        '#include "api.h"\nint add(int a,int b){return a+b;}\n',
        "scalar",
        "CHECK_PASS",
        True,
    ),
    CCase(
        "C02-scalar-wrong",
        ExecutableCheckKind.PUBLIC_API_INVOCATION,
        "int add(int left, int right);\n",
        '#include "api.h"\nint add(int a,int b){return a-b;}\n',
        "scalar",
        "CHECK_FAIL",
        False,
    ),
    CCase(
        "C03-boundary-correct",
        ExecutableCheckKind.BOUNDARY_ASSERTION,
        "int add(int left, int right);\n",
        '#include "api.h"\nint add(int a,int b){return a+b;}\n',
        "boundary",
        "CHECK_PASS",
        True,
    ),
    CCase(
        "C04-boundary-wrong",
        ExecutableCheckKind.BOUNDARY_ASSERTION,
        "int add(int left, int right);\n",
        '#include "api.h"\nint add(int a,int b){return a+b+1;}\n',
        "boundary",
        "CHECK_FAIL",
        False,
    ),
    CCase(
        "C05-invariant-correct",
        ExecutableCheckKind.INVARIANT_ASSERTION,
        "unsigned identity(unsigned value);\n",
        '#include "api.h"\nunsigned identity(unsigned v){return v;}\n',
        "invariant",
        "CHECK_PASS",
        True,
    ),
    CCase(
        "C06-invariant-wrong",
        ExecutableCheckKind.INVARIANT_ASSERTION,
        "unsigned identity(unsigned value);\n",
        '#include "api.h"\nunsigned identity(unsigned v){return v+1;}\n',
        "invariant",
        "CHECK_FAIL",
        False,
    ),
    CCase(
        "C07-h11-correct",
        ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        (
            "#include <stddef.h>\n"
            "unsigned checksum_bytes(const unsigned char *data, size_t size);\n"
        ),
        (
            '#include "api.h"\n'
            "unsigned checksum_bytes(const unsigned char*d,size_t n){"
            "unsigned r=0;for(size_t i=0;i<n;i++)r=r*31u+d[i];return r;}\n"
        ),
        "bytes",
        "CHECK_PASS",
        True,
    ),
    CCase(
        "C08-h11-wrong",
        ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        (
            "#include <stddef.h>\n"
            "unsigned checksum_bytes(const unsigned char *data, size_t size);\n"
        ),
        (
            '#include "api.h"\n'
            "unsigned checksum_bytes(const unsigned char*d,size_t n){"
            "(void)d;(void)n;return 0;}\n"
        ),
        "bytes",
        "CHECK_FAIL",
        False,
    ),
    CCase(
        "C09-compile-error",
        ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        "int add(int left, int right);\n",
        "not valid C\n",
        "scalar",
        "HARNESS_COMPILE_ERROR",
        None,
    ),
    CCase(
        "C10-link-error",
        ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        "int add(int left, int right);\n",
        "int unrelated(void){return 0;}\n",
        "scalar",
        "HARNESS_LINK_ERROR",
        None,
    ),
    CCase(
        "C11-runtime-error",
        ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        "int add(int left, int right);\n",
        (
            '#include <stdlib.h>\n#include "api.h"\n'
            "int add(int a,int b){(void)a;(void)b;abort();}\n"
        ),
        "scalar",
        "CHECK_RUNTIME_ERROR",
        None,
    ),
    CCase(
        "C12-pointer-graph",
        ExecutableCheckKind.PUBLIC_API_INVOCATION,
        "struct node; int inspect(const struct node *node);\n",
        "int placeholder;\n",
        "unsupported",
        "CHECK_NOT_DERIVABLE",
        None,
    ),
)


def corpus_identity() -> str:
    payload = [
        (
            case.case_id,
            case.kind.value,
            case.header,
            case.source,
            case.invocation,
            case.expected_outcome,
        )
        for case in CASES
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if len(CASES) != 12 or len({case.case_id for case in CASES}) != 12:
        raise RuntimeError("A92 requires twelve unique frozen cases")

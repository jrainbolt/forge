"""Frozen public-interface inventory for A93."""

from __future__ import annotations

import hashlib
import json

RUN_ID = "a93-typed-c-verification-generalization-v2-declaration-identity"
REAL_INTERFACES = (
    "window_accepts",
    "backoff_delay",
    "health_is_healthy",
    "clock_elapsed",
    "checksum_bytes",
)
SUPPORTED_CELLS = tuple(
    f"{symbol}-{control}"
    for symbol in REAL_INTERFACES
    for control in ("correct", "wrong")
) + ("text-buffer-correct", "text-buffer-wrong")
UNSUPPORTED_DECLARATIONS = (
    ("opaque-struct", "struct item; int inspect(const struct item *value);"),
    ("callback", "int visit(int (*callback)(int));"),
    ("pointer-to-pointer", "int split(const char **values);"),
    ("heap-ownership", "char *allocate(size_t size);"),
    ("mutable-output", "int parse(const char *text, int *output);"),
    ("variadic", "int format(const char *format, ...);"),
    ("function-pointer", "int apply(int value, int (*operation)(int));"),
    ("ambiguous-typedef", "word_t transform(word_t value);"),
)
CELLS = SUPPORTED_CELLS + tuple(item[0] for item in UNSUPPORTED_DECLARATIONS)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            (SUPPORTED_CELLS, UNSUPPORTED_DECLARATIONS), separators=(",", ":")
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if len(CELLS) != 20 or len(set(CELLS)) != 20:
        raise RuntimeError("A93 requires twenty unique frozen interface cells")

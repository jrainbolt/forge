"""Frozen A78 held-out realistic coding corpus."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import (
    REPOSITORY,
    AuthorityMode,
    FrozenTask,
    OperationClass,
    _task,
)
from forge.evaluation.realworld import SetupReplacement
from forge.models import MutationRepresentationPolicy

SUITE = "default-candidate-confirmation-v1"
VERSION = 2
SCHEMA_VERSION = 1
QUALIFICATION_RUN_ID = "a78-held-out-corpus-v2-workflow-qualified"
RUN_ID = "a78-qwen-large-default-candidate-confirmation-v4-qualified-workflow"
CHECKPOINT_NAMESPACE = "a78-default-candidate-confirmation-v4-qualified-workflow"
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")
PRIMARY_PROFILES = ("qwen-small", "qwen-large")
REPRESENTATION = MutationRepresentationPolicy.LINE_RANGE
SEED = 42
CONTEXT_SIZE = 8192
ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / "oracle.py"


def _base(
    task_id: str,
    prompt: str,
    expected: tuple[str, ...],
    edits: tuple[str, ...],
    creates: tuple[str, ...],
    *,
    setup: tuple[SetupReplacement, ...] = (),
):  # type: ignore[no-untyped-def]
    task = _task(task_id, prompt, expected, edits, creates, setup=setup)
    return replace(
        task,
        oracle_commands=(("python3", str(ORACLE), task_id),),
        # A creation target may be a configure prerequisite. Requiring configure
        # before the model can create it makes the workflow unreachable.
        setup_commands=() if creates else task.setup_commands,
    )


def tasks() -> tuple[FrozenTask, ...]:
    return (
        FrozenTask(
            "H01",
            1,
            "Python",
            OperationClass.EDIT_SINGLE,
            AuthorityMode.DISCOVERY_REQUIRED,
            _base(
                "H01",
                "Fix pyservice/metrics.py so Counter.increment accepts zero, rejects negative amounts, and accumulates nonnegative increments. Modify only that file and verify.",
                ("pyservice/metrics.py",),
                ("pyservice/metrics.py",),
                (),
                setup=(
                    SetupReplacement(
                        "pyservice/metrics.py", "amount < 0", "amount <= 0"
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H02",
            1,
            "Python",
            OperationClass.EDIT_SINGLE,
            AuthorityMode.DISCOVERY_REQUIRED,
            _base(
                "H02",
                "Fix pyservice/storage.py so putting a response for an existing request ID replaces the previous response while missing IDs still return None. Modify only that file and verify.",
                ("pyservice/storage.py",),
                ("pyservice/storage.py",),
                (),
                setup=(
                    SetupReplacement(
                        "pyservice/storage.py",
                        "self._values[request_id] = response",
                        "self._values.setdefault(request_id, response)",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H03",
            1,
            "C17",
            OperationClass.EDIT_SINGLE,
            AuthorityMode.DISCOVERY_REQUIRED,
            _base(
                "H03",
                "Fix cengine/clock.c so clock_elapsed returns the elapsed value when time advances and saturates at zero if the clock moves backwards. Modify only that file. Configure, build, and test.",
                ("cengine/clock.c", "cengine/clock.h"),
                ("cengine/clock.c",),
                (),
                setup=(
                    SetupReplacement(
                        "cengine/clock.c",
                        "return now >= started ? now - started : 0;",
                        "return now - started;",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H04",
            1,
            "Python",
            OperationClass.EDIT_MULTI,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H04",
                "Harden pyservice/config.py and pyservice/headers.py: timeouts must be positive, and valid header names must be trimmed and case-folded after validation. Both files must change. Verify.",
                ("pyservice/config.py", "pyservice/headers.py"),
                ("pyservice/config.py", "pyservice/headers.py"),
                (),
                setup=(
                    SetupReplacement("pyservice/config.py", "value <= 0", "value < 0"),
                    SetupReplacement(
                        "pyservice/headers.py",
                        "return candidate.casefold()",
                        "return candidate",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H05",
            1,
            "Python",
            OperationClass.EDIT_MULTI,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H05",
                "Repair pyservice/queue.py and pyservice/service.py: the request queue must remain FIFO, and missing request IDs must be rejected before empty payloads receive a 204 response. Both files must change. Verify.",
                ("pyservice/queue.py", "pyservice/service.py"),
                ("pyservice/queue.py", "pyservice/service.py"),
                (),
                setup=(
                    SetupReplacement(
                        "pyservice/queue.py",
                        "self._values.popleft()",
                        "self._values.pop()",
                    ),
                    SetupReplacement(
                        "pyservice/service.py",
                        'if not request.request_id:\n        return Response(400, b"missing request id")\n    if not request.payload:',
                        'if not request.payload:\n        return Response(204, b"")\n    if not request.request_id:',
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H06",
            1,
            "C17",
            OperationClass.EDIT_MULTI,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H06",
                "Repair cengine/parser.c and cengine/quota.c: port zero must be rejected, and a quota reservation may exactly fill remaining capacity. Both files must change. Configure, build, and test.",
                ("cengine/parser.c", "cengine/quota.c"),
                ("cengine/parser.c", "cengine/quota.c"),
                (),
                setup=(
                    SetupReplacement("cengine/parser.c", "value < 1", "value < 0"),
                    SetupReplacement(
                        "cengine/quota.c",
                        "amount > value->limit - value->used",
                        "amount >= value->limit - value->used",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H07",
            1,
            "Python",
            OperationClass.CREATE,
            AuthorityMode.PATH_KNOWN,
            _base(
                "H07",
                "Create missing pyservice/metrics.py with a Counter starting at zero whose increment method defaults to one, accumulates nonnegative amounts, and raises ValueError for negative amounts. Do not modify existing files. Verify.",
                (),
                (),
                ("pyservice/metrics.py",),
            ),
        ),
        FrozenTask(
            "H08",
            1,
            "C17",
            OperationClass.CREATE,
            AuthorityMode.PATH_KNOWN,
            _base(
                "H08",
                "Create missing cengine/checksum.c implementing checksum_bytes from checksum.h using the documented rolling multiply-by-33 XOR algorithm over every input byte. Do not modify existing files. Configure, build, and test.",
                ("cengine/checksum.h",),
                (),
                ("cengine/checksum.c",),
            ),
        ),
        FrozenTask(
            "H09",
            1,
            "Python",
            OperationClass.CREATE,
            AuthorityMode.PATH_KNOWN,
            _base(
                "H09",
                "Create missing pyservice/config.py with parse_enabled accepting trimmed case-insensitive true/false forms and defaults for absent keys, plus parse_timeout requiring a positive integer. Do not modify existing files. Verify.",
                (),
                (),
                ("pyservice/config.py",),
            ),
        ),
        FrozenTask(
            "H10",
            1,
            "Python",
            OperationClass.MIXED_EDIT_CREATE,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H10",
                "Fix pyservice/state.py so queued jobs may be cancelled as well as started, and create missing pyservice/metrics.py with a nonnegative accumulating Counter. Both paths are required. Verify.",
                ("pyservice/state.py",),
                ("pyservice/state.py",),
                ("pyservice/metrics.py",),
                setup=(
                    SetupReplacement(
                        "pyservice/state.py",
                        "frozenset({JobState.RUNNING, JobState.CANCELLED})",
                        "frozenset({JobState.RUNNING})",
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H11",
            1,
            "C17",
            OperationClass.MIXED_EDIT_CREATE,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H11",
                "Fix cengine/window.c so only a positive limit with events strictly below it is accepted, and create missing cengine/checksum.c implementing checksum_bytes from checksum.h. Both paths are required. Configure, build, and test.",
                ("cengine/window.c", "cengine/checksum.h"),
                ("cengine/window.c",),
                ("cengine/checksum.c",),
                setup=(
                    SetupReplacement(
                        "cengine/window.c", "events < limit", "events <= limit"
                    ),
                ),
            ),
        ),
        FrozenTask(
            "H12",
            1,
            "Python",
            OperationClass.MIXED_EDIT_CREATE,
            AuthorityMode.TRUSTED_REQUIRED_CANDIDATES,
            _base(
                "H12",
                "Fix pyservice/storage.py so later puts replace prior responses for the same request ID, and create missing pyservice/config.py with strict boolean and positive-timeout parsing. Both paths are required. Verify.",
                ("pyservice/storage.py",),
                ("pyservice/storage.py",),
                ("pyservice/config.py",),
                setup=(
                    SetupReplacement(
                        "pyservice/storage.py",
                        "self._values[request_id] = response",
                        "self._values.setdefault(request_id, response)",
                    ),
                ),
            ),
        ),
    )


def corpus_identity() -> str:
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "settings": {
            "seed": SEED,
            "temperature": 0,
            "context": CONTEXT_SIZE,
            "output": 512,
        },
        "repository": hashlib.sha256(
            b"".join(
                path.read_bytes()
                for path in sorted(REPOSITORY.rglob("*"))
                if path.is_file()
            )
        ).hexdigest(),
        "oracle": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "tasks": [
            {
                "id": item.versioned_id,
                "class": item.operation_class.value,
                "prompt": item.production_task.prompt,
                "allowed": item.production_task.allowed_paths,
                "setup": [
                    (x.path, x.expected, x.replacement)
                    for x in item.production_task.setup
                ],
                "creates": item.create_paths,
                "setup_commands": item.production_task.setup_commands,
            }
            for item in tasks()
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    values = tasks()
    if len(values) != 12 or len({item.task_id for item in values}) != 12:
        raise ValueError("A78 requires 12 unique tasks")
    counts = {
        kind: sum(item.operation_class is kind for item in values)
        for kind in OperationClass
    }
    if any(counts[kind] != 3 for kind in OperationClass):
        raise ValueError("A78 requires three tasks in every operation class")

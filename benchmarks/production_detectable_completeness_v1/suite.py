"""Frozen A82 tasks and production-visible structural checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from benchmarks.default_candidate_confirmation_v1.suite import tasks as a78_tasks
from benchmarks.realistic_coding_v2.suite import tasks as realistic_tasks
from forge.models import MutationRepresentationPolicy

SUITE = "production-detectable-completeness-v1"
VERSION = 2
RUN_ID = "a82-production-detectable-completeness-v2-component-role"
TASK_IDS = ("H11", "C07", "C08", "C09", "C10", "C11", "C12", "H08", "H07")
REPRESENTATION = MutationRepresentationPolicy.LINE_RANGE


class CheckKind(StrEnum):
    DECLARATION_IMPLEMENTATION_RELATION = "DECLARATION_IMPLEMENTATION_RELATION"
    CALLER_CALLEE_RELATION = "CALLER_CALLEE_RELATION"
    REGISTRATION_OR_USAGE_RELATION = "REGISTRATION_OR_USAGE_RELATION"
    IMPORT_INCLUDE_RELATION = "IMPORT_INCLUDE_RELATION"
    REQUIRED_SYMBOL_PRESENCE = "REQUIRED_SYMBOL_PRESENCE"
    REQUIRED_COMPONENT_ROLE = "REQUIRED_COMPONENT_ROLE"


@dataclass(frozen=True, slots=True)
class CompletenessCheck:
    check_id: str
    kind: CheckKind
    source_path: str
    target_path: str | None
    symbol: str | None
    caller: str | None
    authorized_scope: tuple[str, ...]

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "check_id": self.check_id,
                    "kind": self.kind.value,
                    "source_path_identity": _identity(self.source_path),
                    "target_path_identity": _identity(self.target_path),
                    "symbol_identity": _identity(self.symbol),
                    "caller_identity": _identity(self.caller),
                    "authorized_scope": tuple(
                        _identity(path) for path in self.authorized_scope
                    ),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusCase:
    task_id: str
    role: str
    checks: tuple[CompletenessCheck, ...]


def _identity(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


def _check(
    task: str,
    name: str,
    kind: CheckKind,
    source: str,
    *,
    target: str | None = None,
    symbol: str | None = None,
    caller: str | None = None,
    scope: tuple[str, ...],
) -> CompletenessCheck:
    return CompletenessCheck(
        f"{task}-{name}", kind, source, target, symbol, caller, scope
    )


K = CheckKind
CASES = (
    CorpusCase(
        "H11",
        "STRONG_TARGET",
        (
            _check(
                "H11",
                "checksum-declaration-implementation",
                K.DECLARATION_IMPLEMENTATION_RELATION,
                "cengine/checksum.h",
                target="cengine/checksum.c",
                symbol="checksum_bytes",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
            _check(
                "H11",
                "checksum-include",
                K.IMPORT_INCLUDE_RELATION,
                "cengine/checksum.c",
                target="cengine/checksum.h",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
            _check(
                "H11",
                "checksum-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "cengine/checksum.c",
                symbol="checksum_bytes",
                scope=("cengine/checksum.c",),
            ),
            _check(
                "H11",
                "checksum-component-role",
                K.REQUIRED_COMPONENT_ROLE,
                "cengine/checksum.c",
                target="cengine/checksum.h",
                symbol="checksum_bytes",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
        ),
    ),
    CorpusCase(
        "C07",
        "KNOWN_GOOD_CONTROL",
        (
            _check(
                "C07",
                "priority-import",
                K.IMPORT_INCLUDE_RELATION,
                "pyservice/scheduler.py",
                target="pyservice/priority.py",
                symbol="priority_rank",
                scope=("pyservice/scheduler.py", "pyservice/priority.py"),
            ),
            _check(
                "C07",
                "priority-call",
                K.CALLER_CALLEE_RELATION,
                "pyservice/scheduler.py",
                target="pyservice/priority.py",
                symbol="priority_rank",
                caller="schedule",
                scope=("pyservice/scheduler.py", "pyservice/priority.py"),
            ),
            _check(
                "C07",
                "priority-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "pyservice/priority.py",
                symbol="priority_rank",
                scope=("pyservice/priority.py",),
            ),
            _check(
                "C07",
                "priority-role",
                K.REQUIRED_COMPONENT_ROLE,
                "pyservice/priority.py",
                symbol="priority_rank",
                scope=("pyservice/priority.py",),
            ),
        ),
    ),
    CorpusCase(
        "C08",
        "RELATIONSHIP_TARGET",
        (
            _check(
                "C08",
                "serialization-import",
                K.IMPORT_INCLUDE_RELATION,
                "pyservice/response_wire.py",
                target="pyservice/serialization.py",
                symbol="encode_response",
                scope=("pyservice/response_wire.py", "pyservice/serialization.py"),
            ),
            _check(
                "C08",
                "serialization-call",
                K.CALLER_CALLEE_RELATION,
                "pyservice/response_wire.py",
                target="pyservice/serialization.py",
                symbol="encode_response",
                caller="wire_response",
                scope=("pyservice/response_wire.py", "pyservice/serialization.py"),
            ),
            _check(
                "C08",
                "serialization-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "pyservice/serialization.py",
                symbol="encode_response",
                scope=("pyservice/serialization.py",),
            ),
        ),
    ),
    CorpusCase(
        "C09",
        "RELATIONSHIP_TARGET",
        (
            _check(
                "C09",
                "backoff-declaration-implementation",
                K.DECLARATION_IMPLEMENTATION_RELATION,
                "cengine/backoff.h",
                target="cengine/backoff.c",
                symbol="backoff_delay",
                scope=("cengine/backoff.h", "cengine/backoff.c"),
            ),
            _check(
                "C09",
                "backoff-call",
                K.CALLER_CALLEE_RELATION,
                "cengine/retry_policy.c",
                target="cengine/backoff.c",
                symbol="backoff_delay",
                caller="retry_wait",
                scope=("cengine/retry_policy.c", "cengine/backoff.c"),
            ),
            _check(
                "C09",
                "backoff-include",
                K.IMPORT_INCLUDE_RELATION,
                "cengine/backoff.c",
                target="cengine/backoff.h",
                scope=("cengine/backoff.h", "cengine/backoff.c"),
            ),
            _check(
                "C09",
                "backoff-component-role",
                K.REQUIRED_COMPONENT_ROLE,
                "cengine/backoff.c",
                target="cengine/backoff.h",
                symbol="backoff_delay",
                scope=("cengine/backoff.h", "cengine/backoff.c"),
            ),
        ),
    ),
    CorpusCase(
        "C10",
        "RELATIONSHIP_TARGET",
        (
            _check(
                "C10",
                "dispatch-import",
                K.IMPORT_INCLUDE_RELATION,
                "pyservice/dispatch.py",
                target="pyservice/dispatch_rules.py",
                symbol="choose_handler",
                scope=("pyservice/dispatch.py", "pyservice/dispatch_rules.py"),
            ),
            _check(
                "C10",
                "dispatch-call",
                K.CALLER_CALLEE_RELATION,
                "pyservice/dispatch.py",
                target="pyservice/dispatch_rules.py",
                symbol="choose_handler",
                caller="dispatch",
                scope=("pyservice/dispatch.py", "pyservice/dispatch_rules.py"),
            ),
            _check(
                "C10",
                "dispatch-usage",
                K.REGISTRATION_OR_USAGE_RELATION,
                "pyservice/dispatch.py",
                target="pyservice/dispatch_rules.py",
                symbol="choose_handler",
                caller="dispatch",
                scope=("pyservice/dispatch.py", "pyservice/dispatch_rules.py"),
            ),
            _check(
                "C10",
                "dispatch-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "pyservice/dispatch_rules.py",
                symbol="choose_handler",
                scope=("pyservice/dispatch_rules.py",),
            ),
        ),
    ),
    CorpusCase(
        "C11",
        "RELATIONSHIP_TARGET",
        (
            _check(
                "C11",
                "currency-import",
                K.IMPORT_INCLUDE_RELATION,
                "pyservice/billing.py",
                target="pyservice/currency.py",
                symbol="format_cents",
                scope=("pyservice/billing.py", "pyservice/currency.py"),
            ),
            _check(
                "C11",
                "currency-call",
                K.CALLER_CALLEE_RELATION,
                "pyservice/billing.py",
                target="pyservice/currency.py",
                symbol="format_cents",
                caller="invoice_line",
                scope=("pyservice/billing.py", "pyservice/currency.py"),
            ),
            _check(
                "C11",
                "currency-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "pyservice/currency.py",
                symbol="format_cents",
                scope=("pyservice/currency.py",),
            ),
        ),
    ),
    CorpusCase(
        "C12",
        "RELATIONSHIP_TARGET",
        (
            _check(
                "C12",
                "health-declaration-implementation",
                K.DECLARATION_IMPLEMENTATION_RELATION,
                "cengine/health_policy.h",
                target="cengine/health_policy.c",
                symbol="health_is_healthy",
                scope=("cengine/health_policy.h", "cengine/health_policy.c"),
            ),
            _check(
                "C12",
                "health-call",
                K.CALLER_CALLEE_RELATION,
                "cengine/health.c",
                target="cengine/health_policy.c",
                symbol="health_is_healthy",
                caller="health_status",
                scope=("cengine/health.c", "cengine/health_policy.c"),
            ),
            _check(
                "C12",
                "health-include",
                K.IMPORT_INCLUDE_RELATION,
                "cengine/health_policy.c",
                target="cengine/health_policy.h",
                scope=("cengine/health_policy.h", "cengine/health_policy.c"),
            ),
            _check(
                "C12",
                "health-component-role",
                K.REQUIRED_COMPONENT_ROLE,
                "cengine/health_policy.c",
                target="cengine/health_policy.h",
                symbol="health_is_healthy",
                scope=("cengine/health_policy.h", "cengine/health_policy.c"),
            ),
        ),
    ),
    CorpusCase(
        "H08",
        "KNOWN_GOOD_CONTROL",
        (
            _check(
                "H08",
                "checksum-declaration-implementation",
                K.DECLARATION_IMPLEMENTATION_RELATION,
                "cengine/checksum.h",
                target="cengine/checksum.c",
                symbol="checksum_bytes",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
            _check(
                "H08",
                "checksum-include",
                K.IMPORT_INCLUDE_RELATION,
                "cengine/checksum.c",
                target="cengine/checksum.h",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
            _check(
                "H08",
                "checksum-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "cengine/checksum.c",
                symbol="checksum_bytes",
                scope=("cengine/checksum.c",),
            ),
            _check(
                "H08",
                "checksum-component-role",
                K.REQUIRED_COMPONENT_ROLE,
                "cengine/checksum.c",
                target="cengine/checksum.h",
                symbol="checksum_bytes",
                scope=("cengine/checksum.h", "cengine/checksum.c"),
            ),
        ),
    ),
    CorpusCase(
        "H07",
        "NEGATIVE_SEMANTIC_CONTROL",
        (
            _check(
                "H07",
                "counter-symbol",
                K.REQUIRED_SYMBOL_PRESENCE,
                "pyservice/metrics.py",
                symbol="Counter",
                scope=("pyservice/metrics.py",),
            ),
            _check(
                "H07",
                "counter-role",
                K.REQUIRED_COMPONENT_ROLE,
                "pyservice/metrics.py",
                symbol="Counter",
                scope=("pyservice/metrics.py",),
            ),
        ),
    ),
)


def tasks():  # type: ignore[no-untyped-def]
    definitions = {item.task_id: item for item in (*a78_tasks(), *realistic_tasks())}
    return tuple(definitions[task_id] for task_id in TASK_IDS)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "suite": SUITE,
                "version": VERSION,
                "tasks": TASK_IDS,
                "checks": [check.identity for case in CASES for check in case.checks],
                "representation": REPRESENTATION.value,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if tuple(case.task_id for case in CASES) != TASK_IDS:
        raise RuntimeError("A82 case/task mismatch")
    if len({check.check_id for case in CASES for check in case.checks}) != sum(
        len(case.checks) for case in CASES
    ):
        raise RuntimeError("A82 check IDs are not unique")
    present = {check.kind for case in CASES for check in case.checks}
    if present != set(CheckKind):
        raise RuntimeError("A82 does not cover every allowed structural check kind")

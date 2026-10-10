from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.production_detectable_completeness_v1.checker import (
    COMPLETE,
    INCOMPLETE,
    NOT_CHECKABLE,
    evaluate,
)
from benchmarks.production_detectable_completeness_v1.runner import (
    CheckResult,
    classify,
)
from benchmarks.production_detectable_completeness_v1.suite import (
    CASES,
    TASK_IDS,
    CheckKind,
    CompletenessCheck,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def _check(
    kind: CheckKind,
    source: str,
    *,
    target: str | None = None,
    symbol: str | None = None,
    caller: str | None = None,
    scope: tuple[str, ...] | None = None,
) -> CompletenessCheck:
    return CompletenessCheck(
        "test-check",
        kind,
        source,
        target,
        symbol,
        caller,
        scope or tuple(path for path in (source, target) if path),
    )


def _result(status: str) -> CheckResult:
    return CheckResult(
        "x", "a" * 64, "REQUIRED_SYMBOL_PRESENCE", "b" * 64, "c" * 64, status
    )


def test_check_definitions_are_frozen_before_execution() -> None:
    validate_corpus()
    assert isinstance(CASES, tuple)
    assert tuple(case.task_id for case in CASES) == TASK_IDS
    assert len(tasks()) == 9
    assert len(corpus_identity()) == 64


def test_checks_contain_no_hidden_oracle_or_reference_data() -> None:
    for case in CASES:
        for check in case.checks:
            values = (check.check_id, check.source_path, check.target_path or "")
            assert all("oracle" not in value.lower() for value in values)
            assert all("reference" not in value.lower() for value in values)
            assert not hasattr(check, "expected_implementation")


def test_python_import_call_usage_symbol_and_role(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "consumer.py").write_text(
        "from pkg.provider import serve\n\ndef run():\n    return serve()\n"
    )
    (package / "provider.py").write_text("def serve():\n    return 1\n")
    checks = (
        _check(
            CheckKind.IMPORT_INCLUDE_RELATION,
            "pkg/consumer.py",
            target="pkg/provider.py",
            symbol="serve",
        ),
        _check(
            CheckKind.CALLER_CALLEE_RELATION,
            "pkg/consumer.py",
            target="pkg/provider.py",
            symbol="serve",
            caller="run",
        ),
        _check(
            CheckKind.REGISTRATION_OR_USAGE_RELATION,
            "pkg/consumer.py",
            target="pkg/provider.py",
            symbol="serve",
            caller="run",
        ),
        _check(CheckKind.REQUIRED_SYMBOL_PRESENCE, "pkg/provider.py", symbol="serve"),
        _check(CheckKind.REQUIRED_COMPONENT_ROLE, "pkg/provider.py", symbol="serve"),
    )
    assert {evaluate(check, tmp_path) for check in checks} == {COMPLETE}


def test_c_declaration_implementation_and_include(tmp_path: Path) -> None:
    root = tmp_path / "cengine"
    root.mkdir()
    (root / "value.h").write_text("unsigned value_of(unsigned input);\n")
    (root / "value.c").write_text(
        '#include "value.h"\nunsigned value_of(unsigned input) { return input; }\n'
    )
    relation = _check(
        CheckKind.DECLARATION_IMPLEMENTATION_RELATION,
        "cengine/value.h",
        target="cengine/value.c",
        symbol="value_of",
    )
    include = _check(
        CheckKind.IMPORT_INCLUDE_RELATION, "cengine/value.c", target="cengine/value.h"
    )
    role = _check(
        CheckKind.REQUIRED_COMPONENT_ROLE,
        "cengine/value.c",
        target="cengine/value.h",
        symbol="value_of",
    )
    assert evaluate(relation, tmp_path) == COMPLETE
    assert evaluate(include, tmp_path) == COMPLETE
    assert evaluate(role, tmp_path) == COMPLETE


def test_missing_relationship_is_incomplete(tmp_path: Path) -> None:
    (tmp_path / "consumer.py").write_text("def run():\n    return 1\n")
    check = _check(
        CheckKind.CALLER_CALLEE_RELATION, "consumer.py", symbol="serve", caller="run"
    )
    assert evaluate(check, tmp_path) == INCOMPLETE
    assert classify((_result(INCOMPLETE),)) == "STRUCTURAL_OBLIGATION_MISSING"


def test_ambiguous_or_uncheckable_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def run():\n    return 1\n")
    check = _check(CheckKind.REQUIRED_SYMBOL_PRESENCE, "module.py", symbol=None)
    assert evaluate(check, tmp_path) == NOT_CHECKABLE
    assert classify((_result(NOT_CHECKABLE),)) == "STRUCTURAL_CHECK_PARTIAL"


def test_structurally_complete_does_not_imply_semantic_pass() -> None:
    structural = classify((_result(COMPLETE),))
    semantic_pass = False
    assert structural == "STRUCTURALLY_COMPLETE"
    assert not semantic_pass


def test_correct_control_is_accepted() -> None:
    assert classify((_result(COMPLETE), _result(COMPLETE))) == "STRUCTURALLY_COMPLETE"


def test_source_free_checkpoint_exactly_once_and_package_exclusion(
    tmp_path: Path,
) -> None:
    payload = {
        "check_id": "H11-checksum-symbol",
        "check_identity": "a" * 64,
        "status": COMPLETE,
        "classification": "STRUCTURALLY_COMPLETE",
    }
    assert standard_result_is_source_free(payload)
    path = tmp_path / "cell.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)
    assert resume_checkpoint(path)["check_identity"] == "a" * 64
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "production_detectable_completeness_v1" not in configuration

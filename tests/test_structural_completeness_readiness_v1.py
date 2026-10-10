from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.production_detectable_completeness_v1 import checker
from benchmarks.production_detectable_completeness_v1.checker import (
    NOT_CHECKABLE,
    _c_component_role,
    evaluate_detailed,
)
from benchmarks.production_detectable_completeness_v1.suite import (
    CheckKind,
    CompletenessCheck,
)
from benchmarks.structural_completeness_readiness_v1.runner import proposed_policy
from benchmarks.structural_completeness_readiness_v1.suite import (
    CASES,
    corpus_identity,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def _include_check(source: str, target: str, scope: tuple[str, ...] | None = None):  # type: ignore[no-untyped-def]
    return CompletenessCheck(
        "include-control",
        CheckKind.IMPORT_INCLUDE_RELATION,
        source,
        target,
        None,
        None,
        scope or (source,),
    )


def test_frozen_readiness_corpus() -> None:
    validate_corpus()
    assert len(CASES) == 28
    assert sum(case.model_call for case in CASES) == 24
    assert len(corpus_identity()) == 64


@pytest.mark.parametrize(
    ("decision", "operational", "expected"),
    [
        ("STRUCTURAL_SHADOW_PASS", False, "ALLOW"),
        ("STRUCTURAL_SHADOW_FAIL", False, "BLOCK"),
        ("STRUCTURAL_SHADOW_PARTIAL", False, "CONTINUE_WITH_DIAGNOSTIC"),
        ("STRUCTURAL_SHADOW_NOT_APPLICABLE", False, "ALLOW_NOT_APPLICABLE"),
        ("STRUCTURAL_SHADOW_PARTIAL", True, "FAIL_OPEN_CONTINUE_WITH_DIAGNOSTIC"),
    ],
)
def test_proposed_offline_policy(
    decision: str, operational: bool, expected: str
) -> None:
    assert proposed_policy(decision, operational) == expected


def test_compiler_timeout_is_operational_not_structural(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "value.c"
    source.write_text("int value(void) { return 1; }\n")

    def timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(checker.subprocess, "run", timeout)
    result = _c_component_role(tmp_path, "value.c")
    assert result.status == NOT_CHECKABLE
    assert result.operational_failure == "COMPILER_TIMEOUT"
    assert result.subprocess_exit == "TIMEOUT"


def test_subprocess_allowlist_shell_and_bounded_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "value.c").write_text("int value(void) { return 1; }\n")
    captured = {}

    def complete(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(checker.subprocess, "run", complete)
    result = _c_component_role(tmp_path, "value.c")
    assert result.status == "COMPLETE"
    assert captured["argv"][0] == "/usr/bin/cc"
    assert captured["shell"] is False
    assert captured["timeout"] == 2
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.DEVNULL
    assert not any(token in captured["argv"] for token in ("curl", "pip", "npm"))
    rejected = _c_component_role(tmp_path, "value.c", executable="/bin/sh")
    assert rejected.operational_failure == "EXECUTABLE_NOT_ALLOWLISTED"


def test_workspace_confinement_and_unsupported_language(tmp_path: Path) -> None:
    escaped = _c_component_role(tmp_path, "../outside.c")
    assert escaped.operational_failure == "WORKSPACE_CONFINEMENT_FAILURE"
    (tmp_path / "module.rs").write_text("fn serve() {}\n")
    check = CompletenessCheck(
        "rust",
        CheckKind.REQUIRED_SYMBOL_PRESENCE,
        "module.rs",
        None,
        "serve",
        None,
        ("module.rs",),
    )
    detail = evaluate_detailed(check, tmp_path)
    assert detail.operational_failure == "UNSUPPORTED_LANGUAGE"


@pytest.mark.parametrize(
    ("source_path", "include", "target_path"),
    [
        ("src/value.c", "./header.h", "src/header.h"),
        ("src/tests/value.c", "../header.h", "src/header.h"),
        ("src/tests/value.c", "../include/header.h", "src/include/header.h"),
        ("src/tests/value.c", "./nested/../../header.h", "src/header.h"),
    ],
)
def test_relative_include_resolution(
    tmp_path: Path, source_path: str, include: str, target_path: str
) -> None:
    source = tmp_path / source_path
    target = tmp_path / target_path
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("int value(void);\n")
    source.write_text(f'#include "{include}"\n')
    assert (
        evaluate_detailed(_include_check(source_path, target_path), tmp_path).status
        == "COMPLETE"
    )


def test_include_escape_and_nonexistent_target(tmp_path: Path) -> None:
    source = tmp_path / "src/value.c"
    source.parent.mkdir()
    source.write_text('#include "../../outside.h"\n')
    escaped = evaluate_detailed(_include_check("src/value.c", "src/header.h"), tmp_path)
    assert escaped.status == NOT_CHECKABLE
    assert escaped.operational_failure == "INCLUDE_WORKSPACE_ESCAPE"
    source.write_text('#include "missing.h"\n')
    missing = evaluate_detailed(
        _include_check("src/value.c", "src/missing.h"), tmp_path
    )
    assert missing.status == "INCOMPLETE"


def test_include_inspection_does_not_expand_mutation_authority(tmp_path: Path) -> None:
    source = tmp_path / "tests/value.c"
    target = tmp_path / "include/header.h"
    source.parent.mkdir()
    target.parent.mkdir()
    source.write_text('#include "../include/header.h"\n')
    target.write_text("int value(void);\n")
    check = _include_check("tests/value.c", "include/header.h", ("tests/value.c",))
    assert evaluate_detailed(check, tmp_path).status == "COMPLETE"
    assert check.authorized_scope == ("tests/value.c",)


def test_c05_and_c06_parent_relative_includes() -> None:
    repository = (
        Path(__file__).parents[1] / "benchmarks/realistic_semantic_v1/repository"
    )
    assert (
        evaluate_detailed(
            _include_check("cengine/tests/test_window.c", "cengine/window.h"),
            repository,
        ).status
        == "COMPLETE"
    )
    assert (
        evaluate_detailed(
            _include_check("cengine/tests/test_status.c", "cengine/status.h"),
            repository,
        ).status
        == "COMPLETE"
    )


def test_generic_indentation_parser_proves_c11_incomplete(tmp_path: Path) -> None:
    billing = tmp_path / "pyservice/billing.py"
    currency = tmp_path / "pyservice/currency.py"
    billing.parent.mkdir()
    currency.write_text("def format_cents(value):\n    return str(value)\n")
    billing.write_text(
        "from pyservice.currency import format_cents\n\n"
        "def invoice_line(value):\n"
        "    if value < 0:\n"
        "    return format_cents(value)\n"
    )
    imported = CompletenessCheck(
        "C11-import",
        CheckKind.IMPORT_INCLUDE_RELATION,
        "pyservice/billing.py",
        "pyservice/currency.py",
        "format_cents",
        None,
        ("pyservice/billing.py",),
    )
    called = CompletenessCheck(
        "C11-call",
        CheckKind.CALLER_CALLEE_RELATION,
        "pyservice/billing.py",
        "pyservice/currency.py",
        "format_cents",
        "invoice_line",
        ("pyservice/billing.py",),
    )
    assert evaluate_detailed(imported, tmp_path).status == "INCOMPLETE"
    assert evaluate_detailed(called, tmp_path).status == "INCOMPLETE"


def test_operational_parser_failure_remains_partial(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def broken(:\n")
    check = CompletenessCheck(
        "parser",
        CheckKind.REQUIRED_SYMBOL_PRESENCE,
        "module.py",
        None,
        "serve",
        None,
        ("module.py",),
    )
    detail = evaluate_detailed(check, tmp_path)
    assert detail.status == NOT_CHECKABLE
    assert detail.operational_failure == "PARSER_FAILURE"


def test_h12_truth_records_real_structural_miss_without_new_check_kind() -> None:
    h12 = next(case for case in CASES if case.case_id == "H12")
    assert h12.truth == "SEMANTIC_NEGATIVE_STRUCTURAL_MISS"
    assert {check.kind for check in h12.checks} <= set(CheckKind)


def test_source_free_checkpoint_resume_and_package_exclusion(tmp_path: Path) -> None:
    payload = {
        "decision": "STRUCTURAL_SHADOW_PARTIAL",
        "operational_failure": "PARSER_FAILURE",
        "policy": "FAIL_OPEN_CONTINUE_WITH_DIAGNOSTIC",
    }
    assert standard_result_is_source_free(payload)
    path = tmp_path / "cell.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "structural_completeness_readiness_v1" not in configuration

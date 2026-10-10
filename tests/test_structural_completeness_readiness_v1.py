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

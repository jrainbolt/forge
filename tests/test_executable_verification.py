from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.executable_verification import (
    CByteArrayInvocation,
    CCompiler,
    CheckProvenance,
    CInvocation,
    ExecutableCheck,
    ExecutableCheckKind,
    ExecutableCheckOutcome,
    ExecutableVerificationPlan,
    PythonInvocation,
    PythonMethodCall,
    PythonObjectScenario,
    derive_c_check,
    derive_python_check,
    derive_python_object_check,
    extract_c_declaration,
    source_free_record,
)
from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.orchestration.coding_task import CodingTaskState
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_registry,
)


def _python_check(
    expected: object,
    *,
    kind: ExecutableCheckKind = ExecutableCheckKind.EXPLICIT_EXAMPLE_ASSERTION,
    module: str = "value",
    symbol: str = "current",
    timeout: float = 2,
) -> ExecutableCheck:
    case = PythonInvocation(module, symbol, expected=expected)
    check, failure = derive_python_check(
        check_id="explicit-value",
        kind=kind,
        provenance=CheckProvenance.EXPLICIT_TASK_EXAMPLE,
        component_identity=f"{module}.{symbol}",
        cases=(case,),
        timeout_seconds=timeout,
    )
    assert failure is None and check is not None
    return check


@pytest.mark.parametrize(
    "kind,provenance",
    (
        (
            ExecutableCheckKind.EXPLICIT_EXAMPLE_ASSERTION,
            CheckProvenance.EXPLICIT_TASK_EXAMPLE,
        ),
        (
            ExecutableCheckKind.BOUNDARY_ASSERTION,
            CheckProvenance.EXPLICIT_BOUNDARY_REQUIREMENT,
        ),
        (
            ExecutableCheckKind.INVARIANT_ASSERTION,
            CheckProvenance.EXPLICIT_INVARIANT,
        ),
        (
            ExecutableCheckKind.PUBLIC_API_INVOCATION,
            CheckProvenance.PUBLIC_API_CONTRACT,
        ),
    ),
)
def test_typed_python_derivation(kind, provenance) -> None:  # type: ignore[no-untyped-def]
    check, failure = derive_python_check(
        check_id="contract-check",
        kind=kind,
        provenance=provenance,
        component_identity="package.public_api",
        cases=(PythonInvocation("package", "public_api", (1,), expected=2),),
    )
    assert failure is None
    assert check is not None and check.kind is kind


def test_unsupported_and_hidden_provenance_are_not_derivable() -> None:
    for provenance in ("SEMANTIC_ORACLE", "REFERENCE_IMPLEMENTATION"):
        check, failure = derive_python_check(
            check_id="hidden-check",
            kind=ExecutableCheckKind.INVARIANT_ASSERTION,
            provenance=provenance,
            component_identity="hidden.truth",
            cases=(PythonInvocation("package", "function", expected=True),),
        )
        assert check is None
        assert failure is ExecutableCheckOutcome.CHECK_NOT_DERIVABLE


def test_construction_error_fails_safe() -> None:
    check, failure = derive_python_check(
        check_id="bad-check",
        kind=ExecutableCheckKind.BOUNDARY_ASSERTION,
        provenance=CheckProvenance.EXPLICIT_BOUNDARY_REQUIREMENT,
        component_identity="package.function",
        cases=(PythonInvocation("package", "function", expected=True),),
        timeout_seconds=999,
    )
    assert check is None
    assert failure is ExecutableCheckOutcome.CHECK_CONSTRUCTION_ERROR


def test_correct_wrong_execution_error_and_timeout_are_distinct(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text(
        "def current(): return 2\n"
        "def broken(): raise RuntimeError('x')\n"
        "def forever():\n while True: pass\n"
    )
    checks = (
        _python_check(2),
        _python_check(3),
        _python_check(None, symbol="broken"),
        _python_check(None, symbol="forever", timeout=0.05),
    )
    outcomes = [
        ExecutableVerificationPlan(f"plan-{index}", (check,))
        .execute(tmp_path, 1)
        .results[0]
        .outcome
        for index, check in enumerate(checks)
    ]
    assert outcomes == [
        "CHECK_PASS",
        "CHECK_FAIL",
        "CHECK_RUNTIME_ERROR",
        "CHECK_TIMEOUT",
    ]


def test_c_probe_paths_are_workspace_confined() -> None:
    with pytest.raises(ValueError, match="confined"):
        CInvocation("../hidden.h", ("source.c",), "run", (), 0)


class _PassthroughSandbox:
    identity = "test-passthrough"

    def available(self) -> bool:
        return True

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        return argv


def _enable_c_test_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forge.tools.project.platform_sandbox", lambda: _PassthroughSandbox()
    )


def _write_c_fixture(
    root: Path, header: str, source: str, *, name: str = "sample"
) -> None:
    (root / f"{name}.h").write_text(header)
    (root / f"{name}.c").write_text(source)


def _derive_scalar(root: Path, expected: int = 5):  # type: ignore[no-untyped-def]
    return derive_c_check(
        root,
        check_id="scalar-check",
        kind=ExecutableCheckKind.EXPLICIT_EXAMPLE_ASSERTION,
        provenance=CheckProvenance.PUBLIC_API_CONTRACT,
        component_identity="sample.add",
        invocation=CInvocation("sample.h", ("sample.c",), "add", (2, 3), expected),
        compiler=CCompiler.CC,
    )


def test_trusted_header_type_extraction() -> None:
    scalar = extract_c_declaration("int add(int left, int right);", "add")
    assert scalar is not None
    assert scalar.return_type == "int"
    assert scalar.parameter_types == ("int", "int")
    byte_array = extract_c_declaration(
        "unsigned checksum(const unsigned char *data, size_t size);", "checksum"
    )
    assert byte_array is not None
    assert byte_array.parameter_types == ("const unsigned char *", "size_t")


def test_integer_and_byte_array_correct_controls_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_c_test_execution(monkeypatch)
    scalar = tmp_path / "scalar"
    scalar.mkdir()
    _write_c_fixture(
        scalar,
        "int add(int left, int right);\n",
        '#include "sample.h"\nint add(int left,int right){return left+right;}\n',
    )
    scalar_check, failure = _derive_scalar(scalar)
    assert failure is None and scalar_check is not None
    scalar_result = ExecutableVerificationPlan("scalar", (scalar_check,)).execute(
        scalar, 1
    )
    assert scalar_result.results[0].outcome == "CHECK_PASS"

    arrays = tmp_path / "arrays"
    arrays.mkdir()
    _write_c_fixture(
        arrays,
        (
            "#include <stddef.h>\n"
            "unsigned checksum(const unsigned char *data, size_t size);\n"
        ),
        (
            '#include "sample.h"\n'
            "unsigned checksum(const unsigned char *d,size_t n){"
            "unsigned r=0;for(size_t i=0;i<n;i++)r+=d[i];return r;}\n"
        ),
    )
    array_check, failure = derive_c_check(
        arrays,
        check_id="array-check",
        kind=ExecutableCheckKind.COMPILE_OR_USE_PROBE,
        provenance=CheckProvenance.PUBLIC_API_CONTRACT,
        component_identity="sample.checksum",
        invocation=CByteArrayInvocation(
            "sample.h", ("sample.c",), "checksum", (1, 2, 3), 6
        ),
    )
    assert failure is None and array_check is not None
    array_result = ExecutableVerificationPlan("array", (array_check,)).execute(
        arrays, 1
    )
    assert array_result.results[0].outcome == "CHECK_PASS"


def test_two_known_wrong_c_behaviors_are_proven_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_c_test_execution(monkeypatch)
    scalar = tmp_path / "wrong-scalar"
    scalar.mkdir()
    _write_c_fixture(
        scalar,
        "int add(int left, int right);\n",
        '#include "sample.h"\nint add(int left,int right){return left-right;}\n',
    )
    scalar_check, _ = _derive_scalar(scalar)
    assert scalar_check is not None
    assert (
        ExecutableVerificationPlan("wrong-scalar", (scalar_check,))
        .execute(scalar, 1)
        .results[0]
        .outcome
        == "CHECK_FAIL"
    )

    arrays = tmp_path / "wrong-array"
    arrays.mkdir()
    _write_c_fixture(
        arrays,
        (
            "#include <stddef.h>\n"
            "unsigned checksum(const unsigned char *data, size_t size);\n"
        ),
        (
            '#include "sample.h"\n'
            "unsigned checksum(const unsigned char *d,size_t n){"
            "(void)d;(void)n;return 0;}\n"
        ),
    )
    array_check, _ = derive_c_check(
        arrays,
        check_id="wrong-array-check",
        kind=ExecutableCheckKind.INVARIANT_ASSERTION,
        provenance=CheckProvenance.PUBLIC_API_CONTRACT,
        component_identity="sample.checksum",
        invocation=CByteArrayInvocation(
            "sample.h", ("sample.c",), "checksum", (1, 2, 3), 6
        ),
    )
    assert array_check is not None
    assert (
        ExecutableVerificationPlan("wrong-array", (array_check,))
        .execute(arrays, 1)
        .results[0]
        .outcome
        == "CHECK_FAIL"
    )


def test_unsupported_pointer_graph_is_not_derivable(tmp_path: Path) -> None:
    _write_c_fixture(
        tmp_path,
        "struct item; int inspect(const struct item *value);\n",
        "int placeholder;\n",
    )
    check, failure = _derive_scalar(tmp_path)
    assert check is None
    assert failure is ExecutableCheckOutcome.CHECK_NOT_DERIVABLE


@pytest.mark.parametrize(
    "source,expected",
    (
        ("this is not C\n", "HARNESS_COMPILE_ERROR"),
        ("int unrelated(void){return 0;}\n", "HARNESS_LINK_ERROR"),
        (
            "#include <stdlib.h>\nint add(int a,int b){(void)a;(void)b;abort();}\n",
            "CHECK_RUNTIME_ERROR",
        ),
    ),
)
def test_c_operational_failures_fail_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected: str,
) -> None:
    _enable_c_test_execution(monkeypatch)
    _write_c_fixture(tmp_path, "int add(int left, int right);\n", source)
    check, failure = _derive_scalar(tmp_path)
    assert failure is None and check is not None
    result = ExecutableVerificationPlan("operational", (check,)).execute(tmp_path, 1)
    assert result.results[0].outcome == expected
    assert not result.proven_failure


def test_header_change_becomes_construction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_c_test_execution(monkeypatch)
    _write_c_fixture(
        tmp_path,
        "int add(int left, int right);\n",
        "int add(int left,int right){return left+right;}\n",
    )
    check, _ = _derive_scalar(tmp_path)
    assert check is not None
    (tmp_path / "sample.h").write_text("unsigned add(unsigned, unsigned);\n")
    result = ExecutableVerificationPlan("changed", (check,)).execute(tmp_path, 1)
    assert result.results[0].outcome == "HARNESS_CONSTRUCTION_ERROR"


def test_c_latency_telemetry_is_source_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_c_test_execution(monkeypatch)
    _write_c_fixture(
        tmp_path,
        "int add(int left, int right);\n",
        "int add(int left,int right){return left+right;}\n",
    )
    check, _ = _derive_scalar(tmp_path)
    assert check is not None and check.derivation_latency_seconds >= 0
    record = ExecutableVerificationPlan("timing", (check,)).execute(tmp_path, 1)
    item = source_free_record(record)["results"][0]  # type: ignore[index]
    assert item["construction_latency_seconds"] >= 0  # type: ignore[index]
    assert item["compile_latency_seconds"] > 0  # type: ignore[index]
    assert item["link_latency_seconds"] > 0  # type: ignore[index]
    assert item["execution_latency_seconds"] > 0  # type: ignore[index]


def test_object_invariant_and_c_byte_array_are_typed() -> None:
    scenario = PythonObjectScenario(
        "metrics",
        "Counter",
        calls=(PythonMethodCall("increment", (1,), expected=None),),
    )
    check, failure = derive_python_object_check(
        check_id="counter-invariant",
        kind=ExecutableCheckKind.INVARIANT_ASSERTION,
        provenance=CheckProvenance.EXPLICIT_INVARIANT,
        component_identity="metrics.Counter",
        scenario=scenario,
    )
    assert failure is None and check is not None
    byte_case = CByteArrayInvocation(
        "checksum.h", ("checksum.c",), "checksum_bytes", (1, 2), 35
    )
    assert byte_case.values == (1, 2)


def test_check_schema_has_no_arbitrary_shell_surface() -> None:
    assert "argv" not in ExecutableCheck.__dataclass_fields__
    assert "command" not in ExecutableCheck.__dataclass_fields__


def test_source_free_telemetry_and_identity(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text("def current(): return 2\n")
    record = ExecutableVerificationPlan("source-free", (_python_check(2),)).execute(
        tmp_path, 4
    )
    serialized = json.dumps(source_free_record(record))
    assert "def current" not in serialized
    assert "return 2" not in serialized
    assert record.generation == 4


def test_source_free_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.py").write_text("def current(): return 2\n")
    payload = source_free_record(
        ExecutableVerificationPlan("checkpoint", (_python_check(2),)).execute(
            workspace, 7
        )
    )
    checkpoint = tmp_path / "record.json"
    atomic_checkpoint(checkpoint, payload)
    assert resume_checkpoint(checkpoint) == payload


def _session(tmp_path: Path, expected: int) -> object:
    (tmp_path / "value.py").write_text("VALUE = 1\ndef current(): return VALUE\n")
    commands = ProjectCommands(
        test=ProjectCommand(
            (
                sys.executable,
                "-c",
                "import value; assert value.current() == 2",
            ),
            5,
        ),
        verification_plan=VerificationPlan("normal", ("project.test",)),
    )
    registry = create_assist_repository_registry(commands)
    rules = {item.name: PermissionDecision.ALLOW for item in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "structured_edit",
                    "path": "value.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ),
            json.dumps({"type": "final", "answer": "done"}),
        )
    )
    return RepositoryChatSession(
        "test",
        model,
        tmp_path,
        mode=AutonomyMode.ASSIST,
        registry=registry,
        policy=RuleBasedPolicy(rules),
        approval_callback=lambda _invocation, _proposal: True,
        verification_plan=commands.verification_plan,
        executable_verification_plan=ExecutableVerificationPlan(
            "bounded", (_python_check(expected),)
        ),
        required_candidate_paths=("value.py",),
        require_relevant_source=False,
    ).execute_task("Set VALUE to 2; current() must return 2.")


def test_production_lifecycle_passes_proven_check(tmp_path: Path) -> None:
    response = _session(tmp_path, 2)
    result = response.coding_task
    assert result is not None
    assert result.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert result.executable_verification[-1].outcome == "passed"


def test_proven_failure_changes_verification_outcome(tmp_path: Path) -> None:
    response = _session(tmp_path, 3)
    result = response.coding_task
    assert result is not None
    assert result.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    record = result.executable_verification[-1]
    assert record.outcome == "EXECUTABLE_CHECK_FAILURE"
    assert record.results[0].changed_verification_outcome


def test_proven_failure_uses_existing_repair_eligibility_path(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text("def current(): return 2\n")
    record = ExecutableVerificationPlan("failed", (_python_check(3),)).execute(
        tmp_path, 1
    )
    state = CodingTaskState(1, repair_enabled=True)
    state.mutation_count = 1
    state.executable_verification_finished(record)
    assert state.repair_eligible
    assert state.repair_eligibility_outcome == "EXECUTABLE_CHECK_FAILURE"


def test_a91_evaluator_and_results_remain_package_excluded() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'package-dir = { "" = "src" }' in configuration
    assert 'where = ["src"]' in configuration
    assert "benchmarks.bounded_executable_verification" not in configuration
    assert "eval-results" not in configuration

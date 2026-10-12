"""Bounded executable verification derived from trusted explicit contracts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from forge.process_isolation import ExecutionIsolationMode, ExecutionIsolationPolicy
from forge.tools.project import (
    PreparedProjectCommand,
    execute_prepared_project_command,
)
from forge.tools.tool import ToolError

MAX_CHECKS = 8
MAX_CASES = 8
MAX_TIMEOUT_SECONDS = 10.0
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class ExecutableCheckKind(StrEnum):
    EXPLICIT_EXAMPLE_ASSERTION = "EXPLICIT_EXAMPLE_ASSERTION"
    BOUNDARY_ASSERTION = "BOUNDARY_ASSERTION"
    INVARIANT_ASSERTION = "INVARIANT_ASSERTION"
    PUBLIC_API_INVOCATION = "PUBLIC_API_INVOCATION"
    ROUND_TRIP_ASSERTION = "ROUND_TRIP_ASSERTION"
    COMPILE_OR_USE_PROBE = "COMPILE_OR_USE_PROBE"


class CheckProvenance(StrEnum):
    EXPLICIT_TASK_EXAMPLE = "EXPLICIT_TASK_EXAMPLE"
    EXPLICIT_INVARIANT = "EXPLICIT_INVARIANT"
    EXPLICIT_BOUNDARY_REQUIREMENT = "EXPLICIT_BOUNDARY_REQUIREMENT"
    PUBLIC_API_CONTRACT = "PUBLIC_API_CONTRACT"
    TRUSTED_INTERFACE_CONTRACT = "TRUSTED_INTERFACE_CONTRACT"


class ExecutableCheckOutcome(StrEnum):
    CHECK_NOT_DERIVABLE = "CHECK_NOT_DERIVABLE"
    CHECK_CONSTRUCTION_ERROR = "CHECK_CONSTRUCTION_ERROR"
    CHECK_EXECUTION_ERROR = "CHECK_EXECUTION_ERROR"
    CHECK_TIMEOUT = "CHECK_TIMEOUT"
    CHECK_PASS = "CHECK_PASS"
    CHECK_FAIL = "CHECK_FAIL"
    HARNESS_CONSTRUCTION_ERROR = "HARNESS_CONSTRUCTION_ERROR"
    HARNESS_COMPILE_ERROR = "HARNESS_COMPILE_ERROR"
    HARNESS_LINK_ERROR = "HARNESS_LINK_ERROR"
    CHECK_RUNTIME_ERROR = "CHECK_RUNTIME_ERROR"


class CCompiler(StrEnum):
    CC = "cc"
    CLANG = "clang"
    GCC = "gcc"


@dataclass(frozen=True, slots=True)
class CFunctionDeclaration:
    symbol: str
    return_type: str
    parameter_types: tuple[str, ...]
    header_identity: str


@dataclass(frozen=True, slots=True)
class PythonInvocation:
    module: str
    symbol: str
    arguments: tuple[object, ...] = ()
    keyword_arguments: tuple[tuple[str, object], ...] = ()
    expected: object = None
    expected_exception: str | None = None

    def __post_init__(self) -> None:
        if _MODULE.fullmatch(self.module) is None:
            raise ValueError("invalid public Python module identity")
        if _IDENTIFIER.fullmatch(self.symbol) is None:
            raise ValueError("invalid public Python symbol identity")
        if (
            self.expected_exception is not None
            and _IDENTIFIER.fullmatch(self.expected_exception) is None
        ):
            raise ValueError("invalid expected exception identity")
        _json_value(self.arguments)
        _json_value(dict(self.keyword_arguments))
        _json_value(self.expected)


@dataclass(frozen=True, slots=True)
class PythonMethodCall:
    symbol: str
    arguments: tuple[object, ...] = ()
    keyword_arguments: tuple[tuple[str, object], ...] = ()
    expected: object = None
    expected_exception: str | None = None

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.symbol) is None:
            raise ValueError("invalid public Python method identity")
        if (
            self.expected_exception is not None
            and _IDENTIFIER.fullmatch(self.expected_exception) is None
        ):
            raise ValueError("invalid expected exception identity")
        _json_value(self.arguments)
        _json_value(dict(self.keyword_arguments))
        _json_value(self.expected)


@dataclass(frozen=True, slots=True)
class PythonObjectScenario:
    module: str
    symbol: str
    arguments: tuple[object, ...] = ()
    keyword_arguments: tuple[tuple[str, object], ...] = ()
    calls: tuple[PythonMethodCall, ...] = ()

    def __post_init__(self) -> None:
        if (
            _MODULE.fullmatch(self.module) is None
            or _IDENTIFIER.fullmatch(self.symbol) is None
        ):
            raise ValueError("invalid public Python object identity")
        if not self.calls or len(self.calls) > MAX_CASES:
            raise ValueError("object scenario requires bounded method calls")
        _json_value(self.arguments)
        _json_value(dict(self.keyword_arguments))


@dataclass(frozen=True, slots=True)
class CInvocation:
    header: str
    sources: tuple[str, ...]
    symbol: str
    arguments: tuple[int, ...]
    expected: int

    def __post_init__(self) -> None:
        if not _safe_relative(self.header) or not self.header.endswith(".h"):
            raise ValueError("C header must be a confined relative header")
        if not self.sources or any(
            not _safe_relative(source) or not source.endswith(".c")
            for source in self.sources
        ):
            raise ValueError("C sources must be confined relative C files")
        if _IDENTIFIER.fullmatch(self.symbol) is None:
            raise ValueError("invalid public C symbol identity")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.arguments
        ):
            raise ValueError("C probe arguments must be bounded integers")
        if isinstance(self.expected, bool) or not isinstance(self.expected, int):
            raise ValueError("C probe expected result must be an integer")


@dataclass(frozen=True, slots=True)
class CByteArrayInvocation:
    header: str
    sources: tuple[str, ...]
    symbol: str
    values: tuple[int, ...]
    expected: int

    def __post_init__(self) -> None:
        if not _safe_relative(self.header) or not self.header.endswith(".h"):
            raise ValueError("C header must be a confined relative header")
        if not self.sources or any(
            not _safe_relative(source) or not source.endswith(".c")
            for source in self.sources
        ):
            raise ValueError("C sources must be confined relative C files")
        if _IDENTIFIER.fullmatch(self.symbol) is None:
            raise ValueError("invalid public C symbol identity")
        if not self.values or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.values
        ):
            raise ValueError("C byte-array probe values must be bytes")


@dataclass(frozen=True, slots=True)
class ExecutableCheck:
    check_id: str
    kind: ExecutableCheckKind
    provenance: CheckProvenance
    component_identity: str
    input_identity: str
    expected_identity: str
    timeout_seconds: float
    python_cases: tuple[PythonInvocation, ...] = ()
    python_scenario: PythonObjectScenario | None = None
    c_case: CInvocation | None = None
    c_byte_array_case: CByteArrayInvocation | None = None
    c_declaration: CFunctionDeclaration | None = None
    compiler: CCompiler = CCompiler.CC
    derivation_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ExecutableCheckKind):
            raise TypeError("executable check kind must be typed")
        if not isinstance(self.provenance, CheckProvenance):
            raise TypeError("executable check provenance must be trusted and typed")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.check_id) is None:
            raise ValueError("invalid executable check ID")
        if not self.component_identity or len(self.component_identity) > 128:
            raise ValueError("invalid component identity")
        for identity in (self.input_identity, self.expected_identity):
            if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise ValueError("check input/expected identities must be SHA-256")
        if not 0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError("executable check timeout is outside the bounded limit")
        if (
            sum(
                (
                    bool(self.python_cases),
                    self.python_scenario is not None,
                    self.c_case is not None,
                    self.c_byte_array_case is not None,
                )
            )
            != 1
        ):
            raise ValueError("check must contain exactly one typed execution form")
        if len(self.python_cases) > MAX_CASES:
            raise ValueError("too many executable check cases")
        if not isinstance(self.compiler, CCompiler):
            raise TypeError("C compiler must come from the fixed allowlist")


@dataclass(frozen=True, slots=True)
class ExecutableCheckResult:
    check_id: str
    check_identity: str
    kind: str
    provenance: str
    component_identity: str
    outcome: str
    duration_seconds: float
    subprocess_count: int
    changed_verification_outcome: bool = False
    derivation_latency_seconds: float = 0.0
    construction_latency_seconds: float = 0.0
    compile_latency_seconds: float = 0.0
    link_latency_seconds: float = 0.0
    execution_latency_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutableVerificationRecord:
    derived_checks: int = 0
    results: tuple[ExecutableCheckResult, ...] = ()
    outcome: str = "not_run"
    duration_seconds: float = 0.0
    subprocess_count: int = 0
    generation: int | None = None

    @property
    def proven_failure(self) -> bool:
        return any(
            item.outcome == ExecutableCheckOutcome.CHECK_FAIL.value
            for item in self.results
        )


@dataclass(frozen=True, slots=True)
class ExecutableVerificationPlan:
    plan_id: str
    checks: tuple[ExecutableCheck, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.plan_id) is None:
            raise ValueError("invalid executable verification plan ID")
        if len(self.checks) > MAX_CHECKS:
            raise ValueError("too many executable checks")
        if len({item.check_id for item in self.checks}) != len(self.checks):
            raise ValueError("executable check IDs must be unique")

    def execute(self, workspace: Path, generation: int) -> ExecutableVerificationRecord:
        root = workspace.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("executable verification workspace must be a directory")
        started = time.monotonic()
        results = tuple(_execute(check, root) for check in self.checks)
        return ExecutableVerificationRecord(
            len(self.checks),
            results,
            "EXECUTABLE_CHECK_FAILURE"
            if any(
                item.outcome == ExecutableCheckOutcome.CHECK_FAIL for item in results
            )
            else "passed",
            time.monotonic() - started,
            sum(item.subprocess_count for item in results),
            generation,
        )


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def derive_python_check(
    *,
    check_id: str,
    kind: ExecutableCheckKind | str,
    provenance: CheckProvenance | str,
    component_identity: str,
    cases: tuple[PythonInvocation, ...],
    timeout_seconds: float = 5.0,
) -> tuple[ExecutableCheck | None, ExecutableCheckOutcome | None]:
    """Construct one check only from an explicitly trusted typed contract."""
    try:
        typed_kind = ExecutableCheckKind(kind)
        typed_provenance = CheckProvenance(provenance)
    except (TypeError, ValueError):
        return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
    if not cases:
        return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
    try:
        payload = [
            {
                "module": item.module,
                "symbol": item.symbol,
                "arguments": item.arguments,
                "keyword_arguments": item.keyword_arguments,
            }
            for item in cases
        ]
        expected = [
            {"expected": item.expected, "exception": item.expected_exception}
            for item in cases
        ]
        return (
            ExecutableCheck(
                check_id,
                typed_kind,
                typed_provenance,
                component_identity,
                identity(payload),
                identity(expected),
                timeout_seconds,
                python_cases=cases,
            ),
            None,
        )
    except (TypeError, ValueError):
        return None, ExecutableCheckOutcome.CHECK_CONSTRUCTION_ERROR


def derive_python_object_check(
    *,
    check_id: str,
    kind: ExecutableCheckKind | str,
    provenance: CheckProvenance | str,
    component_identity: str,
    scenario: PythonObjectScenario,
    timeout_seconds: float = 5.0,
) -> tuple[ExecutableCheck | None, ExecutableCheckOutcome | None]:
    try:
        typed_kind = ExecutableCheckKind(kind)
        typed_provenance = CheckProvenance(provenance)
    except (TypeError, ValueError):
        return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
    try:
        inputs = {
            "module": scenario.module,
            "symbol": scenario.symbol,
            "arguments": scenario.arguments,
            "calls": [
                (call.symbol, call.arguments, call.keyword_arguments)
                for call in scenario.calls
            ],
        }
        expected = [(call.expected, call.expected_exception) for call in scenario.calls]
        return (
            ExecutableCheck(
                check_id,
                typed_kind,
                typed_provenance,
                component_identity,
                identity(inputs),
                identity(expected),
                timeout_seconds,
                python_scenario=scenario,
            ),
            None,
        )
    except (TypeError, ValueError):
        return None, ExecutableCheckOutcome.CHECK_CONSTRUCTION_ERROR


def derive_c_check(
    workspace: Path,
    *,
    check_id: str,
    kind: ExecutableCheckKind | str,
    provenance: CheckProvenance | str,
    component_identity: str,
    invocation: CInvocation | CByteArrayInvocation,
    compiler: CCompiler = CCompiler.CC,
    timeout_seconds: float = 5.0,
) -> tuple[ExecutableCheck | None, ExecutableCheckOutcome | None]:
    """Freeze a supported invocation against one trusted public declaration."""
    started = time.monotonic()
    try:
        typed_kind = ExecutableCheckKind(kind)
        typed_provenance = CheckProvenance(provenance)
    except (TypeError, ValueError):
        return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
    try:
        root = workspace.resolve(strict=True)
        header = (root / invocation.header).resolve(strict=True)
        if root not in header.parents or not header.is_file():
            return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
        declaration = extract_c_declaration(header.read_text(), invocation.symbol)
        if declaration is None or not _invocation_matches(declaration, invocation):
            return None, ExecutableCheckOutcome.CHECK_NOT_DERIVABLE
        declaration = replace(
            declaration, header_identity=hashlib.sha256(header.read_bytes()).hexdigest()
        )
        inputs = (
            {"arguments": invocation.arguments}
            if isinstance(invocation, CInvocation)
            else {"byte_count": len(invocation.values)}
        )
        check = ExecutableCheck(
            check_id,
            typed_kind,
            typed_provenance,
            component_identity,
            identity(inputs),
            identity({"expected": invocation.expected}),
            timeout_seconds,
            c_case=invocation if isinstance(invocation, CInvocation) else None,
            c_byte_array_case=(
                invocation if isinstance(invocation, CByteArrayInvocation) else None
            ),
            c_declaration=declaration,
            compiler=compiler,
            derivation_latency_seconds=time.monotonic() - started,
        )
        return check, None
    except (OSError, TypeError, ValueError):
        return None, ExecutableCheckOutcome.HARNESS_CONSTRUCTION_ERROR


def extract_c_declaration(header: str, symbol: str) -> CFunctionDeclaration | None:
    """Extract one intentionally small, unambiguous public C function shape."""
    if _IDENTIFIER.fullmatch(symbol) is None:
        return None
    cleaned = re.sub(r"/\*.*?\*/|//[^\n]*", " ", header, flags=re.DOTALL)
    pattern = re.compile(
        rf"\b(?P<return>unsigned|int|size_t|uint64_t|bool)\s+{re.escape(symbol)}"
        r"\s*\((?P<parameters>[^();]*)\)\s*;"
    )
    matches = tuple(pattern.finditer(cleaned))
    if len(matches) != 1:
        return None
    parameters = matches[0].group("parameters").strip()
    types: tuple[str, ...]
    if not parameters or parameters == "void":
        types = ()
    else:
        parsed = tuple(_c_parameter_type(item) for item in parameters.split(","))
        if any(item is None for item in parsed):
            return None
        types = tuple(item for item in parsed if item is not None)
    return CFunctionDeclaration(
        symbol,
        matches[0].group("return"),
        types,
        identity({"declaration": matches[0].group(0)}),
    )


def _c_parameter_type(parameter: str) -> str | None:
    normalized = " ".join(parameter.strip().replace("*", " * ").split())
    supported = (
        "const unsigned char *",
        "unsigned char const *",
        "unsigned",
        "int",
        "size_t",
        "uint64_t",
        "bool",
    )
    for value in supported:
        if normalized == value or normalized.startswith(value + " "):
            remainder = normalized[len(value) :].strip()
            if not remainder or _IDENTIFIER.fullmatch(remainder):
                return "const unsigned char *" if "char" in value else value
    return None


def _invocation_matches(
    declaration: CFunctionDeclaration,
    invocation: CInvocation | CByteArrayInvocation,
) -> bool:
    if declaration.return_type not in {"unsigned", "int", "size_t", "uint64_t", "bool"}:
        return False
    if isinstance(invocation, CByteArrayInvocation):
        return declaration.parameter_types == ("const unsigned char *", "size_t")
    return len(declaration.parameter_types) == len(invocation.arguments) and all(
        item in {"unsigned", "int", "size_t", "uint64_t", "bool"}
        for item in declaration.parameter_types
    )


def source_free_record(record: ExecutableVerificationRecord) -> dict[str, object]:
    return {
        "derived_checks": record.derived_checks,
        "results": [
            {
                "check_id": item.check_id,
                "check_identity": item.check_identity,
                "kind": item.kind,
                "provenance": item.provenance,
                "component_identity": item.component_identity,
                "outcome": item.outcome,
                "duration_seconds": item.duration_seconds,
                "subprocess_count": item.subprocess_count,
                "changed_verification_outcome": item.changed_verification_outcome,
                "derivation_latency_seconds": item.derivation_latency_seconds,
                "construction_latency_seconds": item.construction_latency_seconds,
                "compile_latency_seconds": item.compile_latency_seconds,
                "link_latency_seconds": item.link_latency_seconds,
                "execution_latency_seconds": item.execution_latency_seconds,
            }
            for item in record.results
        ],
        "outcome": record.outcome,
        "duration_seconds": record.duration_seconds,
        "subprocess_count": record.subprocess_count,
        "generation": record.generation,
    }


def _execute(check: ExecutableCheck, workspace: Path) -> ExecutableCheckResult:
    if check.c_case is not None or check.c_byte_array_case is not None:
        return _execute_c(check, workspace)
    started = time.monotonic()
    execution_started = started
    try:
        argv = (
            sys.executable,
            "-I",
            "-B",
            "-c",
            _python_harness(check.python_cases, check.python_scenario),
        )
        count = 1
        prepared = PreparedProjectCommand(
            "test",
            argv,
            workspace,
            check.timeout_seconds,
            ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
        )
        execute_prepared_project_command(prepared)
        outcome = ExecutableCheckOutcome.CHECK_PASS
    except ToolError as error:
        payload = error.output if isinstance(error.output, dict) else {}
        raw = payload.get("outcome")
        exit_code = payload.get("exit_code")
        outcome = (
            ExecutableCheckOutcome.CHECK_TIMEOUT
            if raw == "timeout"
            else ExecutableCheckOutcome.CHECK_FAIL
            if raw == "nonzero_exit"
            and isinstance(exit_code, int)
            and (exit_code == 10 or exit_code < 0)
            else ExecutableCheckOutcome.CHECK_RUNTIME_ERROR
        )
        count = locals().get("count", 0)
    except (OSError, TypeError, ValueError):
        outcome = ExecutableCheckOutcome.CHECK_CONSTRUCTION_ERROR
        count = 0
    return ExecutableCheckResult(
        check.check_id,
        identity(
            {
                "id": check.check_id,
                "kind": check.kind.value,
                "provenance": check.provenance.value,
                "component": check.component_identity,
                "input": check.input_identity,
                "expected": check.expected_identity,
                "timeout": check.timeout_seconds,
            }
        ),
        check.kind.value,
        check.provenance.value,
        check.component_identity,
        outcome.value,
        time.monotonic() - started,
        count,
        False,
        check.derivation_latency_seconds,
        0.0,
        0.0,
        0.0,
        time.monotonic() - execution_started,
    )


def _execute_c(check: ExecutableCheck, workspace: Path) -> ExecutableCheckResult:
    started = time.monotonic()
    construction = compile_latency = link_latency = execution_latency = 0.0
    subprocess_count = 0
    outcome = ExecutableCheckOutcome.HARNESS_CONSTRUCTION_ERROR
    try:
        construction_started = time.monotonic()
        case, harness, binary, sources = _construct_c_harness(check, workspace)
        construction = time.monotonic() - construction_started
        objects: list[Path] = []
        compile_started = time.monotonic()
        for index, source in enumerate((*sources, harness)):
            output = harness.parent / f"{check.check_id}-{index}.o"
            subprocess_count += 1
            _execute_c_command(
                check,
                workspace,
                (
                    check.compiler.value,
                    "-std=c17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(workspace),
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ),
                phase="compile",
            )
            objects.append(output)
        compile_latency = time.monotonic() - compile_started
        link_started = time.monotonic()
        subprocess_count += 1
        _execute_c_command(
            check,
            workspace,
            (check.compiler.value, *(str(item) for item in objects), "-o", str(binary)),
            phase="link",
        )
        link_latency = time.monotonic() - link_started
        execution_started = time.monotonic()
        subprocess_count += 1
        try:
            _execute_c_command(check, workspace, (str(binary),))
            outcome = ExecutableCheckOutcome.CHECK_PASS
        except ToolError as error:
            outcome = _runtime_outcome(error)
        execution_latency = time.monotonic() - execution_started
    except _CPhaseFailure as failure:
        if failure.phase == "compile":
            compile_latency = failure.duration_seconds
            outcome = (
                ExecutableCheckOutcome.CHECK_TIMEOUT
                if failure.timed_out
                else ExecutableCheckOutcome.HARNESS_COMPILE_ERROR
            )
        else:
            link_latency = failure.duration_seconds
            outcome = (
                ExecutableCheckOutcome.CHECK_TIMEOUT
                if failure.timed_out
                else ExecutableCheckOutcome.HARNESS_LINK_ERROR
            )
    except (OSError, TypeError, ValueError):
        construction = time.monotonic() - started
        outcome = ExecutableCheckOutcome.HARNESS_CONSTRUCTION_ERROR
    return _result(
        check,
        outcome,
        time.monotonic() - started,
        subprocess_count,
        construction,
        compile_latency,
        link_latency,
        execution_latency,
    )


@dataclass(frozen=True, slots=True)
class _CPhaseFailure(Exception):
    phase: str
    timed_out: bool
    duration_seconds: float


def _execute_c_command(
    check: ExecutableCheck,
    workspace: Path,
    argv: tuple[str, ...],
    *,
    phase: str | None = None,
) -> None:
    prepared = PreparedProjectCommand(
        "test",
        argv,
        workspace,
        check.timeout_seconds,
        ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT),
    )
    phase_started = time.monotonic()
    try:
        execute_prepared_project_command(prepared)
    except ToolError as error:
        if phase is None:
            raise
        payload = error.output if isinstance(error.output, dict) else {}
        raise _CPhaseFailure(
            phase,
            payload.get("outcome") == "timeout",
            time.monotonic() - phase_started,
        ) from error


def _runtime_outcome(error: ToolError) -> ExecutableCheckOutcome:
    payload = error.output if isinstance(error.output, dict) else {}
    if payload.get("outcome") == "timeout":
        return ExecutableCheckOutcome.CHECK_TIMEOUT
    if payload.get("outcome") == "nonzero_exit" and payload.get("exit_code") == 10:
        return ExecutableCheckOutcome.CHECK_FAIL
    return ExecutableCheckOutcome.CHECK_RUNTIME_ERROR


def _result(
    check: ExecutableCheck,
    outcome: ExecutableCheckOutcome,
    duration: float,
    subprocess_count: int,
    construction: float,
    compile_latency: float,
    link_latency: float,
    execution_latency: float,
) -> ExecutableCheckResult:
    return ExecutableCheckResult(
        check.check_id,
        identity(
            {
                "id": check.check_id,
                "kind": check.kind.value,
                "provenance": check.provenance.value,
                "component": check.component_identity,
                "input": check.input_identity,
                "expected": check.expected_identity,
                "timeout": check.timeout_seconds,
            }
        ),
        check.kind.value,
        check.provenance.value,
        check.component_identity,
        outcome.value,
        duration,
        subprocess_count,
        False,
        check.derivation_latency_seconds,
        construction,
        compile_latency,
        link_latency,
        execution_latency,
    )


def _python_harness(
    cases: tuple[PythonInvocation, ...], scenario: PythonObjectScenario | None
) -> str:
    payload = json.dumps(
        [
            {
                "module": case.module,
                "symbol": case.symbol,
                "args": case.arguments,
                "kwargs": dict(case.keyword_arguments),
                "expected": case.expected,
                "exception": case.expected_exception,
            }
            for case in cases
        ]
    )
    scenario_payload = json.dumps(
        None
        if scenario is None
        else {
            "module": scenario.module,
            "symbol": scenario.symbol,
            "args": scenario.arguments,
            "kwargs": dict(scenario.keyword_arguments),
            "calls": [
                {
                    "symbol": call.symbol,
                    "args": call.arguments,
                    "kwargs": dict(call.keyword_arguments),
                    "expected": call.expected,
                    "exception": call.expected_exception,
                }
                for call in scenario.calls
            ],
        }
    )
    return (
        "import builtins,importlib,json,os,sys\n"
        "sys.path.insert(0,os.getcwd())\n"
        "def audit(event,args):\n"
        " if event.startswith('socket.') or event in "
        "{'subprocess.Popen','os.system','os.posix_spawn'}: "
        "raise PermissionError('operation denied')\n"
        " if event=='open' and len(args)>1 and any(x in str(args[1]) for x in 'wax+'): "
        "raise PermissionError('write denied')\n"
        "sys.addaudithook(audit)\n"
        f"cases=json.loads({payload!r})\n"
        f"scenario=json.loads({scenario_payload!r})\n"
        "try:\n"
        " for c in cases:\n"
        "  f=getattr(importlib.import_module(c['module']),c['symbol'])\n"
        "  if c['exception']:\n"
        "   try: f(*c['args'],**c['kwargs'])\n"
        "   except getattr(builtins,c['exception']): pass\n"
        "   else: raise AssertionError('expected exception')\n"
        "  else: assert f(*c['args'],**c['kwargs'])==c['expected']\n"
        " if scenario:\n"
        "  module=importlib.import_module(scenario['module'])\n"
        "  cls=getattr(module,scenario['symbol'])\n"
        "  obj=cls(*scenario['args'],**scenario['kwargs'])\n"
        "  for c in scenario['calls']:\n"
        "   f=getattr(obj,c['symbol'])\n"
        "   if c['exception']:\n"
        "    try: f(*c['args'],**c['kwargs'])\n"
        "    except getattr(builtins,c['exception']): pass\n"
        "    else: raise AssertionError('expected exception')\n"
        "   else: assert f(*c['args'],**c['kwargs'])==c['expected']\n"
        "except AssertionError: sys.exit(10)\n"
        "except BaseException: sys.exit(11)\n"
    )


def _construct_c_harness(
    check: ExecutableCheck, workspace: Path
) -> tuple[
    CInvocation | CByteArrayInvocation,
    Path,
    Path,
    tuple[Path, ...],
]:
    assert check.c_case is not None or check.c_byte_array_case is not None
    case = check.c_case or check.c_byte_array_case
    assert case is not None
    paths = tuple((workspace / source).resolve(strict=True) for source in case.sources)
    header = (workspace / case.header).resolve(strict=True)
    if any(workspace not in path.parents for path in (*paths, header)):
        raise ValueError("C probe path escaped workspace")
    declaration = check.c_declaration
    if declaration is None:
        raise ValueError("C probe lacks frozen header declaration")
    if hashlib.sha256(header.read_bytes()).hexdigest() != declaration.header_identity:
        raise ValueError("trusted C header changed after derivation")
    observed = extract_c_declaration(header.read_text(), case.symbol)
    if observed is None or (
        observed.return_type,
        observed.parameter_types,
    ) != (declaration.return_type, declaration.parameter_types):
        raise ValueError("C public declaration changed after derivation")
    scratch = workspace / ".forge-exec"
    scratch.mkdir(mode=0o700, exist_ok=True)
    harness = scratch / f"{check.check_id}.c"
    binary = scratch / f"{check.check_id}.bin"
    if isinstance(case, CByteArrayInvocation):
        values = ",".join(str(value) for value in case.values)
        array_declaration = f"const unsigned char data[]={{{values}}};"
        arguments = f"data,{len(case.values)}"
    else:
        array_declaration = ""
        arguments = ",".join(str(value) for value in case.arguments)
    harness.write_text(
        f'#include "{case.header}"\n'
        f"int main(void){{{array_declaration}"
        f"return {case.symbol}({arguments})=={case.expected}?0:10;}}\n"
    )
    return case, harness, binary, paths


def _safe_relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _json_value(value: object) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise ValueError("check values must be bounded JSON values") from error

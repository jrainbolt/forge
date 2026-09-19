"""Experimental human-gated, task-local acceptance evidence for Python projects.

Approval authorizes execution of one exact model-generated assertion script. It is
not a semantic correctness judgment and never grants a repository tool capability.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    OutputSpecification,
    ResponseFormat,
)

MAX_TEST_SOURCE_CHARS = 6000
MAX_CONTEXT_BYTES = 32_000
TEST_TIMEOUT_SECONDS = 10.0
MAX_FAILURE_OUTPUT = 2000


class EphemeralAcceptanceMode(StrEnum):
    OFF = "off"
    OPTIONAL = "optional"
    REQUIRED = "required"


class EphemeralAcceptanceState(StrEnum):
    OFF = "off"
    UNAVAILABLE = "unavailable"
    BASELINE_NOT_DETECTED = "baseline_not_detected"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"
    POSTMUTATION_PASS = "postmutation_pass"
    POSTMUTATION_FAIL = "postmutation_fail"
    POSTMUTATION_ERROR = "postmutation_error"


@dataclass(frozen=True, slots=True)
class EphemeralAcceptanceCandidate:
    test_name: str
    test_source: str
    language: str = "Python"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.test_source.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EphemeralAcceptancePreview:
    task: str
    test_name: str
    test_source: str
    language: str
    execution_description: str
    baseline_result: str
    baseline_failure_output: str
    context_files: tuple[str, ...]
    candidate_sha256: str
    warning: str = (
        "MODEL-GENERATED, SEMANTICALLY UNVERIFIED. Approval authorizes only this "
        "exact ephemeral check for this task and source state; full project "
        "verification remains required."
    )


@dataclass(frozen=True, slots=True)
class EphemeralAcceptanceApproval:
    """Separate from WRITE/TEST/BUILD/CONFIGURE InvocationApproval."""

    identity: str
    candidate_sha256: str
    task_sha256: str
    workspace: str
    generation: int


@dataclass(frozen=True, slots=True)
class EphemeralAcceptanceMetrics:
    used: bool = False
    candidate_sha256: str | None = None
    candidate_size: int = 0
    baseline_outcome: str = "not_run"
    approval_outcome: str = "not_requested"
    postmutation_outcome: str = "not_run"
    generation_latency_seconds: float = 0.0
    baseline_latency_seconds: float = 0.0
    postmutation_latency_seconds: float = 0.0
    generation_calls: int = 0


def _confined_source(workspace: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("acceptance context path is not confined")
    if any(
        part.startswith(".")
        or part.casefold() in {"eval-results", "benchmarks", "oracle"}
        for part in path.parts
    ):
        raise ValueError("acceptance context path is not production-visible source")
    root = workspace.resolve(strict=True)
    target = root.joinpath(*path.parts)
    if target.is_symlink():
        raise ValueError("acceptance context may not follow symlinks")
    resolved = target.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("acceptance context is not a workspace file")
    return resolved


def _context_hashes(
    workspace: Path, paths: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            path,
            hashlib.sha256(_confined_source(workspace, path).read_bytes()).hexdigest(),
        )
        for path in paths
    )


def _candidate_valid(candidate: EphemeralAcceptanceCandidate, import_root: str) -> bool:
    source = candidate.test_source
    if (
        candidate.language != "Python"
        or not candidate.test_name.strip()
        or not source.strip()
        or len(source) > MAX_TEST_SOURCE_CHARS
        or any(
            marker in source.casefold()
            for marker in (
                "hidden oracle",
                "reference mutation",
                "wrong mutation",
                "oracle.py",
                "../",
                "/users/",
                "eval-results",
                "api_key",
            )
        )
    ):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    if not any(isinstance(node, ast.Assert) for node in tree.body):
        return False
    if any(
        not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assert))
        for node in tree.body
    ):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if not modules or any(
                module != import_root and not module.startswith(import_root + ".")
                for module in modules
            ):
                return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
        if isinstance(node, ast.Attribute) and node.attr in {
            "system",
            "popen",
            "spawn",
            "run",
            "Popen",
            "environ",
            "getenv",
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
            "unlink",
            "remove",
            "rmdir",
            "mkdir",
            "connect",
            "urlopen",
            "request",
        }:
            return False
        if isinstance(node, ast.Name) and (
            node.id.startswith("__")
            or node.id
            in {
                "open",
                "exec",
                "eval",
                "compile",
                "__import__",
                "globals",
                "locals",
                "vars",
                "getattr",
                "setattr",
                "delattr",
                "input",
                "breakpoint",
            }
        ):
            return False
        if isinstance(
            node,
            (
                ast.Lambda,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.With,
                ast.AsyncWith,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Global,
                ast.Nonlocal,
                ast.NamedExpr,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            return False
    return True


def _execute_python_candidate(
    candidate: EphemeralAcceptanceCandidate, workspace: Path
) -> tuple[str, str, float]:
    """Run a revalidated test from a temporary location, never project source."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="forge-ephemeral-acceptance-") as name:
        temporary = Path(name)
        script = temporary / "candidate.py"
        script.write_text(candidate.test_source, encoding="utf-8")
        environment = {
            "PATH": os.pathsep.join(("/usr/bin", "/bin", "/usr/sbin", "/sbin")),
            "PYTHONPATH": str(workspace.resolve(strict=True)),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": str(temporary),
            "TMPDIR": str(temporary),
            "LANG": "C",
        }
        try:
            completed = subprocess.run(
                (sys.executable, "-B", str(script)),
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                timeout=TEST_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return (
                "error",
                "execution unavailable or timed out",
                time.perf_counter() - started,
            )
        output = (completed.stderr or completed.stdout).decode(
            "utf-8", errors="replace"
        )
        return (
            "pass"
            if completed.returncode == 0
            else "fail"
            if completed.returncode == 1 and "AssertionError" in output
            else "error",
            output[:MAX_FAILURE_OUTPUT],
            time.perf_counter() - started,
        )


def _approval_identity(
    task: str,
    candidate: EphemeralAcceptanceCandidate,
    workspace: Path,
    generation: int,
    hashes: tuple[tuple[str, str], ...],
    baseline_output: str,
    execution_description: str,
) -> str:
    material = (
        task,
        candidate.test_name,
        candidate.test_source,
        candidate.sha256,
        candidate.language,
        str(workspace.resolve(strict=True)),
        generation,
        hashes,
        "fail",
        hashlib.sha256(baseline_output.encode()).hexdigest(),
        execution_description,
    )
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class EphemeralAcceptanceGate:
    """One task-local gate. No profile or autonomy setting can approve it."""

    def __init__(
        self,
        mode: EphemeralAcceptanceMode = EphemeralAcceptanceMode.OFF,
        *,
        context_paths: tuple[str, ...] = (),
        import_root: str = "",
    ) -> None:
        self.mode = EphemeralAcceptanceMode(mode)
        self.context_paths = tuple(context_paths)
        self.import_root = import_root
        if self.mode is not EphemeralAcceptanceMode.OFF and (
            len(self.context_paths) < 2
            or len(self.context_paths) > 4
            or not import_root.isidentifier()
        ):
            raise ValueError(
                "ephemeral mode requires 2-4 trusted paths and an import root"
            )
        self.state = EphemeralAcceptanceState.OFF
        self.metrics = EphemeralAcceptanceMetrics()
        self.events: list[str] = []
        self._attempted = False
        self._prepared_once = False
        self._candidate: EphemeralAcceptanceCandidate | None = None
        self._approval: EphemeralAcceptanceApproval | None = None
        self._hashes: tuple[tuple[str, str], ...] = ()
        self._preview: EphemeralAcceptancePreview | None = None

    @property
    def preview(self) -> EphemeralAcceptancePreview | None:
        return self._preview

    @property
    def approved(self) -> bool:
        return self.state is EphemeralAcceptanceState.APPROVED

    def generate(
        self,
        model: Model,
        task: str,
        workspace: Path,
        generation: int,
        config: GenerationConfig,
        review: Callable[[EphemeralAcceptancePreview], bool] | None,
    ) -> EphemeralAcceptanceState:
        if self.mode is EphemeralAcceptanceMode.OFF:
            return self.state
        if self._attempted:
            raise RuntimeError("one acceptance candidate attempt per task")
        self._attempted = True
        if review is None:
            self.state = EphemeralAcceptanceState.UNAVAILABLE
            return self.state
        try:
            sources = []
            for path in self.context_paths:
                content = _confined_source(workspace, path).read_bytes()
                if len(content) > MAX_CONTEXT_BYTES:
                    raise ValueError("acceptance context exceeds limit")
                sources.append(
                    f"BEGIN VISIBLE FILE {path}\n"
                    f"{content.decode('utf-8')}\nEND VISIBLE FILE"
                )
        except (OSError, UnicodeError, ValueError):
            self.state = EphemeralAcceptanceState.UNAVAILABLE
            return self.state
        prompt = "\n\n".join(
            (
                "Generate one focused standalone Python assertion script "
                "for this task. "
                "Return JSON with test_name and test_source only. Do not explain. "
                "Do not read files, invoke commands, or access network.",
                f"Task: {task}",
                *sources,
            )
        )
        # A conservative character-per-token upper bound prevents this auxiliary
        # request from increasing the configured model context ceiling.
        if len(prompt) + config.max_tokens + 512 > (model.context_capacity or 4096):
            self.state = EphemeralAcceptanceState.UNAVAILABLE
            return self.state
        schema = {
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "test_source": {"type": "string"},
            },
            "required": ["test_name", "test_source"],
            "additionalProperties": False,
        }
        started = time.perf_counter()
        response = model.generate(
            ModelRequest(
                (Message(MessageRole.USER, prompt),),
                GenerationConfig(
                    max_tokens=config.max_tokens,
                    temperature=0.0,
                    seed=config.seed,
                ),
                OutputSpecification(ResponseFormat.JSON, schema),
            )
        )
        self.metrics = EphemeralAcceptanceMetrics(
            generation_latency_seconds=time.perf_counter() - started,
            generation_calls=1,
        )
        try:
            value = json.loads(response.text)
            if not isinstance(value, dict) or set(value) != {
                "test_name",
                "test_source",
            }:
                raise ValueError
            if not all(isinstance(value[key], str) for key in value):
                raise ValueError
            candidate = EphemeralAcceptanceCandidate(
                value["test_name"], value["test_source"]
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            self.state = EphemeralAcceptanceState.UNAVAILABLE
            return self.state
        return self.prepare_candidate(candidate, task, workspace, generation, review)

    def prepare_candidate(
        self,
        candidate: EphemeralAcceptanceCandidate,
        task: str,
        workspace: Path,
        generation: int,
        review: Callable[[EphemeralAcceptancePreview], bool] | None,
    ) -> EphemeralAcceptanceState:
        if self.mode is EphemeralAcceptanceMode.OFF:
            return self.state
        if self._prepared_once:
            raise RuntimeError("acceptance candidate already prepared")
        self._prepared_once = True
        self._attempted = True
        self.events.append("ephemeral_test_generated")
        if review is None or not _candidate_valid(candidate, self.import_root):
            self.state = EphemeralAcceptanceState.UNAVAILABLE
            return self.state
        hashes = _context_hashes(workspace, self.context_paths)
        baseline, output, duration = _execute_python_candidate(candidate, workspace)
        self.metrics = EphemeralAcceptanceMetrics(
            candidate_sha256=candidate.sha256,
            candidate_size=len(candidate.test_source.encode()),
            baseline_outcome=baseline,
            generation_latency_seconds=self.metrics.generation_latency_seconds,
            baseline_latency_seconds=duration,
            generation_calls=self.metrics.generation_calls,
        )
        if _context_hashes(workspace, self.context_paths) != hashes:
            self.state = EphemeralAcceptanceState.INVALIDATED
            self.events.append("ephemeral_test_invalidated")
            return self.state
        if baseline != "fail":
            self.state = (
                EphemeralAcceptanceState.BASELINE_NOT_DETECTED
                if baseline == "pass"
                else EphemeralAcceptanceState.UNAVAILABLE
            )
            return self.state
        self.events.append("ephemeral_test_baseline_failed")
        description = (
            "Python assertion script, trusted interpreter, -B, 10-second timeout"
        )
        preview = EphemeralAcceptancePreview(
            task,
            candidate.test_name,
            candidate.test_source,
            candidate.language,
            description,
            "FAIL",
            output,
            self.context_paths,
            candidate.sha256,
        )
        self._candidate = candidate
        self._hashes = hashes
        self._preview = preview
        self.state = EphemeralAcceptanceState.REVIEW_REQUIRED
        self.events.append("ephemeral_test_review_required")
        approved = review(preview)
        if not approved:
            self.state = EphemeralAcceptanceState.REJECTED
            self.events.append("ephemeral_test_rejected")
            self.metrics = EphemeralAcceptanceMetrics(
                candidate_sha256=candidate.sha256,
                candidate_size=self.metrics.candidate_size,
                baseline_outcome="fail",
                approval_outcome="rejected",
                generation_latency_seconds=self.metrics.generation_latency_seconds,
                baseline_latency_seconds=duration,
                generation_calls=self.metrics.generation_calls,
            )
            return self.state
        identity = _approval_identity(
            task, candidate, workspace, generation, hashes, output, description
        )
        self._approval = EphemeralAcceptanceApproval(
            identity,
            candidate.sha256,
            hashlib.sha256(task.encode()).hexdigest(),
            str(workspace.resolve(strict=True)),
            generation,
        )
        self.state = EphemeralAcceptanceState.APPROVED
        self.events.append("ephemeral_test_approved")
        self.metrics = EphemeralAcceptanceMetrics(
            used=True,
            candidate_sha256=candidate.sha256,
            candidate_size=self.metrics.candidate_size,
            baseline_outcome="fail",
            approval_outcome="approved",
            generation_latency_seconds=self.metrics.generation_latency_seconds,
            baseline_latency_seconds=duration,
            generation_calls=self.metrics.generation_calls,
        )
        return self.state

    def before_mutation(self, task: str, workspace: Path, generation: int) -> bool:
        if not self.approved:
            return False
        approval = self._approval
        candidate = self._candidate
        assert approval is not None and candidate is not None
        try:
            hashes = _context_hashes(workspace, self.context_paths)
        except (OSError, ValueError):
            hashes = ()
        if (
            approval.task_sha256 != hashlib.sha256(task.encode()).hexdigest()
            or approval.candidate_sha256 != candidate.sha256
            or approval.workspace != str(workspace.resolve(strict=True))
            or approval.generation != generation
            or hashes != self._hashes
        ):
            self.state = EphemeralAcceptanceState.INVALIDATED
            self.events.append("ephemeral_test_invalidated")
            return False
        return True

    def postmutation(
        self, workspace: Path, generation: int
    ) -> EphemeralAcceptanceState:
        candidate = self._candidate
        approval = self._approval
        if (
            not self.approved
            or candidate is None
            or approval is None
            or approval.generation + 1 != generation
            or approval.workspace != str(workspace.resolve(strict=True))
            or approval.candidate_sha256 != candidate.sha256
            or not _candidate_valid(candidate, self.import_root)
        ):
            self.state = EphemeralAcceptanceState.INVALIDATED
            self.events.append("ephemeral_test_invalidated")
            return self.state
        outcome, _output, duration = _execute_python_candidate(candidate, workspace)
        self.state = {
            "pass": EphemeralAcceptanceState.POSTMUTATION_PASS,
            "fail": EphemeralAcceptanceState.POSTMUTATION_FAIL,
            "error": EphemeralAcceptanceState.POSTMUTATION_ERROR,
        }[outcome]
        self.events.append(
            "ephemeral_test_postmutation_passed"
            if outcome == "pass"
            else "ephemeral_test_postmutation_failed"
        )
        self.metrics = EphemeralAcceptanceMetrics(
            used=True,
            candidate_sha256=candidate.sha256,
            candidate_size=self.metrics.candidate_size,
            baseline_outcome="fail",
            approval_outcome="approved",
            postmutation_outcome=outcome,
            generation_latency_seconds=self.metrics.generation_latency_seconds,
            baseline_latency_seconds=self.metrics.baseline_latency_seconds,
            postmutation_latency_seconds=duration,
            generation_calls=self.metrics.generation_calls,
        )
        return self.state

    def discard_source(self) -> None:
        self._candidate = None
        self._preview = None

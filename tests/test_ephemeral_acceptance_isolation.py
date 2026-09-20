"""ephemeral-acceptance-isolation-v1 model-free policy contracts I01–I18."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.ephemeral_acceptance import (
    EphemeralAcceptanceCandidate,
    EphemeralAcceptanceGate,
    EphemeralAcceptanceMode,
    EphemeralAcceptanceState,
    EphemeralExecutionClass,
    _candidate_valid,
    _execute_python_candidate,
)
from forge.orchestration.coding_task import CodingTaskState, CodingTaskStatus
from forge.process_isolation import (
    EPHEMERAL_ENVIRONMENT_POLICY_ID,
    EPHEMERAL_TEMP_POLICY_ID,
    MacOSEphemeralTestSandbox,
    UnavailableEphemeralTestSandbox,
    ephemeral_test_environment,
)

TASK = "Fix the broken value."
SOURCE = "from pkg.value import value\nassert value() == 2\n"


class _FakeStrictSandbox:
    """Only a deterministic contract adapter; not an OS containment claim."""

    identity = "test-only-strict-v1"

    def __init__(self) -> None:
        self.denied = False

    def available(self) -> bool:
        return True

    def wrap(
        self, argv: tuple[str, ...], workspace: Path, temporary: Path
    ) -> tuple[str, ...]:
        if self.denied:
            return (
                argv[0],
                "-S",
                "-c",
                "raise PermissionError('Operation not permitted')",
            )
        return argv


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "value.py").write_text("def value():\n    return 1\n")
    (root / "test_value.py").write_text("assert True\n")
    return root


def _gate(workspace: Path, sandbox: object) -> EphemeralAcceptanceGate:
    return EphemeralAcceptanceGate(
        EphemeralAcceptanceMode.REQUIRED,
        context_paths=("pkg/value.py", "test_value.py"),
        import_root="pkg",
        sandbox=sandbox,  # type: ignore[arg-type]
    )


def _profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = tmp_path / "private-temp"
    temporary.mkdir()
    runtime = tmp_path / "runtime"
    app = runtime / "Resources/Python.app/Contents/MacOS/Python"
    app.parent.mkdir(parents=True)
    app.write_text("trusted launcher")
    monkeypatch.setattr("forge.process_isolation.sys.base_prefix", str(runtime))
    argv = MacOSEphemeralTestSandbox().wrap(
        (sys.executable, "-S", "-B", str(temporary / "candidate.py")),
        workspace,
        temporary,
    )
    return argv[2], workspace, temporary


def test_i01_strict_profile_and_no_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _, _ = _profile(tmp_path, monkeypatch)
    assert "(deny default)" in profile
    assert MacOSEphemeralTestSandbox.identity.endswith("ephemeral-test-v1")


def test_i02_unavailable_strict_adapter_never_runs_candidate(workspace: Path) -> None:
    gate = _gate(workspace, UnavailableEphemeralTestSandbox())
    seen = []
    state = gate.prepare_candidate(
        EphemeralAcceptanceCandidate("value", SOURCE),
        TASK,
        workspace,
        0,
        lambda preview: seen.append(preview) or True,
    )
    assert state is EphemeralAcceptanceState.ISOLATION_UNAVAILABLE
    assert gate.metrics.baseline_execution_class == "EPHEMERAL_ISOLATION_UNAVAILABLE"
    assert not seen


def test_i03_real_home_is_replaced(workspace: Path, tmp_path: Path) -> None:
    temporary = tmp_path / "temp"
    temporary.mkdir()
    environment = ephemeral_test_environment(workspace, temporary)
    assert environment["HOME"] == str(temporary)
    assert environment["HOME"] != str(Path.home())


def test_i04_ambient_secret_is_excluded(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.setenv("FORGE_TEST_SECRET", "fake-secret")
    environment = ephemeral_test_environment(workspace, temporary)
    assert "FORGE_TEST_SECRET" not in environment
    assert environment["PYTHONPATH"] == str(workspace)


def test_i05_private_temp_is_writable(workspace: Path) -> None:
    (workspace / "pkg" / "scratch.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['TMPDIR'], 'ok').write_text('ok')\nVALUE = True\n"
    )
    candidate = EphemeralAcceptanceCandidate(
        "scratch", "from pkg.scratch import VALUE\nassert VALUE\n"
    )
    result = _execute_python_candidate(
        candidate, workspace, _FakeStrictSandbox(), ("pkg/value.py",)
    )
    assert result.classification is EphemeralExecutionClass.PASS


def test_i06_project_write_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, workspace, temporary = _profile(tmp_path, monkeypatch)
    assert f'(allow file-write* (subpath "{temporary}"))' in profile
    assert f'(allow file-write* (subpath "{workspace}"))' not in profile


def test_i07_sibling_write_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, workspace, _ = _profile(tmp_path, monkeypatch)
    assert str(workspace.parent / "sibling") not in profile
    assert "(allow file-write*)" not in profile


def test_i08_real_home_write_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _, _ = _profile(tmp_path, monkeypatch)
    assert f'(subpath "{Path.home()}")' not in profile


def test_i09_network_not_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _, _ = _profile(tmp_path, monkeypatch)
    assert "allow network" not in profile


def test_i10_arbitrary_subprocess_not_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _, _ = _profile(tmp_path, monkeypatch)
    assert "(allow process*)" not in profile
    assert "/usr/bin/true" not in profile
    assert "allow process-exec" in profile


def test_i11_benign_project_import_passes(workspace: Path) -> None:
    result = _execute_python_candidate(
        EphemeralAcceptanceCandidate(
            "value", "from pkg.value import value\nassert value() == 1\n"
        ),
        workspace,
        _FakeStrictSandbox(),
        ("pkg/value.py",),
    )
    assert result.classification is EphemeralExecutionClass.PASS


@pytest.mark.parametrize(
    "case,body",
    (
        (
            "i12",
            "from pathlib import Path\n"
            "Path('/tmp/outside').write_text('bad')\nVALUE=True\n",
        ),
        (
            "i13",
            "from pathlib import Path\n"
            "Path.home().joinpath('secret').read_text()\nVALUE=True\n",
        ),
        ("i14", "import socket\nsocket.socket().bind(('127.0.0.1', 0))\nVALUE=True\n"),
    ),
)
def test_i12_i14_imported_code_not_screened_as_candidate(
    workspace: Path, case: str, body: str
) -> None:
    (workspace / "pkg" / f"{case}.py").write_text(body)
    candidate = EphemeralAcceptanceCandidate(
        case, f"from pkg.{case} import VALUE\nassert VALUE\n"
    )
    assert _candidate_valid(candidate, "pkg")


def test_i15_sandbox_failure_cannot_qualify_baseline(workspace: Path) -> None:
    sandbox = _FakeStrictSandbox()
    sandbox.denied = True
    gate = _gate(workspace, sandbox)
    seen = []
    state = gate.prepare_candidate(
        EphemeralAcceptanceCandidate("value", SOURCE),
        TASK,
        workspace,
        0,
        lambda preview: seen.append(preview) or True,
    )
    assert state is EphemeralAcceptanceState.SANDBOX_VIOLATION
    assert gate.metrics.baseline_execution_class == "EPHEMERAL_SANDBOX_VIOLATION"
    assert not seen


def test_i16_postapproval_sandbox_failure_blocks_and_does_not_repair(
    workspace: Path,
) -> None:
    sandbox = _FakeStrictSandbox()
    gate = _gate(workspace, sandbox)
    assert (
        gate.prepare_candidate(
            EphemeralAcceptanceCandidate("value", SOURCE),
            TASK,
            workspace,
            0,
            lambda _preview: True,
        )
        is EphemeralAcceptanceState.APPROVED
    )
    sandbox.denied = True
    (workspace / "pkg" / "value.py").write_text("def value():\n    return 2\n")
    assert gate.postmutation(workspace, 1) is EphemeralAcceptanceState.SANDBOX_VIOLATION
    state = CodingTaskState(0, repair_enabled=True)
    state.mutation_count = 1
    state.ephemeral_acceptance_failed(gate.metrics)
    result = state.finish("blocked")
    assert result.status is CodingTaskStatus.EPHEMERAL_ACCEPTANCE_FAILED
    assert not result.repair_eligible and result.test.status == "not_run"


def test_i17_policy_identity_change_invalidates_approval(workspace: Path) -> None:
    sandbox = _FakeStrictSandbox()
    gate = _gate(workspace, sandbox)
    gate.prepare_candidate(
        EphemeralAcceptanceCandidate("value", SOURCE),
        TASK,
        workspace,
        0,
        lambda _preview: True,
    )
    sandbox.identity = "test-only-strict-v2"
    assert not gate.before_mutation(TASK, workspace, 0)
    assert gate.state is EphemeralAcceptanceState.INVALIDATED


def test_i18_source_integrity_checked_on_success(workspace: Path) -> None:
    candidate = EphemeralAcceptanceCandidate(
        "value", "from pkg.value import value\nassert value() == 1\n"
    )
    before = (workspace / "pkg" / "value.py").read_bytes()
    result = _execute_python_candidate(
        candidate, workspace, _FakeStrictSandbox(), ("pkg/value.py",)
    )
    assert result.classification is EphemeralExecutionClass.PASS
    assert result.source_integrity_checked
    assert (workspace / "pkg" / "value.py").read_bytes() == before


def test_policy_ids_are_versioned() -> None:
    assert EPHEMERAL_TEMP_POLICY_ID.endswith("-v1")
    assert EPHEMERAL_ENVIRONMENT_POLICY_ID.endswith("-v1")


def test_fake_adapter_can_detect_source_mutation(workspace: Path) -> None:
    (workspace / "pkg" / "mutate.py").write_text(
        "from pathlib import Path\n"
        "Path('pkg/value.py').write_text('changed')\nVALUE=True\n"
    )
    candidate = EphemeralAcceptanceCandidate(
        "mutate", "from pkg.mutate import VALUE\nassert VALUE\n"
    )
    outcome = _execute_python_candidate(
        candidate, workspace, _FakeStrictSandbox(), ("pkg/value.py",)
    )
    assert outcome.classification is EphemeralExecutionClass.WORKSPACE_MODIFIED

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from forge.tools import (
    AllowAllPolicy,
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    DenyAllPolicy,
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    PermissionPolicy,
    RuleBasedPolicy,
    Tool,
    ToolError,
    ToolErrorKind,
    ToolExecutionMetadata,
    ToolExecutor,
    ToolInvocation,
    ToolMetadata,
    ToolRegistrationError,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolValidationError,
)


class CountingTool(Tool):
    def __init__(
        self,
        name: str = "test.echo",
        schema: ArgumentSchema | None = None,
        *,
        behavior: str = "echo",
    ) -> None:
        self._metadata = ToolMetadata(
            name,
            "A deterministic side-effect-free test tool.",
            schema
            or ArgumentSchema(
                (ArgumentSpec("text", ArgumentType.STRING, "Text to return."),)
            ),
        )
        self.behavior = behavior
        self.calls = 0
        self.received: list[Mapping[str, str | int | bool]] = []

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> object:
        self.calls += 1
        self.received.append(arguments)
        if self.behavior == "failure":
            raise ToolError("expected test failure")
        if self.behavior == "unexpected":
            raise RuntimeError("sensitive native detail")
        if self.behavior == "bad-output":
            return {"unsupported"}
        if self.behavior == "add":
            return {"sum": int(arguments["a"]) + int(arguments["b"])}
        return arguments["text"]


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.25
        return current


def context(tmp_path: Path) -> ExecutionContext:
    return ExecutionContext(tmp_path.resolve())


def invocation(
    invocation_id: str = "invocation-1",
    tool_name: str = "test.echo",
    arguments: Mapping[str, object] | None = None,
) -> ToolInvocation:
    return ToolInvocation(
        invocation_id,
        tool_name,
        {"text": "hello"} if arguments is None else arguments,
    )


def executor(
    tool: Tool, policy: PermissionPolicy, *, clock: StepClock | None = None
) -> ToolExecutor:
    return ToolExecutor(
        ToolRegistry((tool,)),
        policy,
        clock=clock or StepClock(),
    )


@pytest.mark.parametrize(
    "name",
    ["", "Echo", "test echo", ".echo", "test..echo", "test/echo", "test-echo"],
)
def test_tool_metadata_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ToolValidationError, match="tool name"):
        ToolMetadata(name, "Description", ArgumentSchema())


def test_tool_metadata_requires_description() -> None:
    with pytest.raises(ToolValidationError, match="description"):
        ToolMetadata("test.echo", " ", ArgumentSchema())


def test_argument_schema_supports_required_and_optional_values() -> None:
    schema = ArgumentSchema(
        (
            ArgumentSpec("text", ArgumentType.STRING, "Text"),
            ArgumentSpec("uppercase", ArgumentType.BOOLEAN, "Uppercase", False),
        )
    )
    assert dict(schema.validate({"text": "hello"})) == {"text": "hello"}
    assert dict(schema.validate({"text": "hello", "uppercase": True})) == {
        "text": "hello",
        "uppercase": True,
    }


def test_argument_schema_rejects_duplicate_definitions() -> None:
    with pytest.raises(ToolValidationError, match="duplicate"):
        ArgumentSchema(
            (
                ArgumentSpec("value", ArgumentType.STRING, "First"),
                ArgumentSpec("value", ArgumentType.INTEGER, "Second"),
            )
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "missing required"),
        ({"text": 4}, "must be string"),
        ({"text": "hello", "extra": True}, "unknown arguments"),
    ],
)
def test_argument_validation_errors_are_clear(
    arguments: Mapping[str, object], message: str
) -> None:
    schema = ArgumentSchema((ArgumentSpec("text", ArgumentType.STRING, "Text"),))
    with pytest.raises(ToolValidationError, match=message):
        schema.validate(arguments)


def test_integer_argument_does_not_accept_boolean() -> None:
    schema = ArgumentSchema((ArgumentSpec("value", ArgumentType.INTEGER, "Value"),))
    with pytest.raises(ToolValidationError, match="integer"):
        schema.validate({"value": True})


def test_invocation_copies_and_freezes_caller_arguments() -> None:
    supplied: dict[str, object] = {"text": "original"}
    request = invocation(arguments=supplied)
    supplied["text"] = "changed"
    assert dict(request.arguments) == {"text": "original"}
    with pytest.raises(TypeError):
        request.arguments["text"] = "changed"  # type: ignore[index]


def test_invocation_is_immutable() -> None:
    request = invocation()
    with pytest.raises(FrozenInstanceError):
        request.tool_name = "test.other"  # type: ignore[misc]


def test_registry_registration_lookup_and_deterministic_metadata() -> None:
    second = CountingTool("test.zeta")
    first = CountingTool("test.alpha")
    registry = ToolRegistry()
    registry.register(second)
    registry.register(first)
    assert registry.get("test.alpha") is first
    assert [item.name for item in registry.metadata] == ["test.alpha", "test.zeta"]
    assert first.calls == second.calls == 0


def test_registry_rejects_duplicates_and_unknown_tools() -> None:
    registry = ToolRegistry((CountingTool(),))
    with pytest.raises(ToolRegistrationError, match="duplicate"):
        registry.register(CountingTool())
    with pytest.raises(ToolRegistrationError, match="unknown"):
        registry.get("test.absent")


def test_registries_are_independent() -> None:
    first = ToolRegistry((CountingTool(),))
    second = ToolRegistry()
    assert len(first.metadata) == 1
    assert second.metadata == ()


def test_permission_policies_are_deterministic_and_deny_by_default(
    tmp_path: Path,
) -> None:
    tool = CountingTool()
    request = invocation()
    execution_context = context(tmp_path)
    assert (
        AllowAllPolicy().evaluate(tool, request, execution_context)
        is PermissionDecision.ALLOW
    )
    assert (
        DenyAllPolicy().evaluate(tool, request, execution_context)
        is PermissionDecision.DENY
    )
    policy = RuleBasedPolicy({"test.echo": PermissionDecision.ASK})
    assert policy.evaluate(tool, request, execution_context) is PermissionDecision.ASK
    assert (
        RuleBasedPolicy({}).evaluate(tool, request, execution_context)
        is PermissionDecision.DENY
    )


def test_allow_executes_exactly_once(tmp_path: Path) -> None:
    tool = CountingTool()
    result = executor(tool, AllowAllPolicy()).execute(invocation(), context(tmp_path))
    assert result.status is ToolResultStatus.SUCCESS
    assert result.output == "hello"
    assert tool.calls == 1
    assert result.invocation_id == "invocation-1"
    assert result.tool_name == "test.echo"


def test_deny_executes_zero_times(tmp_path: Path) -> None:
    tool = CountingTool()
    result = executor(tool, DenyAllPolicy()).execute(invocation(), context(tmp_path))
    assert result.status is ToolResultStatus.DENIED
    assert result.metadata.permission_decision is PermissionDecision.DENY
    assert tool.calls == 0


def test_ask_without_approval_executes_zero_times(tmp_path: Path) -> None:
    tool = CountingTool()
    policy = RuleBasedPolicy({"test.echo": PermissionDecision.ASK})
    result = executor(tool, policy).execute(invocation(), context(tmp_path))
    assert result.status is ToolResultStatus.APPROVAL_REQUIRED
    assert tool.calls == 0


def test_ask_with_exact_approval_executes_once(tmp_path: Path) -> None:
    tool = CountingTool()
    request = invocation()
    approval = InvocationApproval.for_invocation(request)
    policy = RuleBasedPolicy({"test.echo": PermissionDecision.ASK})
    result = executor(tool, policy).execute(
        request, context(tmp_path), approval=approval
    )
    assert result.status is ToolResultStatus.SUCCESS
    assert result.metadata.permission_decision is PermissionDecision.ASK
    assert tool.calls == 1


def test_approval_does_not_cross_invocations_or_changed_arguments(
    tmp_path: Path,
) -> None:
    tool = CountingTool()
    policy = RuleBasedPolicy({"test.echo": PermissionDecision.ASK})
    runner = executor(tool, policy)
    approved = invocation("A", arguments={"text": "reviewed"})
    approval = InvocationApproval.for_invocation(approved)

    other_id = invocation("B", arguments={"text": "reviewed"})
    changed = invocation("A", arguments={"text": "changed"})
    assert (
        runner.execute(other_id, context(tmp_path), approval=approval).status
        is ToolResultStatus.APPROVAL_REQUIRED
    )
    assert (
        runner.execute(changed, context(tmp_path), approval=approval).status
        is ToolResultStatus.APPROVAL_REQUIRED
    )
    assert tool.calls == 0


def test_invalid_arguments_are_rejected_before_permission_and_execution(
    tmp_path: Path,
) -> None:
    tool = CountingTool()
    result = executor(tool, AllowAllPolicy()).execute(
        invocation(arguments={"text": 7}), context(tmp_path)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.error_kind is ToolErrorKind.VALIDATION
    assert tool.calls == 0


def test_unknown_tool_fails_without_executing_registered_tool(tmp_path: Path) -> None:
    tool = CountingTool()
    result = executor(tool, AllowAllPolicy()).execute(
        invocation(tool_name="test.absent"), context(tmp_path)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.error_kind is ToolErrorKind.UNKNOWN_TOOL
    assert tool.calls == 0


def test_expected_and_unexpected_tool_failures_become_structured_results(
    tmp_path: Path,
) -> None:
    expected = CountingTool(behavior="failure")
    expected_result = executor(expected, AllowAllPolicy()).execute(
        invocation(), context(tmp_path)
    )
    assert expected_result.status is ToolResultStatus.FAILURE
    assert expected_result.error_kind is ToolErrorKind.TOOL_FAILURE
    assert expected_result.error_message == "expected test failure"

    unexpected = CountingTool(behavior="unexpected")
    unexpected_result = executor(unexpected, AllowAllPolicy()).execute(
        invocation(), context(tmp_path)
    )
    assert unexpected_result.status is ToolResultStatus.FAILURE
    assert unexpected_result.error_kind is ToolErrorKind.UNEXPECTED
    assert "sensitive native detail" not in unexpected_result.error_message


def test_unsupported_tool_output_becomes_unexpected_failure(tmp_path: Path) -> None:
    tool = CountingTool(behavior="bad-output")
    result = executor(tool, AllowAllPolicy()).execute(invocation(), context(tmp_path))
    assert result.status is ToolResultStatus.FAILURE
    assert result.error_kind is ToolErrorKind.UNEXPECTED


def test_add_tool_returns_immutable_structured_output(tmp_path: Path) -> None:
    tool = CountingTool(
        "test.add",
        ArgumentSchema(
            (
                ArgumentSpec("a", ArgumentType.INTEGER, "First integer"),
                ArgumentSpec("b", ArgumentType.INTEGER, "Second integer"),
            )
        ),
        behavior="add",
    )
    result = executor(tool, AllowAllPolicy()).execute(
        invocation("add-1", "test.add", {"a": 2, "b": 3}), context(tmp_path)
    )
    assert dict(result.output) == {"sum": 5}  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        result.output["sum"] = 6  # type: ignore[index]


def test_tool_result_recursively_freezes_nested_structured_output() -> None:
    source = {"items": [{"name": "before"}]}
    result = ToolResult(
        "nested-1",
        "test.nested",
        ToolResultStatus.SUCCESS,
        ToolExecutionMetadata(PermissionDecision.ALLOW, 0.0),
        source,
    )
    source["items"][0]["name"] = "after"
    assert result.output["items"][0]["name"] == "before"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.output["items"][0]["name"] = "changed"  # type: ignore[index]


def test_validated_arguments_do_not_mutate_invocation_arguments(tmp_path: Path) -> None:
    tool = CountingTool()
    request = invocation(arguments={"text": "unchanged"})
    executor(tool, AllowAllPolicy()).execute(request, context(tmp_path))
    assert dict(request.arguments) == {"text": "unchanged"}
    with pytest.raises(TypeError):
        tool.received[0]["text"] = "mutation"  # type: ignore[index]


def test_execution_metadata_uses_injected_monotonic_clock(tmp_path: Path) -> None:
    result = executor(CountingTool(), AllowAllPolicy(), clock=StepClock()).execute(
        invocation(), context(tmp_path)
    )
    assert result.metadata.duration_seconds == 0.25


def test_execution_context_is_absolute_normalized_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    execution_context = ExecutionContext(workspace.resolve())
    monkeypatch.chdir(tmp_path)
    assert execution_context.workspace == workspace.resolve()
    with pytest.raises(FrozenInstanceError):
        execution_context.workspace = tmp_path  # type: ignore[misc]


def test_execution_context_rejects_relative_missing_and_file_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ExecutionContext(Path("relative"))
    with pytest.raises(ValueError, match="does not exist"):
        ExecutionContext((tmp_path / "missing").resolve())
    file_path = tmp_path / "file"
    file_path.touch()
    with pytest.raises(ValueError, match="not a directory"):
        ExecutionContext(file_path.resolve())


def test_logs_identifiers_but_not_sensitive_arguments(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    secret = "do-not-log-this-value"
    executor(CountingTool(), AllowAllPolicy()).execute(
        invocation("safe-id", arguments={"text": secret}), context(tmp_path)
    )
    assert "safe-id" in caplog.text
    assert "test.echo" in caplog.text
    assert secret not in caplog.text

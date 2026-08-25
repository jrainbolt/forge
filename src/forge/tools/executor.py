"""Central permission-enforcing tool executor."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from forge.tools.permissions import PermissionPolicy
from forge.tools.registry import ToolRegistrationError, ToolRegistry
from forge.tools.tool import ToolError
from forge.tools.types import (
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    ToolErrorKind,
    ToolExecutionMetadata,
    ToolInvocation,
    ToolResult,
    ToolResultStatus,
    ToolValidationError,
)

LOGGER = logging.getLogger(__name__)
Clock = Callable[[], float]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._clock = clock

    def permission(
        self, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        """Return the effective decision without executing or requesting approval."""
        try:
            tool = self._registry.get(invocation.tool_name)
        except ToolRegistrationError:
            return PermissionDecision.DENY
        return self._policy.evaluate(tool, invocation, context)

    def execute(
        self,
        invocation: ToolInvocation,
        context: ExecutionContext,
        *,
        approval: InvocationApproval | None = None,
    ) -> ToolResult:
        started = self._clock()
        LOGGER.info(
            "Tool requested name=%s invocation_id=%s",
            invocation.tool_name,
            invocation.invocation_id,
        )
        try:
            tool = self._registry.get(invocation.tool_name)
        except ToolRegistrationError:
            LOGGER.warning(
                "Unknown tool name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._failure(
                invocation,
                started,
                PermissionDecision.DENY,
                ToolErrorKind.UNKNOWN_TOOL,
                "requested tool is not registered",
            )

        try:
            arguments = tool.metadata.argument_schema.validate(invocation.arguments)
        except ToolValidationError as error:
            LOGGER.warning(
                "Invalid tool arguments name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._failure(
                invocation,
                started,
                PermissionDecision.DENY,
                ToolErrorKind.VALIDATION,
                str(error),
            )

        decision = self._policy.evaluate(tool, invocation, context)
        LOGGER.info(
            "Tool permission name=%s invocation_id=%s decision=%s",
            invocation.tool_name,
            invocation.invocation_id,
            decision.value,
        )
        if decision is PermissionDecision.DENY:
            LOGGER.info(
                "Tool denied name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._result(invocation, started, decision, ToolResultStatus.DENIED)
        if decision is PermissionDecision.ASK and (
            approval is None or not approval.matches(invocation)
        ):
            LOGGER.info(
                "Tool approval required name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._result(
                invocation, started, decision, ToolResultStatus.APPROVAL_REQUIRED
            )

        LOGGER.info(
            "Tool started name=%s invocation_id=%s",
            invocation.tool_name,
            invocation.invocation_id,
        )
        try:
            output = tool.execute(arguments, context)
            result = self._result(
                invocation,
                started,
                decision,
                ToolResultStatus.SUCCESS,
                output=output,
            )
        except ToolError as error:
            LOGGER.warning(
                "Tool failed name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._failure(
                invocation,
                started,
                decision,
                ToolErrorKind.TOOL_FAILURE,
                str(error),
                output=error.output,
            )
        except Exception:
            LOGGER.error(
                "Unexpected tool failure name=%s invocation_id=%s",
                invocation.tool_name,
                invocation.invocation_id,
            )
            return self._failure(
                invocation,
                started,
                decision,
                ToolErrorKind.UNEXPECTED,
                "tool raised an unexpected internal error",
            )
        LOGGER.info(
            "Tool completed name=%s invocation_id=%s",
            invocation.tool_name,
            invocation.invocation_id,
        )
        return result

    def _failure(
        self,
        invocation: ToolInvocation,
        started: float,
        decision: PermissionDecision,
        kind: ToolErrorKind,
        message: str,
        *,
        output: object = None,
    ) -> ToolResult:
        return self._result(
            invocation,
            started,
            decision,
            ToolResultStatus.FAILURE,
            error_kind=kind,
            error_message=message,
            output=output,
        )

    def _result(
        self,
        invocation: ToolInvocation,
        started: float,
        decision: PermissionDecision,
        status: ToolResultStatus,
        *,
        output: object = None,
        error_kind: ToolErrorKind | None = None,
        error_message: str | None = None,
    ) -> ToolResult:
        return ToolResult(
            invocation_id=invocation.invocation_id,
            tool_name=invocation.tool_name,
            status=status,
            output=output,  # type: ignore[arg-type]
            error_kind=error_kind,
            error_message=error_message,
            metadata=ToolExecutionMetadata(
                permission_decision=decision,
                duration_seconds=self._clock() - started,
            ),
        )

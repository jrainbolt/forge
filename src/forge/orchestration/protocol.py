"""Strict model-facing protocol and deterministic safe renderers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from forge.tools import ToolRegistry, ToolResult
from forge.tools.types import StructuredValue, validate_tool_name

TOOL_CALL_START = "<forge_tool_call>\n"
TOOL_CALL_END = "\n</forge_tool_call>"
TOOL_RESULT_START = "<forge_tool_result>\n"
TOOL_RESULT_END = "\n</forge_tool_result>"
MAX_TOOL_CALL_PAYLOAD_BYTES = 16 * 1024
INVOCATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ProtocolError(ValueError):
    """A model response resembles control data but violates the protocol."""


class ToolCallOutcome(Enum):
    FINAL = "final"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolCall:
    invocation_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(
            self.invocation_id, str
        ) or not INVOCATION_ID_PATTERN.fullmatch(self.invocation_id):
            raise ProtocolError("tool-call id has an invalid format")
        try:
            validate_tool_name(self.tool_name)
        except (TypeError, ValueError) as error:
            raise ProtocolError("tool-call tool has an invalid name") from error
        if not isinstance(self.arguments, Mapping):
            raise ProtocolError("tool-call arguments must be an object")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ParsedModelOutput:
    outcome: ToolCallOutcome
    text: str | None = None
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        if self.outcome is ToolCallOutcome.FINAL:
            if self.text is None or self.tool_call is not None:
                raise ValueError("final output must contain only text")
        elif self.outcome is ToolCallOutcome.TOOL_CALL:
            if self.tool_call is None or self.text is not None:
                raise ValueError("tool output must contain only one tool call")
        else:
            raise TypeError("outcome must be a ToolCallOutcome")


def parse_model_output(text: str) -> ParsedModelOutput:
    """Classify one complete model response without executing any capability."""
    if not isinstance(text, str):
        raise TypeError("model output must be text")
    if not text.strip():
        raise ProtocolError("model output is empty")
    contains_call_frame = "<forge_tool_call>" in text or "</forge_tool_call>" in text
    if not contains_call_frame:
        return ParsedModelOutput(ToolCallOutcome.FINAL, text=text)
    if not (text.startswith(TOOL_CALL_START) and text.endswith(TOOL_CALL_END)):
        raise ProtocolError("tool call must be the entire model response")
    if text.count("<forge_tool_call>") != 1 or text.count("</forge_tool_call>") != 1:
        raise ProtocolError("model response must contain exactly one tool call")
    payload_text = text[len(TOOL_CALL_START) : -len(TOOL_CALL_END)]
    if len(payload_text.encode("utf-8")) > MAX_TOOL_CALL_PAYLOAD_BYTES:
        raise ProtocolError("tool-call payload exceeds the protocol limit")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ProtocolError("tool-call payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ProtocolError("tool-call payload must be an object")
    expected = {"id", "tool", "arguments"}
    if set(payload) != expected:
        raise ProtocolError(
            "tool-call payload must contain exactly id, tool, arguments"
        )
    if not isinstance(payload["id"], str):
        raise ProtocolError("tool-call id must be text")
    if not isinstance(payload["tool"], str):
        raise ProtocolError("tool-call tool must be text")
    call = ToolCall(payload["id"], payload["tool"], payload["arguments"])
    return ParsedModelOutput(ToolCallOutcome.TOOL_CALL, tool_call=call)


def render_tool_definitions(registry: ToolRegistry) -> str:
    """Render safe registry metadata as deterministic model-facing JSON."""
    tools = []
    for metadata in registry.metadata:
        arguments = []
        for argument in metadata.argument_schema.arguments:
            arguments.append(
                {
                    "description": argument.description,
                    "name": argument.name,
                    "required": argument.required,
                    "type": argument.value_type.value,
                }
            )
        tools.append(
            {
                "arguments": arguments,
                "description": metadata.description,
                "name": metadata.name,
                "risk": metadata.risk.value,
            }
        )
    return json.dumps({"tools": tools}, ensure_ascii=False, sort_keys=True)


def render_tool_result(result: ToolResult) -> str:
    """Serialize one trusted executor result without Python representations."""
    payload: dict[str, object] = {
        "id": result.invocation_id,
        "status": result.status.value,
        "tool": result.tool_name,
    }
    if result.status.value == "success":
        payload["output"] = _json_value(result.output)
    else:
        payload["error"] = {
            "kind": result.error_kind.value if result.error_kind is not None else None,
            "message": result.error_message,
        }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"{TOOL_RESULT_START}{encoded}{TOOL_RESULT_END}"


def _json_value(value: StructuredValue) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value

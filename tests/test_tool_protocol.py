from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from forge.models import ResponseFormat
from forge.orchestration.protocol import (
    MAX_MODEL_RESPONSE_BYTES,
    REPOSITORY_RESPONSE_OUTPUT,
    ProtocolError,
    ToolCallOutcome,
    build_repository_output,
    parse_model_output,
    render_tool_definitions,
    render_tool_result,
)
from forge.tools import (
    PermissionDecision,
    ToolErrorKind,
    ToolExecutionMetadata,
    ToolResult,
    ToolResultStatus,
    create_readonly_repository_registry,
)


def encoded(payload: object) -> str:
    return json.dumps(payload)


def test_parser_accepts_valid_tool_call_and_final_envelopes() -> None:
    tool = parse_model_output(
        encoded(
            {
                "type": "tool_call",
                "id": "call-1",
                "tool": "repository.read_file",
                "arguments": {"path": "src/forge/session.py"},
            }
        )
    )
    assert tool.outcome is ToolCallOutcome.TOOL_CALL
    assert tool.tool_call is not None
    assert tool.tool_call.invocation_id == "call-1"
    assert dict(tool.tool_call.arguments) == {"path": "src/forge/session.py"}

    final = parse_model_output(encoded({"type": "final", "answer": "Grounded."}))
    assert final.outcome is ToolCallOutcome.FINAL
    assert final.text == "Grounded."


@pytest.mark.parametrize(
    "text, message",
    (
        ("", "empty"),
        ("{bad}", "valid JSON"),
        (encoded({"type": "unknown"}), "type"),
        (
            encoded({"type": "tool_call", "tool": "git.status", "arguments": {}}),
            "exactly",
        ),
        (encoded({"type": "tool_call", "id": "x", "arguments": {}}), "exactly"),
        (encoded({"type": "tool_call", "id": "x", "tool": "git.status"}), "exactly"),
        (
            encoded(
                {
                    "type": "tool_call",
                    "id": "x",
                    "tool": "git.status",
                    "arguments": [],
                }
            ),
            "arguments",
        ),
        (
            encoded(
                {
                    "type": "tool_call",
                    "id": "x",
                    "tool": "git.status",
                    "arguments": {},
                    "answer": "wrong",
                }
            ),
            "exactly",
        ),
        (
            encoded({"type": "final", "answer": "answer", "tool": "git.status"}),
            "exactly",
        ),
        (encoded({"type": "final"}), "exactly"),
        (encoded({"type": "final", "answer": 3}), "answer"),
        (
            '{"type":"final","answer":"one"}{"type":"final","answer":"two"}',
            "valid JSON",
        ),
        ('before {"type":"final","answer":"x"}', "valid JSON"),
        ('{"type":"final","answer":"x"} after', "valid JSON"),
        ('```json\n{"type":"final","answer":"x"}\n```', "valid JSON"),
    ),
)
def test_parser_rejects_malformed_or_ambiguous_json(text: str, message: str) -> None:
    with pytest.raises(ProtocolError, match=message):
        parse_model_output(text)


def test_parser_rejects_oversized_response() -> None:
    text = encoded({"type": "final", "answer": "x" * MAX_MODEL_RESPONSE_BYTES})
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_model_output(text)


def test_repository_output_spec_is_generic_json_with_immutable_schema() -> None:
    assert REPOSITORY_RESPONSE_OUTPUT.format is ResponseFormat.JSON
    assert REPOSITORY_RESPONSE_OUTPUT.schema is not None
    with pytest.raises(TypeError):
        REPOSITORY_RESPONSE_OUTPUT.schema["other"] = True  # type: ignore[index]


def test_dynamic_output_schema_enforces_discovered_path_and_query_provenance() -> None:
    registry = create_readonly_repository_registry()
    initial = build_repository_output(
        registry,
        allow_final=False,
        candidate_files=set(),
        candidate_directories={"."},
        candidate_queries={"ChatSession", "workspace"},
    )
    initial_branches = initial.schema["oneOf"]  # type: ignore[index]
    initial_tools = {
        branch["properties"]["tool"]["const"]: branch  # type: ignore[index]
        for branch in initial_branches
    }
    assert "repository.read_file" not in initial_tools
    search_arguments = initial_tools["repository.search_files"]["properties"][
        "arguments"
    ]["properties"]
    assert tuple(search_arguments["path"]["enum"]) == (".",)
    assert tuple(search_arguments["query"]["enum"]) == ("ChatSession", "workspace")

    discovered = build_repository_output(
        registry,
        allow_final=True,
        candidate_files={"src/forge/session.py", "src/forge/tools/paths.py"},
        candidate_directories={".", "src/forge"},
        candidate_queries={"workspace"},
    )
    branches = discovered.schema["oneOf"]  # type: ignore[index]
    tools = {
        branch["properties"]["tool"]["const"]: branch  # type: ignore[index]
        for branch in branches
        if branch["properties"]["type"]["const"] == "tool_call"  # type: ignore[index]
    }
    read_paths = tools["repository.read_file"]["properties"]["arguments"][  # type: ignore[index]
        "properties"
    ]["path"]["enum"]
    list_paths = tools["repository.list_directory"]["properties"]["arguments"][  # type: ignore[index]
        "properties"
    ]["path"]["enum"]
    assert tuple(read_paths) == (
        "src/forge/session.py",
        "src/forge/tools/paths.py",
    )
    assert tuple(list_paths) == (".", "src/forge")
    assert any(
        branch["properties"]["type"]["const"] == "final"  # type: ignore[index]
        for branch in branches
    )


def test_dynamic_output_schema_intersects_routed_tool_names() -> None:
    specification = build_repository_output(
        create_readonly_repository_registry(),
        allow_final=False,
        candidate_files={"src/target.py"},
        candidate_directories={"."},
        candidate_queries={"target"},
        allowed_tool_names={"repository.read_range", "git.status"},
    )
    tools = {
        branch["properties"]["tool"]["const"]  # type: ignore[index]
        for branch in specification.schema["oneOf"]  # type: ignore[index]
    }
    assert tools == {"repository.read_range", "git.status"}


def test_tool_definitions_are_deterministic_schema_derived_and_categorized() -> None:
    rendered = render_tool_definitions(create_readonly_repository_registry())
    payload = json.loads(rendered)
    assert [tool["name"] for tool in payload["tools"]] == [
        "git.diff",
        "git.status",
        "repository.file_outline",
        "repository.find_references",
        "repository.find_symbol",
        "repository.list_directory",
        "repository.read_file",
        "repository.read_range",
        "repository.search_files",
    ]
    evidence = {tool["name"]: tool["evidence"] for tool in payload["tools"]}
    assert evidence == {
        "git.diff": "git_working_state",
        "git.status": "git_working_state",
        "repository.file_outline": "discovery",
        "repository.find_references": "discovery",
        "repository.find_symbol": "discovery",
        "repository.list_directory": "discovery",
        "repository.read_file": "source_content",
        "repository.read_range": "source_content",
        "repository.search_files": "discovery",
    }
    assert render_tool_definitions(create_readonly_repository_registry()) == rendered


def result(
    status: ToolResultStatus,
    *,
    output: object = None,
    kind: ToolErrorKind | None = None,
    message: str | None = None,
) -> ToolResult:
    return ToolResult(
        "call-1",
        "repository.read_file",
        status,
        ToolExecutionMetadata(PermissionDecision.ALLOW, 0.1),
        output,  # type: ignore[arg-type]
        kind,
        message,
    )


@pytest.mark.parametrize(
    "tool_result",
    (
        result(
            ToolResultStatus.FAILURE, kind=ToolErrorKind.TOOL_FAILURE, message="bad"
        ),
        result(ToolResultStatus.DENIED),
        result(ToolResultStatus.APPROVAL_REQUIRED),
    ),
)
def test_tool_result_renderer_handles_non_success_states(
    tool_result: ToolResult,
) -> None:
    rendered = render_tool_result(tool_result)
    payload = json.loads(rendered)
    assert payload["type"] == "tool_result"
    assert payload["status"] == tool_result.status.value
    assert set(payload) == {
        "type",
        "error",
        "evidence",
        "guidance",
        "id",
        "status",
        "tool",
    }
    assert "Traceback" not in rendered


def test_tool_result_renderer_serializes_nested_output_deterministically() -> None:
    tool_result = result(
        ToolResultStatus.SUCCESS,
        output={"entries": [{"path": "src/a.py", "line": 2}], "clean": False},
    )
    rendered = render_tool_result(tool_result)
    payload = json.loads(rendered)
    output = payload["output"]
    assert isinstance(output, Mapping)
    assert output == {
        "clean": False,
        "entries": [{"line": 2, "path": "src/a.py"}],
    }
    assert render_tool_result(tool_result) == rendered

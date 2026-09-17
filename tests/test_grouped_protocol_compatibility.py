from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forge.evaluation import (
    GROUPED_PROTOCOL_COMPATIBILITY_V1,
    PRODUCTION_GROUPED_OUTPUT_BUILDER,
    SYNTHETIC_GROUPED_FIXTURES,
    EvaluationOutcome,
    GroupedFailure,
    GroupedResponseClass,
    ModelExchange,
    aggregate_grouped_results,
    build_grouped_protocol_run,
    classify_grouped_response,
    full_forge_grouped_result,
    grouped_protocol_to_dict,
    production_grouped_output,
    run_grouped_fixture,
)
from forge.models import (
    FinishReason,
    Message,
    MessageRole,
    MockModel,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    MutationRepresentationPolicy,
)
from forge.orchestration import RepositoryChatSession, RepositoryOrchestrationError
from forge.orchestration.protocol import build_mutation_ready_output
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def _p02_responses(*, production: str | None = None) -> tuple[str, ...]:
    implementation = {
        "start_line": 2,
        "end_line": 2,
        "new_text": "    return min(a + b, limit)\n",
    }
    test = {
        "start_line": 4,
        "end_line": 4,
        "new_text": (
            "assert capped_add(8, 7, 10) == 10\nassert capped_add(4, 6, 10) == 10\n"
        ),
    }
    edits = [
        {"path": "calculator.py", **implementation},
        {"path": "test_calculator.py", **test},
    ]
    return (
        "The implementation must cap at the exact limit boundary. The test should "
        "assert that an exact-limit sum equals 10.",
        json.dumps({"implementation": implementation, "test": test}),
        json.dumps({"edits": edits}),
        production
        or json.dumps({"type": "multi_file_line_range_edit", "edits": edits}),
    )


def test_g1_to_g4_success_uses_two_independent_files_and_real_oracle() -> None:
    model = MockModel(_p02_responses(), context_capacity=8192)
    results = run_grouped_fixture("mock", model, SYNTHETIC_GROUPED_FIXTURES[1])

    assert [item.layer for item in results] == ["G1", "G2", "G3", "G4"]
    assert results[0].concept_implementation
    assert results[0].concept_test
    assert results[0].concept_overall
    assert results[1].implementation_target_valid
    assert results[1].test_target_valid
    assert results[1].implementation_semantic
    assert results[1].test_semantic
    assert all(item.oracle == "PASS" for item in results[1:])
    assert all(item.failure == GroupedFailure.PASS.value for item in results)
    assert "chain-of-thought" not in model.requests[0].messages[0].content
    assert "hidden reasoning steps" in model.requests[0].messages[0].content


def test_g2_records_per_file_target_failure_without_collapsing() -> None:
    responses = list(_p02_responses())
    payload = json.loads(responses[1])
    payload["implementation"]["start_line"] = 99
    payload["implementation"]["end_line"] = 99
    responses[1] = json.dumps(payload)
    result = run_grouped_fixture(
        "mock",
        MockModel(responses, context_capacity=8192),
        SYNTHETIC_GROUPED_FIXTURES[1],
    )[1]

    assert result.implementation_target_valid is False
    assert result.implementation_delta is False
    assert result.test_target_valid is True
    assert result.test_delta is True
    assert result.oracle == "NOT_RUN"


def test_g3_rejects_duplicate_and_unexpected_paths() -> None:
    responses = list(_p02_responses())
    payload = json.loads(responses[2])
    payload["edits"][1]["path"] = "calculator.py"
    responses[2] = json.dumps(payload)
    result = run_grouped_fixture(
        "mock",
        MockModel(responses, context_capacity=8192),
        SYNTHETIC_GROUPED_FIXTURES[1],
    )[2]

    assert result.schema_valid
    assert result.duplicate_path is True
    assert result.path_set_valid is False
    assert result.oracle == "NOT_RUN"


def test_g4_uses_exact_production_builder_and_parser_schema() -> None:
    paths = ("implementation.py", "test_implementation.py")
    assert PRODUCTION_GROUPED_OUTPUT_BUILDER is build_mutation_ready_output
    assert production_grouped_output(paths) == build_mutation_ready_output(
        paths,
        representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    schema = production_grouped_output(paths).schema
    assert schema is not None
    branch = schema["oneOf"][0]  # type: ignore[index]
    assert branch["properties"]["type"]["const"] == "multi_file_line_range_edit"  # type: ignore[index]
    assert branch["properties"]["edits"]["minItems"] == 2  # type: ignore[index]
    assert branch["properties"]["edits"]["maxItems"] == 4  # type: ignore[index]
    assert set(branch["properties"]["edits"]["items"]["required"]) == {  # type: ignore[index]
        "path",
        "start_line",
        "end_line",
        "new_text",
    }


def test_production_response_classification_is_explicit() -> None:
    assert classify_grouped_response("") is GroupedResponseClass.EMPTY
    assert (
        classify_grouped_response(json.dumps({"type": "final", "answer": "done"}))
        is GroupedResponseClass.FINAL
    )
    assert (
        classify_grouped_response(
            json.dumps(
                {
                    "type": "line_range_edit",
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 1,
                    "new_text": "x",
                }
            )
        )
        is GroupedResponseClass.SINGLE_FILE_EDIT
    )
    assert classify_grouped_response("not json") is GroupedResponseClass.SCHEMA_INVALID


def test_g4_premature_final_is_not_schema_invalid() -> None:
    responses = _p02_responses(
        production=json.dumps({"type": "final", "answer": "No safe edit."})
    )
    result = run_grouped_fixture(
        "mock",
        MockModel(responses, context_capacity=8192),
        SYNTHETIC_GROUPED_FIXTURES[1],
    )[3]
    assert result.response_class == GroupedResponseClass.FINAL.value
    assert result.failure == GroupedFailure.PREMATURE_FINAL.value


def test_g4_single_file_edit_is_incomplete_grouped_mutation() -> None:
    response = json.dumps(
        {
            "type": "line_range_edit",
            "path": "calculator.py",
            "start_line": 2,
            "end_line": 2,
            "new_text": "    return min(a + b, limit)\n",
        }
    )
    result = run_grouped_fixture(
        "mock",
        MockModel(_p02_responses(production=response), context_capacity=8192),
        SYNTHETIC_GROUPED_FIXTURES[1],
    )[3]
    assert result.response_class == GroupedResponseClass.SINGLE_FILE_EDIT.value
    assert result.failure == GroupedFailure.INCOMPLETE_GROUPED_MUTATION.value


class _FinishModel(Model):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("finish", "test")

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    @property
    def context_capacity(self) -> int:
        return 8192

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse('{"edits": [', FinishReason.MAX_TOKENS, self.identity)

    def close(self) -> None:
        pass


def test_output_limit_finish_is_classified_as_truncation() -> None:
    results = run_grouped_fixture(
        "finish", _FinishModel(), SYNTHETIC_GROUPED_FIXTURES[1]
    )
    assert results[1].response_class == GroupedResponseClass.OUTPUT_TRUNCATED.value
    assert results[2].failure == GroupedFailure.OUTPUT_TRUNCATED.value
    assert results[3].failure == GroupedFailure.OUTPUT_TRUNCATED.value


def test_transition_aggregation_and_serialization() -> None:
    results = run_grouped_fixture(
        "mock",
        MockModel(_p02_responses(), context_capacity=8192),
        SYNTHETIC_GROUPED_FIXTURES[1],
    )
    aggregate = aggregate_grouped_results("mock", results)
    assert (aggregate.g1_concepts, aggregate.g2_oracle_passes) == (1, 1)
    assert (aggregate.g3_oracle_passes, aggregate.g4_oracle_passes) == (1, 1)
    run = build_grouped_protocol_run(results)
    payload = grouped_protocol_to_dict(run)
    assert payload["suite"] == GROUPED_PROTOCOL_COMPATIBILITY_V1
    assert len(payload["results"]) == 4


def test_suite_has_three_two_file_tasks_and_two_language_families() -> None:
    assert [item.task_id for item in SYNTHETIC_GROUPED_FIXTURES] == [
        "P01",
        "P02",
        "P03",
    ]
    assert all(len(item.paths) == 2 for item in SYNTHETIC_GROUPED_FIXTURES)
    assert {item.language for item in SYNTHETIC_GROUPED_FIXTURES} == {"C17", "Python"}


def _production_session(tmp_path, responses):  # type: ignore[no-untyped-def]
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")
    model = MockModel(responses, context_capacity=8192)
    session = RepositoryChatSession(
        "mock",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        minimum_source_files=2,
        required_candidate_paths=("b.py", "a.py"),
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    return session, model


def _grouped_fixture_response() -> str:
    return json.dumps(
        {
            "type": "multi_file_line_range_edit",
            "edits": [
                {
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 1,
                    "new_text": "A = 2",
                },
                {
                    "path": "b.py",
                    "start_line": 1,
                    "end_line": 1,
                    "new_text": "B = 2",
                },
            ],
        }
    )


def test_full_forge_corrects_incomplete_single_file_group_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    single = json.dumps(
        {
            "type": "line_range_edit",
            "path": "a.py",
            "start_line": 1,
            "end_line": 1,
            "new_text": "A = 2",
        }
    )
    session, model = _production_session(
        tmp_path,
        (
            single,
            _grouped_fixture_response(),
            json.dumps({"type": "final", "answer": "done"}),
        ),
    )
    session.execute_task("Change both values")

    assert (tmp_path / "a.py").read_text() == "A = 2\n"
    assert (tmp_path / "b.py").read_text() == "B = 2\n"
    assert "multi_file_line_range_edit" in model.requests[0].messages[-1].content
    assert "one line_range_edit" not in model.requests[0].messages[-1].content
    assert "All required files" in model.requests[1].messages[-1].content
    rendered = [message.content for message in model.requests[0].messages]
    assert rendered.index("PATH: a.py") < rendered.index("PATH: b.py")
    assert any("   1 | A = 1" in value for value in rendered)
    assert any("   1 | B = 1" in value for value in rendered)


def test_full_forge_grouped_premature_final_correction_is_group_specific(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    final = json.dumps({"type": "final", "answer": "No change."})
    session, model = _production_session(tmp_path, (final, final))
    with pytest.raises(RepositoryOrchestrationError, match="no mutation proposed"):
        session.execute_task("Change both values")
    assert "multi_file_line_range_edit" in model.requests[1].messages[-1].content
    assert "every authorized path" in model.requests[1].messages[-1].content


def test_full_forge_grouped_schema_error_correction_reaches_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session, model = _production_session(
        tmp_path,
        (
            "not-json",
            _grouped_fixture_response(),
            json.dumps({"type": "final", "answer": "done"}),
        ),
    )
    session.execute_task("Change both values")
    correction = model.requests[1].messages[-1].content
    assert "grouped mutation JSON schema" in correction
    assert "multi_file_line_range_edit" in correction
    assert "tool_call" not in correction


def test_full_forge_result_records_transition_and_context_fallback() -> None:
    request = ModelRequest(
        (Message(MessageRole.USER, "task"),),
        output=production_grouped_output(("a.py", "b.py")),
    )
    response = ModelResponse(
        _grouped_fixture_response(),
        FinishReason.STOP,
        ModelIdentity("fixture", "mock"),
    )
    metrics = SimpleNamespace(
        mutation_proposed=True,
        mutation_group_preview_created=1,
        preview_created=1,
        mutations=1,
        verification_executed=1,
        verification_result="pass",
        mutation_ready_reached=True,
        tool_executions=4,
        model_calls=1,
        context_peak_estimate=0,
    )
    result = SimpleNamespace(
        elapsed_seconds=3.5,
        metrics=metrics,
        oracle=EvaluationOutcome.PASS,
        task_id="E08",
        seed=42,
        final_status="completed_verified",
    )
    recorded = full_forge_grouped_result(
        "mock",
        result,
        (ModelExchange(request, response, 0.5),),  # type: ignore[arg-type]
    )
    assert recorded.first_response_class == "MULTI_FILE_LINE_RANGE_EDIT"
    assert recorded.second_response_class is None
    assert recorded.correction_category is None
    assert recorded.preview_created and recorded.transaction_executed
    assert recorded.verification_result == "pass"
    assert recorded.context_peak == recorded.input_tokens
    assert recorded.total_elapsed_seconds == 3.5

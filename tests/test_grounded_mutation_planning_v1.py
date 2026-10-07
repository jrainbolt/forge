from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.grounded_mutation_planning_v1.planning import (
    PLAN_SCHEMA,
    PlanClassification,
    PlanningModel,
    validate_plan,
)
from benchmarks.grounded_mutation_planning_v1.runner import (
    Outcome,
    checkpoint_path,
    read_cell,
)
from benchmarks.grounded_mutation_planning_v1.suite import (
    CASES,
    Condition,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.models import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    OutputSpecification,
    ResponseFormat,
)

TRUSTED = frozenset({"src/a.py", "src/b.py"})


def _plan() -> dict[str, list[str]]:
    return {
        "required_behavior": ["Return a stable result"],
        "affected_components": ["src/a.py"],
        "required_changes": ["Update the existing result calculation"],
        "invariants": ["Preserve the public interface"],
    }


class CapturingModel(Model):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("fixture", "fixture")

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    @property
    def context_capacity(self) -> int:
        return 8192

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.responses.pop(0), FinishReason.STOP, self.identity)

    def close(self) -> None:
        return None


def _mutation_request() -> ModelRequest:
    return ModelRequest(
        (
            Message(MessageRole.SYSTEM, "Current authorized mutation targets:"),
            Message(MessageRole.USER, "Change the grounded implementation."),
        ),
        GenerationConfig(max_tokens=512, temperature=0, seed=42),
        OutputSpecification(
            ResponseFormat.JSON,
            {"type": "object", "properties": {}, "additionalProperties": True},
        ),
    )


def test_bounded_plan_schema_and_corpus_distribution() -> None:
    assert PLAN_SCHEMA["properties"]["required_behavior"]["maxItems"] == 3
    assert PLAN_SCHEMA["properties"]["required_changes"]["maxItems"] == 4
    assert PLAN_SCHEMA["properties"]["invariants"]["maxItems"] == 3
    validate_corpus()
    assert len(CASES) == 12


def test_plan_accepts_only_trusted_components() -> None:
    assert validate_plan(_plan(), TRUSTED).affected_components == ("src/a.py",)
    payload = _plan()
    payload["affected_components"] = ["src/hidden.py"]
    with pytest.raises(ValueError, match=PlanClassification.PLAN_INVALID.value):
        validate_plan(payload, TRUSTED)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_changes", ["Also update hidden/expected.py"]),
        ("required_changes", ["```diff\n+return 7\n```"]),
        ("required_behavior", []),
    ],
)
def test_hidden_paths_code_and_incomplete_plans_are_rejected(
    field: str, value: list[str]
) -> None:
    payload = _plan()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_plan(payload, TRUSTED)


def test_p1_adds_only_bounded_plan_and_preserves_generation_and_grounding() -> None:
    mutation = '{"type":"repository.edit_lines","arguments":{}}'
    backend = CapturingModel([json.dumps(_plan()), mutation])
    adapter = PlanningModel(backend, TRUSTED)
    request = _mutation_request()
    response = adapter.generate(request)
    assert response.text == mutation
    assert len(backend.requests) == 2
    planning, generated = backend.requests
    assert planning.generation == generated.generation == request.generation
    assert planning.messages[1:-1] == request.messages[1:]
    assert generated.messages[:-1] == request.messages
    additional = json.loads(generated.messages[-1].content.split("\n", 1)[1])
    assert set(additional) == set(_plan())
    assert adapter.record is not None
    assert adapter.record.plan_id is not None


def test_malformed_plan_does_not_fall_back_to_direct_mutation() -> None:
    backend = CapturingModel(["not json"])
    adapter = PlanningModel(backend, TRUSTED)
    assert adapter.generate(_mutation_request()).text == "not json"
    assert len(backend.requests) == 1
    assert adapter.record is not None
    assert adapter.record.classification == PlanClassification.PLAN_INVALID.value


def test_plan_identity_is_separate_and_durable_record_is_source_free() -> None:
    backend = CapturingModel([json.dumps(_plan()), "{}"])
    adapter = PlanningModel(backend, TRUSTED)
    adapter.generate(_mutation_request())
    assert adapter.record is not None
    record = {
        "plan_id": adapter.record.plan_id,
        "proposal_observation_id": "proposal-1",
        "classification": adapter.record.classification,
    }
    assert record["plan_id"] != record["proposal_observation_id"]
    assert standard_result_is_source_free(record)
    assert "Return a stable result" not in json.dumps(record)


def test_primary_and_repair_outcomes_are_distinct() -> None:
    primary = Outcome("primary", True, True, True, True, True, False, False, None)
    repair = Outcome("repair", True, True, True, True, True, True, True, None)
    assert primary.proposal_observation_id != repair.proposal_observation_id
    assert not primary.semantic_pass and repair.semantic_pass


def test_source_free_checkpoint_resume_is_exactly_once(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "A74-C01", Condition.P1)
    payload = {
        "case_id": "A74-C01",
        "condition": Condition.P1.value,
        "plan": {"plan_id": "a" * 64, "classification": "VALID"},
    }
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"case_id": "A74-C01"}) == payload
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})
    assert standard_result_is_source_free(payload)


def test_evaluator_artifacts_are_excluded_from_package() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "benchmarks.grounded_mutation_planning" not in configuration
    assert "eval-results" not in configuration

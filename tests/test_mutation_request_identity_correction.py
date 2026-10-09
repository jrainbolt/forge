from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from benchmarks.grounded_mutation_planning_v1.paired_identity import (
    MutationRequestOutcome,
    PairedInputCaptureModel,
    bind_request_proposals,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
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


def _request(*, repair: bool = False) -> ModelRequest:
    marker = (
        "Repair evidence is ready" if repair else "Authorized new paths: src/new.py"
    )
    return ModelRequest(
        (
            Message(MessageRole.SYSTEM, marker),
            Message(MessageRole.USER, "Create the requested implementation."),
        ),
        GenerationConfig(max_tokens=512, temperature=0, seed=42),
        OutputSpecification(
            ResponseFormat.JSON,
            {
                "type": "object",
                "properties": {"workspace_generation": {"const": 7}},
            },
        ),
    )


class FixtureModel(Model):
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = responses

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
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return ModelResponse(value, FinishReason.STOP, self.identity)

    def close(self) -> None:
        return None


def _capture(backend: Model, profile: str = "profile-a") -> PairedInputCaptureModel:
    return PairedInputCaptureModel(
        backend,
        task_identity="frozen-task",
        model_config_identity=profile,
        context_size=8192,
        representation="line_range",
    )


def test_request_identity_exists_before_generation() -> None:
    holder: dict[str, PairedInputCaptureModel] = {}

    class InspectingModel(FixtureModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            assert (
                holder["capture"].records[0].outcome is MutationRequestOutcome.IN_FLIGHT
            )
            return super().generate(request)

    capture = _capture(InspectingModel(["{}"]))
    holder["capture"] = capture
    capture.generate(_request())
    assert capture.records[0].mutation_request_id


def test_success_with_proposal_and_multiple_observations() -> None:
    capture = _capture(FixtureModel(["{}"]))
    capture.generate(_request())
    bound = bind_request_proposals(
        tuple(capture.records), (("proposal-1", "proposal-2"),)
    )
    assert bound[0].outcome is MutationRequestOutcome.COMPLETED_WITH_PROPOSAL
    assert bound[0].proposal_observation_ids == ("proposal-1", "proposal-2")
    assert bound[0].mutation_request_id not in bound[0].proposal_observation_ids


def test_h09_no_proposal_regression_retains_complete_identity() -> None:
    request = _request()
    text = "\n".join(message.content for message in request.messages)
    old_predicate = (
        request.output.schema is not None
        and "Current authorized mutation targets:" in text
    )
    assert not old_predicate
    capture = _capture(FixtureModel(["I cannot make this change."]))
    capture.generate(request)
    bound = bind_request_proposals(tuple(capture.records), ())
    assert capture.record is not None
    assert bound[0].outcome is MutationRequestOutcome.COMPLETED_NO_PROPOSAL
    assert bound[0].proposal_observation_ids == ()


def test_malformed_output_and_model_error_retain_identity() -> None:
    malformed = _capture(FixtureModel([" "]))
    malformed.generate(_request())
    assert malformed.records[0].outcome is MutationRequestOutcome.OUTPUT_MALFORMED
    failed = _capture(FixtureModel([RuntimeError("backend failed")]))
    with pytest.raises(RuntimeError, match="backend failed"):
        failed.generate(_request())
    assert failed.records[0].outcome is MutationRequestOutcome.MODEL_ERROR


def test_repair_has_distinct_identity_and_primary_lineage() -> None:
    capture = _capture(FixtureModel(["{}", "{}"]))
    capture.generate(_request())
    capture.generate(_request(repair=True))
    primary, repair = capture.records
    assert primary.mutation_request_id != repair.mutation_request_id
    assert repair.request_kind == "REPAIR_MUTATION"
    assert repair.repair_parent_request_id == primary.mutation_request_id


def test_equivalent_inputs_ignore_profile_as_experimental_variable() -> None:
    first = _capture(FixtureModel(["{}"]), "profile-a")
    second = _capture(FixtureModel(["{}"]), "profile-b")
    first.generate(_request())
    second.generate(_request())
    assert (
        first.records[0].equivalence_identity == second.records[0].equivalence_identity
    )
    assert first.records[0].profile_identity != second.records[0].profile_identity
    assert first.records[0].mutation_request_id != second.records[0].mutation_request_id


def test_proposal_absence_does_not_change_request_equivalence() -> None:
    with_proposal = _capture(FixtureModel(["{}"]))
    without = _capture(FixtureModel(["prose"]))
    with_proposal.generate(_request())
    without.generate(_request())
    bound = bind_request_proposals(tuple(with_proposal.records), (("proposal",),))
    assert bound[0].equivalence_identity == without.records[0].equivalence_identity


def test_source_free_checkpoint_resume_and_absolute_paths_excluded(
    tmp_path: Path,
) -> None:
    request = _request()
    absolute = replace(
        request,
        messages=(
            Message(MessageRole.SYSTEM, "Authorized new paths: /Users/private/new.py"),
            request.messages[1],
        ),
    )
    capture = _capture(FixtureModel(["prose"]))
    capture.generate(absolute)
    payload = asdict(capture.records[0])
    encoded = json.dumps(payload)
    assert "/Users/private" not in encoded
    assert standard_result_is_source_free(payload)
    path = tmp_path / "request.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(encoded)
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})


def test_evaluator_identity_code_is_not_packaged() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "grounded_mutation_planning_v1" not in configuration

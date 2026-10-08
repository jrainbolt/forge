from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.grounded_mutation_contract_v1.contract import (
    MAX_BEHAVIOR_GOALS,
    MAX_COMPONENT_ROLES,
    MAX_INVARIANTS,
    ContractModel,
    build_contract,
)
from benchmarks.grounded_mutation_contract_v1.runner import (
    Outcome,
    checkpoint_path,
    read_cell,
)
from benchmarks.grounded_mutation_contract_v1.suite import (
    CASES,
    RUN_ID,
    Condition,
    validate_corpus,
)
from benchmarks.grounded_mutation_planning_v1.paired_identity import (
    capture_paired_input,
    compare_paired_inputs,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evidence_coverage import decompose_evidence_plan
from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    MockModel,
    ModelRequest,
    OutputSpecification,
    ResponseFormat,
)

TASK = "Ensure enqueue rejects duplicates while preserving insertion order."
TRUSTED = frozenset({"src/a.py", "src/new.py"})


def _request(targets: str = "src/a.py") -> ModelRequest:
    return ModelRequest(
        (
            Message(MessageRole.SYSTEM, "Mutation system"),
            Message(
                MessageRole.USER,
                f"Requested code change:\n{TASK}\n\n"
                f"Current authorized mutation targets:\n{targets}\n\n"
                "Current trusted source follows.",
            ),
            Message(MessageRole.USER, "PATH: src/a.py"),
            Message(MessageRole.USER, "Current trusted source:\nvalue = 1\n"),
            Message(MessageRole.USER, "END FILE: src/a.py"),
        ),
        GenerationConfig(512, 0, 42),
        OutputSpecification(
            ResponseFormat.JSON,
            {
                "type": "object",
                "properties": {"workspace_generation": {"const": 0}},
            },
        ),
    )


def _identity(request: ModelRequest):  # type: ignore[no-untyped-def]
    return capture_paired_input(
        request,
        task_identity="task",
        model_config_identity="model",
        context_size=8192,
        representation="line_range",
    )


def test_contract_reuses_a64_evidence_goals_from_production_visible_task() -> None:
    contract = build_contract(TASK, _request(), TRUSTED)
    plan = decompose_evidence_plan(TASK)
    assert contract.behavior_goals == tuple(
        goal.description for goal in plan.goals if goal.required
    )


def test_contract_is_bounded_and_corpus_is_unchanged() -> None:
    contract = build_contract(TASK, _request(), TRUSTED)
    assert len(contract.behavior_goals) <= MAX_BEHAVIOR_GOALS
    assert len(contract.required_components) <= MAX_COMPONENT_ROLES
    assert len(contract.required_operation_roles) <= MAX_COMPONENT_ROLES
    assert len(contract.invariants) <= MAX_INVARIANTS
    validate_corpus()
    assert len(CASES) == 12


def test_unauthorized_or_hidden_expected_path_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_contract(TASK, _request("src/hidden.py"), TRUSTED)


def test_sentence_punctuation_is_not_part_of_visible_path() -> None:
    task = "Create src/new.py. Use the trusted src/a.py integration."
    contract = build_contract(task, _request("CREATE src/new.py"), TRUSTED)
    assert contract.required_operation_roles == ("CREATE src/new.py",)


def test_patch_or_code_content_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_contract("Ensure value works while return value", _request(), TRUSTED)


def test_create_authority_is_represented_without_expected_contents() -> None:
    contract = build_contract(TASK, _request("CREATE src/new.py"), TRUSTED)
    assert contract.required_components == ("src/new.py", "src/a.py")
    assert contract.required_operation_roles == ("CREATE src/new.py",)
    assert "value = 1" not in json.dumps(asdict(contract))


def test_pure_create_guidance_authority_is_represented_and_paired() -> None:
    request = _request("")
    request = ModelRequest(
        (
            *request.messages,
            Message(
                MessageRole.USER,
                "Create the requested file. Authorized new paths: src/new.py",
            ),
        ),
        request.generation,
        request.output,
    )
    contract = build_contract(TASK, request, TRUSTED)
    assert contract.required_operation_roles == ("CREATE src/new.py",)
    assert contract.required_components == ("src/new.py", "src/a.py")
    identity = _identity(request)
    assert (
        identity.authority_path_identity
        != _identity(_request()).authority_path_identity
    )


def test_pure_create_guidance_overrides_read_only_target_block() -> None:
    request = _request("src/a.py")
    request = ModelRequest(
        (
            *request.messages,
            Message(MessageRole.USER, "Authorized new paths: src/new.py"),
        ),
        request.generation,
        request.output,
    )
    contract = build_contract(TASK, request, TRUSTED)
    assert contract.required_operation_roles == ("CREATE src/new.py",)
    assert set(contract.required_components) == {"src/a.py", "src/new.py"}


def test_contract_adapter_adds_no_model_call_and_only_one_bounded_message() -> None:
    backend = MockModel(['{"type":"final","answer":"x"}'], context_capacity=8192)
    adapter = ContractModel(backend, TASK, TRUSTED)
    adapter.generate(_request())
    assert len(backend.requests) == 1
    assert backend.requests[0].messages[:-1] == _request().messages
    assert "MutationContract" not in backend.requests[0].messages[-1].content
    assert adapter.record is not None and adapter.record.classification == "VALID"


def test_m0_m1_common_pair_identity_excludes_contract_and_condition() -> None:
    direct = _identity(_request())
    augmented = _identity(_request())
    assert compare_paired_inputs(direct, augmented) == ()
    assert direct.common_identity == augmented.common_identity
    assert "contract" not in asdict(direct)
    assert "condition" not in asdict(direct)
    assert Condition.M0.value != Condition.M1.value


def test_proposal_local_funnel_and_primary_repair_are_separate() -> None:
    primary = Outcome("p1", True, True, True, True, True, True, False, False, None)
    repair = Outcome("p2", True, True, True, True, True, True, True, True, None)
    assert primary.proposal_observation_id != repair.proposal_observation_id
    assert primary.model_output and primary.schema_valid
    assert primary.production_validatable and primary.transaction_ready
    assert not primary.semantic_pass and repair.semantic_pass


def test_checkpoint_resume_exactly_once_and_source_free(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "A74-C01", Condition.M1)
    payload = {
        "run_identity": RUN_ID,
        "case_id": "A74-C01",
        "condition": Condition.M1.value,
        "contract": {"contract_id": "a" * 64, "classification": "VALID"},
    }
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"run_identity": RUN_ID}) == payload
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})
    assert standard_result_is_source_free(payload)


def test_evaluator_and_results_are_excluded_from_package() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "benchmarks.grounded_mutation_contract" not in configuration
    assert "eval-results" not in configuration

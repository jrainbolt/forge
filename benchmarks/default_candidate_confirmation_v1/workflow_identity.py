"""Source-free identity for a complete coding-workflow attempt."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from benchmarks.realistic_coding_v2.suite import FrozenTask


class WorkflowOutcome(StrEnum):
    IN_FLIGHT = "WORKFLOW_IN_FLIGHT"
    COMPLETED_WITH_MUTATION_REQUEST = "WORKFLOW_COMPLETED_WITH_MUTATION_REQUEST"
    COMPLETED_NO_MUTATION_REQUEST = "WORKFLOW_COMPLETED_NO_MUTATION_REQUEST"
    MODEL_PROTOCOL_FAILURE = "WORKFLOW_MODEL_PROTOCOL_FAILURE"
    GROUNDING_FAILURE = "WORKFLOW_GROUNDING_FAILURE"
    TOOL_FAILURE = "WORKFLOW_TOOL_FAILURE"
    CONTEXT_FAILURE = "WORKFLOW_CONTEXT_FAILURE"
    MODEL_ERROR = "WORKFLOW_MODEL_ERROR"
    INTERRUPTED = "WORKFLOW_INTERRUPTED"
    OTHER_FAILURE = "WORKFLOW_OTHER_FAILURE"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkflowAttempt:
    identity_version: int
    workflow_attempt_id: str
    equivalence_identity: str
    model_profile_identity: str
    frozen_task_identity: str
    evaluator_corpus_identity: str
    initial_workspace_identity: str
    source_generation_identity: str
    user_task_text_identity: str
    initial_authority_identity: str
    generation_settings_identity: str
    representation_identity: str
    grounding_configuration_identity: str
    repair_ceiling_identity: str
    workflow_kind: str
    outcome: WorkflowOutcome
    mutation_request_ids: tuple[str, ...] = ()
    proposal_lineage: tuple[tuple[str, str], ...] = ()
    transaction_lineage: tuple[tuple[str, str], ...] = ()


def begin_workflow_attempt(
    definition: FrozenTask,
    *,
    evaluator_corpus_identity: str,
    repository_identity: str,
    model_profile_identity: str,
    seed: int,
    temperature: float,
    context_size: int,
    output_budget: int,
    representation: str,
) -> WorkflowAttempt:
    task = definition.production_task
    components = {
        "frozen_task_identity": _hash(
            {
                "task_id": definition.task_id,
                "version": definition.version,
                "operation_class": definition.operation_class.value,
            }
        ),
        "evaluator_corpus_identity": evaluator_corpus_identity,
        "initial_workspace_identity": repository_identity,
        "source_generation_identity": _hash(
            {"repository_identity": repository_identity, "generation": 0}
        ),
        "user_task_text_identity": _hash(task.prompt),
        "initial_authority_identity": _hash(
            {
                "authority_mode": definition.authority_mode.value,
                "allowed_paths": task.allowed_paths,
                "create_paths": task.create_candidate_paths,
            }
        ),
        "generation_settings_identity": _hash(
            {
                "seed": seed,
                "temperature": temperature,
                "context_size": context_size,
                "output_budget": output_budget,
            }
        ),
        "representation_identity": _hash(representation),
        "grounding_configuration_identity": _hash(
            {
                "expected_files": task.expected_files,
                "required_candidate_paths": task.required_candidate_paths,
                "authority_mode": definition.authority_mode.value,
            }
        ),
        "repair_ceiling_identity": _hash(task.max_mutations),
        "workflow_kind": "COMPLETE_CODING_TASK",
    }
    equivalence = _hash(components)
    attempt_id = _hash(
        {
            "equivalence_identity": equivalence,
            "model_profile_identity": model_profile_identity,
        }
    )
    return WorkflowAttempt(
        1,
        attempt_id,
        equivalence,
        model_profile_identity,
        **components,
        outcome=WorkflowOutcome.IN_FLIGHT,
    )


def complete_workflow_attempt(
    attempt: WorkflowAttempt,
    *,
    outcome: WorkflowOutcome,
    mutation_request_ids: tuple[str, ...] = (),
    proposal_lineage: tuple[tuple[str, str], ...] = (),
    transaction_lineage: tuple[tuple[str, str], ...] = (),
) -> WorkflowAttempt:
    if attempt.outcome is not WorkflowOutcome.IN_FLIGHT:
        raise ValueError("workflow attempt already has a terminal outcome")
    if outcome is WorkflowOutcome.IN_FLIGHT:
        raise ValueError("workflow completion requires one terminal outcome")
    return replace(
        attempt,
        outcome=outcome,
        mutation_request_ids=mutation_request_ids,
        proposal_lineage=proposal_lineage,
        transaction_lineage=transaction_lineage,
    )


def workflow_payload(attempt: WorkflowAttempt) -> dict[str, object]:
    return asdict(attempt)

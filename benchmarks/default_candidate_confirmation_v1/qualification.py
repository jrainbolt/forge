"""Model-free workflow reachability qualification for the A78 corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from benchmarks.realistic_coding_v2.runner import snapshot
from benchmarks.realistic_coding_v2.suite import REPOSITORY, FrozenTask
from forge.evaluation.realworld import RealWorldEvaluationRunner, RealWorldFailure
from forge.evaluation.replay import source_state_identity
from forge.models import MockModel, MutationRepresentationPolicy

WORKFLOW_REACHABLE = "WORKFLOW_REACHABLE"
WORKFLOW_UNREACHABLE = "WORKFLOW_UNREACHABLE"


@dataclass(frozen=True, slots=True)
class WorkflowReachability:
    """Source-free evidence that normal execution crossed the model boundary."""

    task_id: str
    classification: str
    setup_succeeded: bool
    session_reached: bool
    model_calls: int
    final_status: str
    failure: str | None

    @property
    def reachable(self) -> bool:
        return self.classification == WORKFLOW_REACHABLE

    def payload(self) -> dict[str, object]:
        return asdict(self)


def qualify_workflow_reachability(
    definition: FrozenTask,
    *,
    representation: MutationRepresentationPolicy,
) -> WorkflowReachability:
    """Probe evaluator setup and session entry without a capability model."""
    model = MockModel(
        ('{"type":"final","answer":"qualification probe"}',),
        context_capacity=8192,
    )
    task = replace(definition.production_task, seeds=(42,))
    repository_identity = source_state_identity(REPOSITORY)
    result = (
        RealWorldEvaluationRunner(
            "workflow-reachability-probe",
            model,
            REPOSITORY,
            mutation_representation=representation,
        )
        .run((task,), snapshot(repository_identity))
        .results[0]
    )
    model_calls = len(model.requests)
    setup_succeeded = result.failure is not RealWorldFailure.INFRASTRUCTURE
    session_reached = model_calls >= 1
    classification = (
        WORKFLOW_REACHABLE
        if setup_succeeded and session_reached
        else WORKFLOW_UNREACHABLE
    )
    return WorkflowReachability(
        definition.task_id,
        classification,
        setup_succeeded,
        session_reached,
        model_calls,
        result.final_status,
        result.failure.value if result.failure is not None else None,
    )

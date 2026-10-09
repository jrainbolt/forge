"""Source-free A80 execution and failure diagnosis over production orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import (
    CellResult as A78CellResult,
)
from benchmarks.default_candidate_confirmation_v1.runner import (
    run_cell as run_a78_cell,
)
from benchmarks.realistic_coding_v2.runner import MUTATION_TYPES, envelope_shape
from benchmarks.realistic_coding_v2.suite import FrozenTask, OperationClass
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.models import (
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)

from .suite import RUN_ID, SUITE, VERSION, SelectionCondition


class ShapeCaptureModel(Model):
    """Capture operation roles and authorized paths without retaining source text."""

    def __init__(self, backend: Model, allowed_paths: frozenset[str]) -> None:
        self.backend = backend
        self.allowed_paths = allowed_paths
        self.mutation_shapes: list[dict[str, object]] = []

    @property
    def identity(self) -> ModelIdentity:
        return self.backend.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = self.backend.generate(request)
        shape = envelope_shape(response.text, self.allowed_paths)
        if shape.get("type") in MUTATION_TYPES:
            self.mutation_shapes.append(shape)
        return response

    def close(self) -> None:
        self.backend.close()


@dataclass(frozen=True, slots=True)
class OperationalCell:
    suite: str
    suite_version: int
    run_identity: str
    corpus_identity: str
    condition: str
    selection_source: str
    selected_profile: str
    task_id: str
    operation_class: str
    operation_shape: dict[str, object] | None
    primary_classification: str
    created_file_materialized: bool
    integration_failure: bool
    workflow_attempt: dict[str, object]
    workflow_trace: dict[str, object]
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]
    proposal: dict[str, object]
    primary_semantic_pass: bool
    repair: dict[str, object] | None
    failure: str
    generation_calls: int
    median_generation_latency_seconds: float | None
    input_tokens: int
    output_tokens: int


def _roles(shape: dict[str, object] | None) -> list[tuple[str, str]]:
    if shape is None:
        return []
    children = shape.get("children")
    if not isinstance(children, list):
        return []
    result = []
    for child in children:
        if isinstance(child, dict):
            kind = child.get("type")
            path = child.get("path")
            if isinstance(kind, str) and isinstance(path, str):
                result.append(("CREATE" if kind == "create_file" else "EDIT", path))
    return result


def classify_create(cell: A78CellResult) -> str:
    proposal = cell.proposal
    if not proposal["model_output"]:
        return "NO_PROPOSAL"
    if not proposal["schema_valid"]:
        return "MODEL_PROTOCOL_FAILURE"
    if not proposal["mechanically_materializable"]:
        return "CREATE_MATERIALIZATION_FAILURE"
    if not proposal["production_validatable"]:
        return "CREATE_AUTHORITY_FAILURE"
    if not proposal["transaction_applied"]:
        return "INTEGRATION_FAILURE"
    if proposal.get("semantic_failure") == "BUILD_PREVENTS_ORACLE":
        return "BUILD_FAILURE"
    if not proposal["verification_pass"]:
        return "VERIFICATION_FAILURE"
    if not cell.primary_semantic_pass:
        return "WRONG_BEHAVIOR"
    return "PASS"


def classify_mixed(
    definition: FrozenTask,
    cell: A78CellResult,
    shape: dict[str, object] | None,
) -> str:
    proposal = cell.proposal
    if not proposal["model_output"] or not proposal["schema_valid"]:
        return "PROTOCOL_FAILURE"
    roles = _roles(shape)
    paths = [path for _, path in roles]
    if len(paths) != len(set(paths)):
        return "DUPLICATE_OPERATION"
    actual = set(roles)
    expected_edits = {("EDIT", path) for path in definition.edit_paths}
    expected_creates = {("CREATE", path) for path in definition.create_paths}
    if any(("CREATE", path) in actual for path in definition.edit_paths) or any(
        ("EDIT", path) in actual for path in definition.create_paths
    ):
        return "WRONG_OPERATION_ROLE"
    if not expected_edits <= actual:
        return "MISSING_EDIT"
    if not expected_creates <= actual:
        return "MISSING_CREATE"
    if not proposal["mechanically_materializable"]:
        return "MATERIALIZATION_FAILURE"
    if not proposal["transaction_applied"]:
        return "TRANSACTION_FAILURE"
    if proposal.get("semantic_failure") == "BUILD_PREVENTS_ORACLE":
        return "BUILD_FAILURE"
    if not proposal["verification_pass"]:
        return "PARTIAL_IMPLEMENTATION"
    if not cell.primary_semantic_pass:
        return (
            "PARTIAL_IMPLEMENTATION"
            if proposal.get("semantic_failure") == "PARTIAL_MULTI_FILE_CHANGE"
            else "WRONG_BEHAVIOR"
        )
    return "PASS"


def run_cell(
    definition: FrozenTask,
    condition: SelectionCondition,
    selected_profile: str,
    backend: Model,
    *,
    artifact: str,
    artifact_size: int,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation,  # type: ignore[no-untyped-def]
) -> OperationalCell:
    capture = ShapeCaptureModel(
        backend, frozenset((*definition.edit_paths, *definition.create_paths))
    )
    inner = run_a78_cell(
        definition,
        selected_profile,
        capture,
        artifact=artifact,
        artifact_size=artifact_size,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
        run_identity=RUN_ID,
    )
    shape = capture.mutation_shapes[0] if capture.mutation_shapes else None
    if definition.operation_class is OperationClass.CREATE:
        classification = classify_create(inner)
    elif definition.operation_class is OperationClass.MIXED_EDIT_CREATE:
        classification = classify_mixed(definition, inner, shape)
    else:
        classification = "PASS" if inner.primary_semantic_pass else inner.failure
    proposal = inner.proposal
    return OperationalCell(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        condition.value,
        "implicit_coding_default"
        if condition is SelectionCondition.DEFAULT
        else "explicit_override",
        selected_profile,
        definition.task_id,
        definition.operation_class.value,
        shape,
        classification,
        bool(proposal["transaction_applied"] and definition.create_paths),
        bool(proposal["transaction_applied"] and not proposal["verification_pass"]),
        inner.workflow_attempt,
        inner.workflow_trace,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
        inner.proposal,
        inner.primary_semantic_pass,
        inner.repair,
        inner.failure,
        inner.generation_calls,
        inner.median_generation_latency_seconds,
        inner.input_tokens,
        inner.output_tokens,
    )


def commit_cell(path: Path, cell: OperationalCell) -> None:
    payload = asdict(cell)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A80 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path) -> dict[str, object] | None:
    return resume_checkpoint(path)

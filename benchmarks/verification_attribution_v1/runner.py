"""A73 fresh diagnostic execution and source-free attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import (
    MUTATION_TYPES,
    RecordingModel,
    snapshot,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY, FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.runner import (
    InstrumentationError,
    _assert_observations,
    _schema_valid,
    bind_proposal_evidence,
)
from benchmarks.verification_attribution_v1.protocol import (
    quadrant,
    semantic_failure,
    verification_failure,
)
from benchmarks.verification_attribution_v1.suite import (
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
    DiagnosticCase,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    run_oracle,
)
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class ProposalDiagnostic:
    proposal_observation_id: str
    transaction_attempt_ids: tuple[str, ...]
    applied: bool
    verification_result: str
    oracle_result: str
    quadrant: str | None
    verification_failure: str | None
    semantic_failure: str | None
    verification_stage: str | None
    diagnostic_targets: tuple[str, ...]
    stdout_sha256: str | None
    stderr_sha256: str | None
    focused_result: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    suite: str
    suite_version: int
    schema_version: int
    case_id: str
    profile: str
    task_id: str
    operation_class: str
    a72_outcome: str
    seed: int
    corpus_identity: str
    repository_identity: str
    model_artifact: str
    model_config_identity: str
    baseline_valid: bool
    verification_plan_audit: str
    proposals: tuple[ProposalDiagnostic, ...]
    observation_count: int
    identity_mismatch_count: int
    final_status: str


def require_valid_baseline(case_id: str, baseline_valid: bool) -> None:
    if not baseline_valid:
        raise InstrumentationError(f"baseline invalid: {case_id}")


def run_case(
    case: DiagnosticCase,
    definition: FrozenTask,
    backend: Model,
    *,
    artifact: str,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
    baseline_valid: bool,
) -> CaseResult:
    require_valid_baseline(case.case_id, baseline_valid)
    recorder = RecordingModel(
        backend, frozenset((*definition.edit_paths, *definition.create_paths))
    )
    observed_evidence: list[dict[str, object]] = []
    primary_semantic: dict[str, str] = {}

    def evidence_callback(_task, _workspace, payload):  # type: ignore[no-untyped-def]
        observed_evidence.append(dict(payload))

    def primary_callback(task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        applied = [
            item
            for item in observed_evidence
            if item.get("event") == "transaction_result"
            and item.get("transaction_outcome") == "success"
        ]
        if not applied:
            raise InstrumentationError("applied mutation lacks proposal identity")
        proposal_id = str(applied[-1]["proposal_observation_id"])
        primary_semantic[proposal_id] = run_oracle(
            workspace, task.oracle_commands
        ).value

    raw = (
        RealWorldEvaluationRunner(
            case.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=primary_callback,
            proposal_evidence_callback=evidence_callback,
        )
        .run(
            (replace(definition.production_task, seeds=(SEED,)),),
            snapshot(repository_identity),
        )
        .results[0]
    )
    envelopes = tuple(
        item for item in recorder.envelopes if item.get("type") in MUTATION_TYPES
    )
    metadata = raw.mutation_ready_metadata
    classifications = raw.mutation_observation_classifications
    _assert_observations(metadata, classifications)
    if len(envelopes) != len(metadata):
        raise InstrumentationError("proposal/observation count mismatch")
    evidence = list(raw.proposal_evidence)
    existing_semantic = {
        str(item.get("proposal_observation_id"))
        for item in evidence
        if item.get("event") == "semantic_result"
    }
    evidence.extend(
        {
            "event": "semantic_result",
            "proposal_observation_id": proposal_id,
            "semantic_outcome": outcome,
        }
        for proposal_id, outcome in primary_semantic.items()
        if proposal_id not in existing_semantic
    )
    binding = bind_proposal_evidence(metadata, tuple(evidence))
    diagnostics = []
    for index, recorded in enumerate(metadata):
        proposal_id = str(recorded["proposal_observation_id"])
        bound = binding[proposal_id]
        if not bound["transaction_applied"]:
            continue
        if not _schema_valid(definition, envelopes[index]):
            raise InstrumentationError("applied proposal is not schema-valid")
        verification_events = [
            item
            for item in evidence
            if item.get("event") == "verification_result"
            and item.get("proposal_observation_id") == proposal_id
        ]
        semantic_events = [
            item
            for item in evidence
            if item.get("event") == "semantic_result"
            and item.get("proposal_observation_id") == proposal_id
        ]
        if len(verification_events) != 1 or len(semantic_events) != 1:
            raise InstrumentationError("proposal diagnostic evidence incomplete")
        verification_event = verification_events[0]
        verification_pass = verification_event.get("verification_outcome") == "pass"
        oracle_pass = (
            semantic_events[0].get("semantic_outcome") == EvaluationOutcome.PASS.value
        )
        failure = (
            verification_failure(verification_event, oracle_pass=oracle_pass).value
            if not verification_pass
            else None
        )
        semantic = (
            semantic_failure(definition.operation_class.value, failure).value
            if not oracle_pass
            else None
        )
        attempts = bound["transaction_attempts"]
        assert isinstance(attempts, list)
        diagnostics.append(
            ProposalDiagnostic(
                proposal_id,
                tuple(str(item["transaction_attempt_id"]) for item in attempts),
                True,
                "PASS" if verification_pass else "FAIL",
                "PASS" if oracle_pass else "FAIL",
                quadrant(verification_pass, oracle_pass).value,
                failure,
                semantic,
                str(verification_event.get("verification_stage"))
                if verification_event.get("verification_stage")
                else None,
                tuple(
                    str(item)
                    for item in verification_event.get("diagnostic_targets", ())
                ),
                str(verification_event.get("stdout_sha256"))
                if verification_event.get("stdout_sha256")
                else None,
                str(verification_event.get("stderr_sha256"))
                if verification_event.get("stderr_sha256")
                else None,
                "NOT_CONFIGURED",
            )
        )
    return CaseResult(
        SUITE,
        VERSION,
        SCHEMA_VERSION,
        case.case_id,
        case.profile,
        case.task_id,
        definition.operation_class.value,
        case.a72_outcome,
        SEED,
        corpus_identity,
        repository_identity,
        artifact,
        model_config_identity,
        baseline_valid,
        "BROADER_BUT_RELEVANT",
        tuple(diagnostics),
        len(metadata),
        0,
        raw.final_status,
    )


def checkpoint_path(root: Path, case_id: str) -> Path:
    return root / "cases" / f"{case_id}.json"


def commit_case(path: Path, result: CaseResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise InstrumentationError("A73 result is not source-free")
    atomic_checkpoint(path, payload)


def read_case(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A73 checkpoint identity mismatch: {path.name}")
    return payload

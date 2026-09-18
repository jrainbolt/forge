"""A47 repair-case corpus and verification-grounded comparison records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.realistic_semantic import (
    RealisticSemanticResult,
    RealisticSemanticRun,
)

REPAIR_EFFECTIVENESS_V1 = "repair-effectiveness-v1"
REPAIR_EFFECTIVENESS_SUITE_VERSION = 1
REPAIR_EFFECTIVENESS_SCHEMA_VERSION = 1


class RepairCaseType(Enum):
    COMPILE_FAILURE = "V1_COMPILE_FAILURE"
    TEST_FAILURE = "V2_TEST_FAILURE"
    CONFIGURE_FAILURE = "V3_CONFIGURE_FAILURE"
    SEMANTIC_ORACLE_FAILURE_WITH_VERIFICATION_PASS = (
        "V4_SEMANTIC_ORACLE_FAILURE_WITH_VERIFICATION_PASS"
    )
    VERIFICATION_PASS_ORACLE_FAIL = "V5_VERIFICATION_PASS_ORACLE_FAIL"
    VERIFICATION_FAIL_ORACLE_PASS = "V6_VERIFICATION_FAIL_ORACLE_PASS"
    SOURCE_REACQUISITION_FAILURE = "V7_SOURCE_REACQUISITION_FAILURE"
    PROTOCOL_EDIT_CONSTRUCTION_FAILURE = "V8_PROTOCOL_EDIT_CONSTRUCTION_FAILURE"


class RepairCondition(Enum):
    CURRENT_REPAIR_BASELINE = "R0"
    CURRENT_SOURCE_WITH_FIRST_MUTATION = "R1"


class RepairOutputClass(Enum):
    ACTION_VALID = "REPAIR_ACTION_VALID"
    SCHEMA_INVALID = "REPAIR_SCHEMA_INVALID"
    NO_OP = "REPAIR_NO_OP"
    WRONG_PATH = "REPAIR_WRONG_PATH"
    VERIFICATION_FAIL = "REPAIR_VERIFICATION_FAIL"
    SEMANTIC_RECOVERY = "REPAIR_SEMANTIC_RECOVERY"
    SEMANTIC_FAIL = "REPAIR_SEMANTIC_FAIL"


class RepairRegression(Enum):
    NO_CHANGE = "NO_CHANGE"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
    VERIFICATION_IMPROVED = "VERIFICATION_IMPROVED"
    VERIFICATION_REGRESSED = "VERIFICATION_REGRESSED"
    SEMANTIC_RECOVERED = "SEMANTIC_RECOVERED"
    SEMANTIC_STILL_FAIL = "SEMANTIC_STILL_FAIL"
    SEMANTIC_REGRESSED = "SEMANTIC_REGRESSED"


@dataclass(frozen=True, slots=True)
class RepairCase:
    case_id: str
    model_profile: str
    task_id: str
    seed: int
    language: str
    mutation_kind: str
    case_type: RepairCaseType
    eligible: bool
    eligibility_reason: str
    primary_mutation_executed: bool
    repair_attempted: bool
    verification_status: str
    semantic_oracle: str
    failure_layer: str


@dataclass(frozen=True, slots=True)
class RepairTrialResult:
    case_id: str
    model_profile: str
    task_id: str
    condition: RepairCondition
    output_class: RepairOutputClass
    regression: RepairRegression
    repair_ready: bool
    schema_valid: bool
    target_valid: bool
    material_delta: bool
    preview_created: bool
    transaction_executed: bool
    verification_rerun: bool
    verification_passed: bool
    semantic_passed: bool
    input_tokens: int | None
    output_tokens: int | None
    generation_seconds: float
    verification_seconds: float


@dataclass(frozen=True, slots=True)
class RepairEffectivenessRun:
    suite: str
    suite_version: int
    schema_version: int
    repository_identity: str
    seed: int
    context_capacity: int
    output_budget: int
    cases: tuple[RepairCase, ...]
    trials: tuple[RepairTrialResult, ...]
    canonical_unchanged: bool


_LANGUAGES = {
    "R01": "Python",
    "R02": "Python",
    "R03": "C17",
    "R04": "C17",
    "R05": "Python",
    "R06": "Python",
    "R07": "C17",
    "R08": "C17",
}


def build_repair_case_corpus(
    runs: tuple[RealisticSemanticRun, ...],
) -> tuple[RepairCase, ...]:
    """Select failed executed primary mutations without inventing repair evidence."""
    cases: list[RepairCase] = []
    for run in runs:
        for result in run.results:
            if not result.transaction_executed:
                continue
            verification_passed = result.verification_status in {"pass", "passed"}
            semantic_passed = result.semantic_oracle == "PASS"
            if verification_passed and semantic_passed:
                continue
            case_type, eligible, reason = _classify_case(
                result, verification_passed, semantic_passed
            )
            cases.append(
                RepairCase(
                    f"{result.model_profile}:{result.seed}:{result.task_id}",
                    result.model_profile,
                    result.task_id,
                    result.seed,
                    _LANGUAGES.get(result.task_id, "unknown"),
                    result.mutation_kind,
                    case_type,
                    eligible,
                    reason,
                    True,
                    result.repair_used,
                    result.verification_status,
                    result.semantic_oracle,
                    result.failure_layer,
                )
            )
    return tuple(cases)


def build_repair_effectiveness_run(
    *,
    repository_identity: str,
    cases: tuple[RepairCase, ...],
    trials: tuple[RepairTrialResult, ...],
    canonical_unchanged: bool = True,
) -> RepairEffectivenessRun:
    known = {case.case_id for case in cases if case.eligible}
    if any(trial.case_id not in known for trial in trials):
        raise ValueError("repair trial does not reference an eligible corpus case")
    grouped = {(trial.case_id, trial.model_profile) for trial in trials}
    for key in grouped:
        conditions = {
            trial.condition
            for trial in trials
            if (trial.case_id, trial.model_profile) == key
        }
        if conditions != set(RepairCondition):
            raise ValueError("each repair case/model requires exactly R0 and R1")
    return RepairEffectivenessRun(
        REPAIR_EFFECTIVENESS_V1,
        REPAIR_EFFECTIVENESS_SUITE_VERSION,
        REPAIR_EFFECTIVENESS_SCHEMA_VERSION,
        repository_identity,
        42,
        8192,
        512,
        cases,
        trials,
        canonical_unchanged,
    )


def repair_effectiveness_to_dict(run: RepairEffectivenessRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    payload = convert(asdict(run))
    assert isinstance(payload, dict)
    return payload


def write_repair_effectiveness_json(run: RepairEffectivenessRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(repair_effectiveness_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_repair_effectiveness_run(path: Path) -> RepairEffectivenessRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("suite") != REPAIR_EFFECTIVENESS_V1:
        raise ValueError("not a repair-effectiveness-v1 artifact")
    cases = tuple(
        RepairCase(
            **{
                **item,
                "case_type": RepairCaseType(item["case_type"]),
            }
        )
        for item in payload["cases"]
    )
    trials = tuple(
        RepairTrialResult(
            **{
                **item,
                "condition": RepairCondition(item["condition"]),
                "output_class": RepairOutputClass(item["output_class"]),
                "regression": RepairRegression(item["regression"]),
            }
        )
        for item in payload["trials"]
    )
    return RepairEffectivenessRun(
        payload["suite"],
        payload["suite_version"],
        payload["schema_version"],
        payload["repository_identity"],
        payload["seed"],
        payload["context_capacity"],
        payload["output_budget"],
        cases,
        trials,
        payload["canonical_unchanged"],
    )


def _classify_case(
    result: RealisticSemanticResult,
    verification_passed: bool,
    semantic_passed: bool,
) -> tuple[RepairCaseType, bool, str]:
    if result.failure_layer == "SOURCE_ACQUISITION_FAILED":
        return (
            RepairCaseType.SOURCE_REACQUISITION_FAILURE,
            False,
            "fresh trusted repair source was unavailable",
        )
    if result.failure_layer in {"PROTOCOL_FAILED", "EDIT_CONSTRUCTION_FAILED"}:
        return (
            RepairCaseType.PROTOCOL_EDIT_CONSTRUCTION_FAILURE,
            False,
            "no executed post-mutation verification failure authorizes repair",
        )
    if verification_passed and not semantic_passed:
        return (
            RepairCaseType.VERIFICATION_PASS_ORACLE_FAIL,
            False,
            "hidden evaluator oracle cannot trigger production repair",
        )
    if not verification_passed and semantic_passed:
        return (
            RepairCaseType.VERIFICATION_FAIL_ORACLE_PASS,
            True,
            "verification failure is production-visible; attribution remains binding",
        )
    if result.task_id == "R03":
        return (
            RepairCaseType.COMPILE_FAILURE,
            True,
            "recorded C17 build/compiler failure",
        )
    return (
        RepairCaseType.TEST_FAILURE,
        True,
        "recorded mutation-associated or unattributed build/test failure",
    )

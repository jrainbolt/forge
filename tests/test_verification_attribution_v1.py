from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.runner import bind_proposal_evidence
from benchmarks.verification_attribution_v1.protocol import (
    Quadrant,
    VerificationFailure,
    quadrant,
    verification_failure,
)
from benchmarks.verification_attribution_v1.runner import (
    InstrumentationError,
    checkpoint_path,
    read_case,
    require_valid_baseline,
)
from benchmarks.verification_attribution_v1.suite import (
    CASES,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evaluation.replay import source_state_identity


@pytest.mark.parametrize(
    ("verification", "oracle", "expected"),
    [
        (True, True, Quadrant.V_PASS_O_PASS),
        (True, False, Quadrant.V_PASS_O_FAIL),
        (False, True, Quadrant.V_FAIL_O_PASS),
        (False, False, Quadrant.V_FAIL_O_FAIL),
    ],
)
def test_all_verification_oracle_quadrants(
    verification: bool, oracle: bool, expected: Quadrant
) -> None:
    assert quadrant(verification, oracle) is expected


def test_oracle_and_verification_evidence_are_independent_when_verification_fails() -> (
    None
):
    metadata = ({"proposal_observation_id": "p1"},)
    evidence = (
        {
            "event": "transaction_result",
            "proposal_observation_id": "p1",
            "transaction_attempt_id": "t1",
            "transaction_outcome": "success",
        },
        {
            "event": "verification_result",
            "proposal_observation_id": "p1",
            "verification_outcome": "fail",
        },
        {
            "event": "semantic_result",
            "proposal_observation_id": "p1",
            "semantic_outcome": "PASS",
        },
    )
    bound = bind_proposal_evidence(metadata, evidence)["p1"]
    assert not bound["verification_pass"]
    assert bound["semantic_pass"]
    assert quadrant(False, True) is Quadrant.V_FAIL_O_PASS


def test_precise_failure_stage_and_unrelated_full_suite_classification() -> None:
    assert (
        verification_failure(
            {"verification_stage": "project.configure"}, oracle_pass=False
        )
        is VerificationFailure.CONFIGURE_FAILURE
    )
    assert (
        verification_failure({"verification_stage": "project.build"}, oracle_pass=False)
        is VerificationFailure.BUILD_FAILURE
    )
    assert (
        verification_failure({"verification_stage": "project.test"}, oracle_pass=True)
        is VerificationFailure.FULL_SUITE_REGRESSION
    )
    assert (
        verification_failure(
            {"verification_stage": "project.test", "timed_out": True},
            oracle_pass=False,
        )
        is VerificationFailure.PROCESS_TIMEOUT
    )


def test_focused_full_oracle_distinction_and_repair_identity() -> None:
    first = {
        "proposal_observation_id": "primary",
        "focused_result": "NOT_CONFIGURED",
        "verification_result": "FAIL",
        "oracle_result": "FAIL",
    }
    repair = {
        "proposal_observation_id": "repair",
        "focused_result": "NOT_CONFIGURED",
        "verification_result": "PASS",
        "oracle_result": "PASS",
    }
    assert first["proposal_observation_id"] != repair["proposal_observation_id"]
    assert (
        first["focused_result"],
        first["verification_result"],
        first["oracle_result"],
    ) == (
        "NOT_CONFIGURED",
        "FAIL",
        "FAIL",
    )


def test_baseline_invalid_is_excluded() -> None:
    with pytest.raises(InstrumentationError, match="baseline invalid"):
        require_valid_baseline("A73-F01", False)


def test_checkpoint_resume_source_free_package_exclusion_and_canonical_identity(
    tmp_path: Path,
) -> None:
    payload = {
        "case_id": "A73-F01",
        "corpus_identity": corpus_identity(),
        "proposals": [
            {
                "proposal_observation_id": "p1",
                "stdout_sha256": "a" * 64,
                "stderr_sha256": "b" * 64,
            }
        ],
    }
    path = checkpoint_path(tmp_path, "A73-F01")
    atomic_checkpoint(path, payload)
    assert read_case(path, {"case_id": "A73-F01"}) == json.loads(json.dumps(payload))
    assert standard_result_is_source_free(payload)
    assert source_state_identity(REPOSITORY) == source_state_identity(REPOSITORY)
    assert (
        'where = ["src"]' in (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )


def test_frozen_corpus_coverage() -> None:
    validate_corpus()
    assert len(CASES) == 16
    assert sum(case.a72_outcome == "verification_fail" for case in CASES) == 12
    assert sum(case.a72_outcome == "verification_pass" for case in CASES) == 4

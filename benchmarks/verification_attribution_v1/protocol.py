"""Independent A73 verification/oracle classification."""

from __future__ import annotations

from enum import StrEnum


class Quadrant(StrEnum):
    V_PASS_O_PASS = "V_PASS_O_PASS"
    V_PASS_O_FAIL = "V_PASS_O_FAIL"
    V_FAIL_O_PASS = "V_FAIL_O_PASS"
    V_FAIL_O_FAIL = "V_FAIL_O_FAIL"


class VerificationFailure(StrEnum):
    CONFIGURE_FAILURE = "CONFIGURE_FAILURE"
    BUILD_FAILURE = "BUILD_FAILURE"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    LINK_FAILURE = "LINK_FAILURE"
    FOCUSED_TEST_FAILURE = "FOCUSED_TEST_FAILURE"
    UNRELATED_TEST_FAILURE = "UNRELATED_TEST_FAILURE"
    FULL_SUITE_REGRESSION = "FULL_SUITE_REGRESSION"
    PROCESS_TIMEOUT = "PROCESS_TIMEOUT"
    PROCESS_EXIT_FAILURE = "PROCESS_EXIT_FAILURE"
    SANDBOX_FAILURE = "SANDBOX_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    OTHER_VERIFICATION_FAILURE = "OTHER_VERIFICATION_FAILURE"


class SemanticFailure(StrEnum):
    WRONG_BEHAVIOR = "WRONG_BEHAVIOR"
    INCOMPLETE_IMPLEMENTATION = "INCOMPLETE_IMPLEMENTATION"
    PARTIAL_MULTI_FILE_CHANGE = "PARTIAL_MULTI_FILE_CHANGE"
    INCORRECT_EDGE_CASE = "INCORRECT_EDGE_CASE"
    BUILD_PREVENTS_ORACLE = "BUILD_PREVENTS_ORACLE"
    OTHER_SEMANTIC_FAILURE = "OTHER_SEMANTIC_FAILURE"


def quadrant(verification_pass: bool, oracle_pass: bool) -> Quadrant:
    return Quadrant(
        f"V_{'PASS' if verification_pass else 'FAIL'}_O_"
        f"{'PASS' if oracle_pass else 'FAIL'}"
    )


def verification_failure(
    evidence: dict[str, object] | None, *, oracle_pass: bool
) -> VerificationFailure:
    if not evidence:
        return VerificationFailure.OTHER_VERIFICATION_FAILURE
    if evidence.get("timed_out") is True:
        return VerificationFailure.PROCESS_TIMEOUT
    if evidence.get("strict_failure_class") not in {None, "none"}:
        return VerificationFailure.SANDBOX_FAILURE
    stage = evidence.get("verification_stage")
    if stage == "project.configure":
        return VerificationFailure.CONFIGURE_FAILURE
    if stage == "project.build":
        return VerificationFailure.BUILD_FAILURE
    if stage == "project.test":
        return (
            VerificationFailure.FULL_SUITE_REGRESSION
            if oracle_pass
            else VerificationFailure.FOCUSED_TEST_FAILURE
        )
    if evidence.get("process_outcome") in {"process_start_failed", "unavailable"}:
        return VerificationFailure.ENVIRONMENT_FAILURE
    if evidence.get("exit_code") not in {None, 0}:
        return VerificationFailure.PROCESS_EXIT_FAILURE
    return VerificationFailure.OTHER_VERIFICATION_FAILURE


def semantic_failure(operation_class: str, verification: str | None) -> SemanticFailure:
    if verification == VerificationFailure.BUILD_FAILURE.value:
        return SemanticFailure.BUILD_PREVENTS_ORACLE
    if operation_class in {"EDIT_MULTI", "MIXED_EDIT_CREATE"}:
        return SemanticFailure.PARTIAL_MULTI_FILE_CHANGE
    return SemanticFailure.WRONG_BEHAVIOR

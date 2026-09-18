"""A49 model-free acceptance-test synthesis qualification scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
)
from forge.evaluation.acceptance_test_synthesis import (
    ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION,
    ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION,
    ACCEPTANCE_TEST_SYNTHESIS_V1,
    AcceptanceQualification,
    AcceptanceTestGenerationResult,
    AcceptanceTestSynthesisRun,
    GeneratedAcceptanceTest,
    SynthesisCondition,
    acceptance_test_synthesis_to_dict,
    build_acceptance_test_request,
    classify_qualification,
    combine_verification_signals,
    contains_evaluator_leakage,
    execute_alternative_correct,
    qualify_acceptance_test,
    run_acceptance_test_synthesis_v1,
    validate_generated_acceptance_test,
)
from forge.models import MockModel


def _definition(task_id: str):  # type: ignore[no-untyped-def]
    return next(
        item
        for item in realistic_semantic_tasks((42,))
        if item.metadata.task_id == task_id
    )


def test_s01_qualified_python_candidate(tmp_path: Path) -> None:
    definition = _definition("R05")
    candidate = GeneratedAcceptanceTest(
        "failed_is_terminal",
        "from pyservice.state import JobState, can_transition\n"
        "assert not can_transition(JobState.FAILED, JobState.RUNNING)\n",
    )
    result = qualify_acceptance_test(
        tmp_path, REPOSITORY, definition.metadata, definition.wrong, candidate
    )
    assert (result.baseline, result.reference, result.wrong) == (
        "fail",
        "pass",
        "fail",
    )
    assert result.qualification is AcceptanceQualification.QUALIFIED


def test_s02_qualified_c_candidate(tmp_path: Path) -> None:
    definition = _definition("R07")
    candidate = GeneratedAcceptanceTest(
        "exclusive_window_limit",
        '#include "cengine/window.h"\n#include <assert.h>\n'
        "int main(void) { assert(!window_accepts(3, 3)); return 0; }\n",
    )
    result = qualify_acceptance_test(
        tmp_path, REPOSITORY, definition.metadata, definition.wrong, candidate
    )
    assert result.qualification is AcceptanceQualification.QUALIFIED


def test_s03_baseline_not_detected() -> None:
    assert classify_qualification("pass", "pass", "fail") is (
        AcceptanceQualification.BASELINE_NOT_DETECTED
    )


def test_s04_reference_rejected() -> None:
    assert classify_qualification("fail", "fail", "fail") is (
        AcceptanceQualification.REFERENCE_REJECTED
    )


def test_s05_wrong_accepted() -> None:
    assert classify_qualification("fail", "pass", "pass") is (
        AcceptanceQualification.WRONG_ACCEPTED
    )


def test_s06_python_syntax_invalid() -> None:
    result = validate_generated_acceptance_test(
        GeneratedAcceptanceTest("broken", "def test(:\n"), "Python"
    )
    assert result.safe and not result.structurally_valid


def test_s07_c_compile_invalid_is_structural(tmp_path: Path) -> None:
    definition = _definition("R07")
    candidate = GeneratedAcceptanceTest(
        "broken",
        '#include "cengine/window.h"\nint main(void) { this is not c; }\n',
    )
    result = qualify_acceptance_test(
        tmp_path, REPOSITORY, definition.metadata, definition.wrong, candidate
    )
    assert "structural_invalid" in {result.baseline, result.reference, result.wrong}


def test_s08_forbidden_shell_and_network_behavior_rejected() -> None:
    python = validate_generated_acceptance_test(
        GeneratedAcceptanceTest("unsafe", "import socket\n"), "Python"
    )
    c = validate_generated_acceptance_test(
        GeneratedAcceptanceTest(
            "unsafe", '#include "cengine/window.h"\nint main(void){system("x");}'
        ),
        "C17",
    )
    assert not python.safe and not c.safe


def test_s09_source_text_patch_matching_rejected() -> None:
    result = validate_generated_acceptance_test(
        GeneratedAcceptanceTest(
            "source_match",
            "from pathlib import Path\n"
            "assert 'expected patch' in Path('pyservice/state.py').read_text()\n",
        ),
        "Python",
    )
    assert not result.safe


def test_s10_evaluator_reference_leakage_rejected() -> None:
    assert contains_evaluator_leakage("read the hidden oracle")
    result = validate_generated_acceptance_test(
        GeneratedAcceptanceTest("leak", "# reference mutation\nassert True\n"),
        "Python",
    )
    assert not result.safe
    request = build_acceptance_test_request(
        _definition("R05").metadata, SynthesisCondition.C1_GROUNDED, REPOSITORY
    )
    assert not contains_evaluator_leakage(request.messages[0].content)


def test_s11_reference_repeat_mismatch_is_flaky() -> None:
    assert (
        classify_qualification("fail", "pass", "fail", reference_repeat="fail")
        is AcceptanceQualification.FLAKY
    )


def test_s12_generated_test_supplements_full_verification() -> None:
    assert combine_verification_signals(
        ("fail", "pass", "fail"), ("pass", "pass", "pass")
    ) == ("fail", "pass", "fail")
    assert combine_verification_signals(
        ("fail", "pass", "fail"), ("fail", "pass", "fail")
    ) == ("fail", "pass", "fail")


def test_s13_result_serialization_excludes_test_source() -> None:
    result = AcceptanceTestGenerationResult(
        "R05",
        "mock",
        SynthesisCondition.C1_GROUNDED,
        42,
        True,
        True,
        True,
        "fail",
        "pass",
        "pass",
        "fail",
        AcceptanceQualification.QUALIFIED,
        "a" * 64,
        100,
        10,
        10,
        0.1,
        0.2,
        "pass",
        "fail",
        "pass",
        "fail",
    )
    run = AcceptanceTestSynthesisRun(
        ACCEPTANCE_TEST_SYNTHESIS_V1,
        ACCEPTANCE_TEST_SYNTHESIS_SUITE_VERSION,
        ACCEPTANCE_TEST_SYNTHESIS_SCHEMA_VERSION,
        "repository",
        "mock",
        8192,
        42,
        512,
        (result,),
        True,
    )
    serialized = acceptance_test_synthesis_to_dict(run)
    assert "test_source" not in str(serialized)
    assert serialized["results"][0]["candidate_sha256"] == "a" * 64  # type: ignore[index]


def test_s14_production_verification_plan_is_unchanged() -> None:
    expected = ("project.configure", "project.build", "project.test")
    assert all(
        item.metadata.production_task.verification_plan.steps == expected
        for item in realistic_semantic_tasks((42,))
    )


def test_behavioral_candidates_accept_two_alternative_correct_implementations(
    tmp_path: Path,
) -> None:
    state = GeneratedAcceptanceTest(
        "terminal_failed",
        "from pyservice.state import JobState, can_transition\n"
        "assert not can_transition(JobState.FAILED, JobState.RUNNING)\n",
    )
    headers = GeneratedAcceptanceTest(
        "reject_underscore",
        "from pyservice.headers import normalize_header_name\n"
        "try:\n"
        "    normalize_header_name('x_header')\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('underscore accepted')\n",
    )
    assert (
        execute_alternative_correct(
            tmp_path, REPOSITORY, _definition("R05").metadata, state
        )
        == "pass"
    )
    assert (
        execute_alternative_correct(
            tmp_path, REPOSITORY, _definition("R06").metadata, headers
        )
        == "pass"
    )


def test_grounded_generation_sees_defective_current_source_not_reference(
    tmp_path: Path,
) -> None:
    response = json.dumps(
        {
            "test_name": "terminal_failed",
            "test_source": "from pyservice.state import JobState, can_transition\n"
            "assert not can_transition(JobState.FAILED, JobState.RUNNING)\n",
        }
    )
    model = MockModel((response, response), context_capacity=8192)
    run_acceptance_test_synthesis_v1(
        tmp_path,
        REPOSITORY,
        (_definition("R05"),),
        "mock",
        model,
        full_verification_signals={"R05": ("pass", "pass", "fail")},
    )
    prompt = model.requests[1].messages[0].content
    assert "JobState.FAILED: frozenset({JobState.RUNNING})," in prompt
    assert "JobState.FAILED: frozenset()," not in prompt

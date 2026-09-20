"""A55 P01-P14: mixed roles, parser, correction, and fixture integrity."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.models import MockModel, MutationRepresentationPolicy
from forge.orchestration.protocol import build_mutation_ready_output
from forge.orchestration.repository_session import (
    MIXED_READY_GUIDANCE,
    MUTATION_READY_SYSTEM_PROMPT,
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)
from scripts.mixed_operation_protocol_v1 import (
    CASES,
    Failure,
    classify_intent,
    classify_mixed,
    classify_operations,
    materialize,
    oracle,
    semantic_result,
)

CASE = CASES[0]


def _edit(*, path: str = CASE.edit_path) -> dict[str, object]:
    return {
        "type": "line_range_edit",
        "path": path,
        "start_line": 1,
        "end_line": 2,
        "new_text": CASE.reference_edit,
    }


def _create(*, path: str = CASE.create_path) -> dict[str, object]:
    return {"type": "create_file", "path": path, "content": CASE.reference_create}


def test_p01_mixed_intent_roles() -> None:
    assert (
        classify_intent(
            json.dumps(
                {
                    "modify_path": CASE.edit_path,
                    "modify_intent": "change caller",
                    "create_path": CASE.create_path,
                    "create_intent": "add implementation",
                }
            ),
            CASE,
        )
        is Failure.PASS
    )
    assert (
        classify_intent(
            json.dumps(
                {
                    "modify_path": CASE.create_path,
                    "modify_intent": "x",
                    "create_path": CASE.edit_path,
                    "create_intent": "y",
                }
            ),
            CASE,
        )
        is Failure.INTENT_ROLE_CONFUSION
    )


def test_p02_independent_children_validate_and_replay() -> None:
    assert classify_operations([_edit(), _create()], CASE) is Failure.PASS
    assert semantic_result([_edit(), _create()], CASE) is Failure.PASS


def test_p03_duplicate_create() -> None:
    assert (
        classify_operations([_create(), _create()], CASE) is Failure.DUPLICATE_OPERATION
    )


def test_p04_duplicate_edit() -> None:
    assert classify_operations([_edit(), _edit()], CASE) is Failure.DUPLICATE_OPERATION


def test_p05_missing_edit() -> None:
    assert classify_operations([_create()], CASE) is Failure.MISSING_OPERATION


def test_p06_missing_create() -> None:
    assert classify_operations([_edit()], CASE) is Failure.MISSING_OPERATION


def test_p07_wrong_operation_role() -> None:
    assert (
        classify_operations(
            [
                _create(path=CASE.edit_path),
                _edit(path=CASE.create_path),
            ],
            CASE,
        )
        is Failure.WRONG_OPERATION_TYPE
    )


def test_p08_unauthorized_path() -> None:
    assert (
        classify_operations(
            [
                _edit(path="surprise.py"),
                _create(),
            ],
            CASE,
        )
        is Failure.UNAUTHORIZED_PATH
    )


def test_p09_minimal_schema() -> None:
    operations = [
        {
            "op": "modify",
            "path": CASE.edit_path,
            "start_line": 1,
            "end_line": 2,
            "new_text": CASE.reference_edit,
        },
        {"op": "create", "path": CASE.create_path, "content": CASE.reference_create},
    ]
    assert (
        classify_mixed(json.dumps({"operations": operations}), CASE, minimal=True)
        is Failure.PASS
    )


def test_p10_exact_production_schema_parser() -> None:
    spec = build_mutation_ready_output(
        (CASE.edit_path,),
        create_paths=(CASE.create_path,),
        representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    assert spec.schema is not None
    assert (
        classify_mixed(
            json.dumps(
                {
                    "type": "multi_file_change",
                    "operations": [_edit(), _create()],
                }
            ),
            CASE,
        )
        is Failure.PASS
    )


def test_p11_mixed_correction_explicit_authority() -> None:
    guidance = MIXED_READY_GUIDANCE.lower()
    assert "exactly" in guidance
    assert "edit" in guidance and "create" in guidance
    assert "duplicate" in guidance
    bound = RepositoryChatSession._mixed_ready_guidance(
        SimpleNamespace(
            _mixed_edit_paths=(CASE.edit_path,),
            _create_paths=(CASE.create_path,),
            _mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
        )
    )
    assert "exactly 2 operations" in bound
    assert f"MODIFY paths: {CASE.edit_path}" in bound
    assert f"CREATE paths: {CASE.create_path}" in bound
    assert "exactly once" in bound
    assert "line_range_edit for MODIFY and create_file for CREATE" in bound


def test_p12_second_invalid_response_is_terminal(tmp_path: Path) -> None:
    (tmp_path / CASE.edit_path).write_text(CASE.source)
    invalid = json.dumps(
        {
            "type": "multi_file_change",
            "operations": [
                _create(),
                _create(),
                _create(),
                _create(),
            ],
        }
    )
    assert classify_mixed(invalid, CASE) is Failure.DUPLICATE_OPERATION
    model = MockModel((invalid, invalid))
    session = RepositoryChatSession(
        "a55-p12",
        model,
        tmp_path,
        registry=create_assist_repository_registry(include_creation=True),
        policy=create_assist_repository_policy(),
        required_candidate_paths=(CASE.edit_path,),
        create_candidate_paths=(CASE.create_path,),
        mixed_file_operations=True,
        require_relevant_source=False,
    )
    with pytest.raises(
        RepositoryOrchestrationError, match="invalid mixed group repeated"
    ):
        session.execute_task(CASE.task)
    assert len(model.requests) == 2
    first_prompt = "\n".join(message.content for message in model.requests[0].messages)
    assert f"MODIFY {CASE.edit_path}" in first_prompt
    assert f"CREATE {CASE.create_path}" in first_prompt
    assert (tmp_path / CASE.edit_path).read_text() == CASE.source
    assert not (tmp_path / CASE.create_path).exists()


def test_p13_structural_vs_semantic() -> None:
    wrong = _edit()
    wrong["new_text"] = "def total(price):\n    return 0\n"
    assert classify_operations([wrong, _create()], CASE) is Failure.PASS
    assert semantic_result([wrong, _create()], CASE) is Failure.SEMANTIC_FAIL


def test_p14_prompt_has_no_single_operation_command() -> None:
    prompt = MUTATION_READY_SYSTEM_PROMPT.lower()
    assert "one coding mutation" in prompt
    assert "one action" not in prompt
    assert "create only" not in MIXED_READY_GUIDANCE.lower()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_corpus_requires_both_operations(case) -> None:  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory(prefix="forge-a55-integrity-") as name:
        workspace = Path(name)
        for edit, create, expected in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (True, True, True),
        ):
            materialize(case, workspace, edit=edit, create=create)
            if not create:
                (workspace / case.create_path).unlink(missing_ok=True)
            assert oracle(case, workspace) is expected

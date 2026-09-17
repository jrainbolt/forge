from __future__ import annotations

import json
from dataclasses import replace

from forge.evaluation import (
    PAIRED_SEMANTIC_PLANNING_V1,
    PAIRED_SEMANTIC_TASKS,
    PlanningCondition,
    PlanningFailure,
    SemanticEligibility,
    aggregate_paired_results,
    audit_oracle_integrity,
    build_paired_planning_run,
    classify_foundation_e08,
    paired_planning_to_dict,
    plan_is_current,
    run_planning_condition,
    validate_plan_response,
)
from forge.models import MockModel


def _p06_plan(*, path: str = "validation.py") -> str:
    return json.dumps(
        {
            "files": [
                {
                    "path": path,
                    "behavior_change": "Make the minimum boundary inclusive with >=.",
                    "preserve": ["Preserve the maximum of 100."],
                    "test_intent": "Implementation accepts one at the minimum.",
                },
                {
                    "path": "test_validation.py",
                    "behavior_change": "Add an assertion for size one.",
                    "preserve": ["Keep the existing upper-bound assertions."],
                    "test_intent": "Test the exact minimum boundary of one.",
                },
            ]
        }
    )


def _p06_edit() -> str:
    return json.dumps(
        {
            "type": "multi_file_line_range_edit",
            "edits": [
                {
                    "path": "validation.py",
                    "start_line": 2,
                    "end_line": 2,
                    "new_text": "    return size >= 1 and size <= 100\n",
                },
                {
                    "path": "test_validation.py",
                    "start_line": 3,
                    "end_line": 3,
                    "new_text": (
                        "assert valid_batch_size(1)\nassert valid_batch_size(2)\n"
                    ),
                },
            ],
        }
    )


def test_corpus_has_six_paired_tasks_and_two_languages() -> None:
    assert [task.task_id for task in PAIRED_SEMANTIC_TASKS] == [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
    ]
    assert {task.language for task in PAIRED_SEMANTIC_TASKS} == {"C17", "Python"}
    assert all(len(task.paths) == 2 for task in PAIRED_SEMANTIC_TASKS)


def test_every_oracle_rejects_baseline_accepts_reference_and_rejects_wrong() -> None:
    audits = tuple(audit_oracle_integrity(task) for task in PAIRED_SEMANTIC_TASKS)
    assert all(not audit.baseline_oracle_pass for audit in audits)
    assert all(audit.reference_oracle_pass for audit in audits)
    assert all(not audit.wrong_mutation_oracle_pass for audit in audits)
    assert all(audit.semantic_oracle_discriminating for audit in audits)
    assert all(audit.semantic_score_eligible for audit in audits)


def test_plan_schema_and_concepts_bind_current_sources() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    plan, failure = validate_plan_response(_p06_plan(), task, workspace_generation=7)
    assert failure is None
    assert plan is not None
    assert plan.required_paths == task.paths
    assert plan_is_current(plan, task, workspace_generation=7)
    assert not plan_is_current(plan, task, workspace_generation=8)
    changed = replace(task, implementation_source=task.implementation_source + "\n")
    assert not plan_is_current(plan, changed, workspace_generation=7)


def test_plan_rejects_unexpected_duplicate_and_missing_paths() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    _, unexpected = validate_plan_response(_p06_plan(path="outside.py"), task)
    assert unexpected is PlanningFailure.PLAN_PATH_SET_INVALID

    payload = json.loads(_p06_plan())
    payload["files"][1]["path"] = "validation.py"
    _, duplicate = validate_plan_response(json.dumps(payload), task)
    assert duplicate is PlanningFailure.PLAN_PATH_SET_INVALID

    payload["files"] = payload["files"][:1]
    _, missing = validate_plan_response(json.dumps(payload), task)
    assert missing is PlanningFailure.PLAN_PATH_SET_INVALID


def test_plan_rejects_schema_bounds_missing_concept_and_contradiction() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    payload = json.loads(_p06_plan())
    payload["files"][0]["behavior_change"] = "x" * 241
    _, oversized = validate_plan_response(json.dumps(payload), task)
    assert oversized is PlanningFailure.PLAN_SCHEMA_INVALID

    payload = json.loads(_p06_plan())
    payload["files"][0]["behavior_change"] = "Change validation."
    payload["files"][0]["test_intent"] = "Implementation behavior."
    payload["files"][0]["preserve"] = []
    _, missing = validate_plan_response(json.dumps(payload), task)
    assert missing is PlanningFailure.PLAN_CONCEPT_MISSING

    payload = json.loads(_p06_plan())
    payload["files"][0]["preserve"].append("Keep > 1 and exclude one.")
    _, contradiction = validate_plan_response(json.dumps(payload), task)
    assert contradiction is PlanningFailure.PLAN_CONTRADICTORY


def test_d0_and_d1_use_same_production_edit_schema_and_d1_adds_plan() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    direct_model = MockModel((_p06_edit(),), context_capacity=8192)
    direct = run_planning_condition(
        "mock", direct_model, task, PlanningCondition.D0_DIRECT_GROUPED_EDIT
    )
    planned_model = MockModel((_p06_plan(), _p06_edit()), context_capacity=8192)
    planned = run_planning_condition(
        "mock", planned_model, task, PlanningCondition.D1_PLAN_THEN_GROUPED_EDIT
    )

    assert direct.group_valid and direct.build_test == "PASS"
    assert direct.oracle == "PASS" and direct.semantic_success
    assert direct.plan_valid is None and direct.model_calls == 1
    assert planned.plan_valid and planned.plan_file_count == 2
    assert planned.group_valid and planned.semantic_success and planned.model_calls == 2
    direct_schema = direct_model.requests[0].output.schema
    planned_schema = planned_model.requests[1].output.schema
    assert direct_schema == planned_schema
    assert direct_model.requests[0].generation.max_tokens == 512
    assert planned_model.requests[0].generation.max_tokens == 256
    assert planned_model.requests[1].generation.max_tokens == 512


def test_invalid_plan_stops_before_edit_and_grants_no_authority() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    model = MockModel((_p06_plan(path="outside.py"),), context_capacity=8192)
    result = run_planning_condition(
        "mock", model, task, PlanningCondition.D1_PLAN_THEN_GROUPED_EDIT
    )
    assert not result.plan_valid
    assert result.failure == PlanningFailure.PLAN_PATH_SET_INVALID.value
    assert result.model_calls == 1
    assert len(model.requests) == 1


def test_prompts_exclude_reference_mutations_and_evaluator_metadata() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    model = MockModel((_p06_plan(), _p06_edit()), context_capacity=8192)
    run_planning_condition(
        "mock", model, task, PlanningCondition.D1_PLAN_THEN_GROUPED_EDIT
    )
    plan_prompt = model.requests[0].messages[0].content
    edit_prompt = model.requests[1].messages[0].content
    assert "chain-of-thought" not in plan_prompt.lower()
    assert "internal deliberation" in plan_prompt
    assert "source_hashes" not in plan_prompt + edit_prompt
    assert "reference_implementation" not in plan_prompt + edit_prompt
    assert "semantic_oracle" not in plan_prompt + edit_prompt
    assert task.reference_implementation not in plan_prompt
    assert "outside.py" not in edit_prompt


def test_non_discriminating_oracle_is_not_semantic_success() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    weak = replace(
        task,
        implementation_source=task.reference_implementation,
        test_source=task.reference_test,
    )
    audit = audit_oracle_integrity(weak)
    assert audit.baseline_oracle_pass
    assert not audit.semantic_oracle_discriminating
    result = run_planning_condition(
        "mock",
        MockModel((_p06_edit(),), context_capacity=8192),
        weak,
        PlanningCondition.D0_DIRECT_GROUPED_EDIT,
    )
    assert not result.semantic_success
    assert result.failure == PlanningFailure.ORACLE_NON_DISCRIMINATING.value


def test_aggregation_and_serialization_separate_structure_and_semantics() -> None:
    task = PAIRED_SEMANTIC_TASKS[5]
    result = run_planning_condition(
        "mock",
        MockModel((_p06_edit(),), context_capacity=8192),
        task,
        PlanningCondition.D0_DIRECT_GROUPED_EDIT,
    )
    aggregate = aggregate_paired_results("mock", "D0", (result,))
    assert aggregate.structural_success == 1
    assert aggregate.semantic_success == 1
    run = build_paired_planning_run((result,))
    payload = paired_planning_to_dict(run)
    assert payload["suite"] == PAIRED_SEMANTIC_PLANNING_V1
    assert payload["integrity"][0]["baseline_oracle_pass"] is False
    assert payload["results"][0]["build_test"] == "PASS"
    assert "implementation_source" not in json.dumps(payload)


def test_legacy_e08_is_explicitly_under_specified_not_silently_versioned() -> None:
    audit = classify_foundation_e08(unchanged_baseline_oracle_pass=True)
    assert audit.task_id == "E08"
    assert audit.legacy_definition_preserved
    assert audit.unchanged_baseline_oracle_pass
    assert audit.semantic_eligibility == (
        SemanticEligibility.SEMANTICALLY_UNDER_SPECIFIED.value
    )
    assert not audit.versioned_replacement_created

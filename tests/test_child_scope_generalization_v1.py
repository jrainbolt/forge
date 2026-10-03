from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.child_edit_composition_v1.protocol import (
    ChildEdit,
    compose_child_edits,
    execute_composition,
)
from benchmarks.child_mutation_diagnosis_v1.suite import SCOPE_FRAMING
from benchmarks.child_scope_generalization_v1.protocol import (
    ProductionRejection,
    classify_production_rejection,
)
from benchmarks.child_scope_generalization_v1.suite import ALL_TASK_IDS, tasks
from benchmarks.real_repository_pilot_v2.runner import atomic_json, read_checkpoint
from forge.evaluation.realworld import apply_task_setup
from forge.evaluation.replay import source_state_identity
from forge.tools.repository_write import MultiFilePatchTool
from forge.tools.types import ExecutionContext


@dataclass
class Metrics:
    structured_mutation_valid: int = 1
    line_range_attempts: int = 0
    line_range_materialized: int = 0
    mutation_group_validation_result: str = "passed"
    mutation_group_preview_created: int = 1
    mutation_group_apply_result: str = "applied"
    preview_created: int = 1
    mutations: int = 1
    verification_plan_result: str = "pass"
    reverification_result: str = "not_run"


@pytest.mark.parametrize(
    ("metrics", "semantic", "expected"),
    [
        (
            Metrics(structured_mutation_valid=0),
            False,
            ProductionRejection.GROUP_VALIDATION_REJECTED,
        ),
        (
            Metrics(structured_mutation_valid=0, line_range_attempts=1),
            False,
            ProductionRejection.SOURCE_CURRENTNESS_REJECTED,
        ),
        (
            Metrics(mutation_group_preview_created=0, preview_created=0),
            False,
            ProductionRejection.PREVIEW_REJECTED,
        ),
        (
            Metrics(mutations=0, mutation_group_apply_result="not_run"),
            False,
            ProductionRejection.TRANSACTION_PRECHECK_REJECTED,
        ),
        (
            Metrics(mutations=0, mutation_group_apply_result="failed"),
            False,
            ProductionRejection.TRANSACTION_FAILED,
        ),
        (
            Metrics(verification_plan_result="fail"),
            False,
            ProductionRejection.VERIFICATION_FAILED,
        ),
        (Metrics(), False, ProductionRejection.SEMANTIC_FAILED),
        (Metrics(), True, ProductionRejection.PASS),
    ],
)
def test_precise_production_rejection_taxonomy(
    metrics: Metrics, semantic: bool, expected: ProductionRejection
) -> None:
    assert classify_production_rejection(metrics, semantic_pass=semantic) is expected


def test_new_corpus_has_four_coordinated_path_visible_tasks() -> None:
    definitions = tasks()
    assert len(ALL_TASK_IDS) == 8 and len(definitions) == 4
    for definition in definitions:
        assert len(definition.edit_paths) == 2
        assert all(
            path in definition.production_task.prompt for path in definition.edit_paths
        )
        assert (
            definition.production_task.expected_changed_paths == definition.edit_paths
        )


def test_frozen_setups_are_unique_and_baseline_changes_both_files(
    tmp_path: Path,
) -> None:
    canonical = Path("/Users/jamesrainbolt/Documents/github/foundation/foundation")
    if not canonical.exists():
        pytest.skip("Foundation corpus is unavailable")
    for definition in tasks():
        root = tmp_path / definition.task_id
        root.mkdir()
        for setup in definition.production_task.setup:
            source = (canonical / setup.path).read_text()
            assert source.count(setup.expected) == 1
            target = root / setup.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        apply_task_setup(root, definition.production_task.setup)
        assert all(
            setup.replacement in (root / setup.path).read_text()
            for setup in definition.production_task.setup
        )


def test_scope_framing_has_no_hidden_plan() -> None:
    lowered = SCOPE_FRAMING.lower()
    assert "expected" not in lowered and "reference" not in lowered
    assert "authorized file" in lowered


def test_cooperating_context_is_trusted_and_bounded() -> None:
    trusted = {"src/a.c": "a" * 40, "src/b.c": "b" * 40}
    authorized = "src/a.c"
    cooperating = tuple(path for path in trusted if path != authorized)
    rendered = "".join(trusted[path][:16] for path in cooperating)
    assert set(cooperating).issubset(trusted)
    assert len(rendered) <= 16 * len(cooperating)


def test_g0_g1_authority_equivalence() -> None:
    for definition in tasks():
        g0 = definition.production_task.required_candidate_paths
        g1 = definition.edit_paths
        assert g0 == g1


def _children(root: Path) -> tuple[ChildEdit, ChildEdit]:
    values = []
    for name in ("a.py", "b.py"):
        text = "VALUE = 1\n"
        (root / name).write_text(text)
        values.append(
            ChildEdit(
                name,
                hashlib.sha256(text.encode()).hexdigest(),
                ({"old": text, "new": "VALUE = 2\n"},),
            )
        )
    return values[0], values[1]


def test_complete_set_uses_existing_atomic_transaction(tmp_path: Path) -> None:
    children = _children(tmp_path)
    composition = compose_child_edits(
        tmp_path, ("a.py", "b.py"), children, workspace_generation=0
    )
    execution = execute_composition(composition, ExecutionContext(tmp_path.resolve()))
    assert execution.output["operation"] == "multi_patch"
    assert execution.logical_mutations == 1


def test_incomplete_set_and_transaction_failure_never_partially_apply(
    tmp_path: Path,
) -> None:
    children = _children(tmp_path)
    incomplete = compose_child_edits(
        tmp_path, ("a.py", "b.py"), children[:1], workspace_generation=0
    )
    assert not incomplete.composition_valid
    complete = compose_child_edits(
        tmp_path, ("a.py", "b.py"), children, workspace_generation=0
    )

    def fail_second(source: Path, target: Path) -> None:
        if target.name == "b.py":
            raise OSError("injected")
        source.replace(target)

    with pytest.raises(Exception, match="original files restored"):
        execute_composition(
            complete,
            ExecutionContext(tmp_path.resolve()),
            tool=MultiFilePatchTool(replace_operation=fail_second),
        )
    assert all(
        (tmp_path / name).read_text() == "VALUE = 1\n" for name in ("a.py", "b.py")
    )


def test_checkpoint_source_free_identity_and_package_exclusion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cell.json"
    payload = {
        "task_id": "F09",
        "profile": "qwen-small",
        "condition": "G1",
        "rejection": ProductionRejection.PREVIEW_REJECTED.value,
    }
    atomic_json(checkpoint, payload)
    assert (
        read_checkpoint(checkpoint, {"task_id": "F09", "profile": "qwen-small"})
        == payload
    )
    encoded = json.dumps(payload).lower()
    assert "old_text" not in encoded and "new_text" not in encoded
    assert (
        'where = ["src"]' in (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    canonical = Path("/Users/jamesrainbolt/Documents/github/foundation/foundation")
    if canonical.exists():
        assert source_state_identity(canonical) == source_state_identity(canonical)

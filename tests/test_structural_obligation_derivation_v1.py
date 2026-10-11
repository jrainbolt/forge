from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.orchestration.coding_task import CodingTaskState
from forge.structural_completeness import StructuralCompletenessDecision
from forge.structural_obligations import (
    StructuralObligation,
    StructuralObligationKind,
    derive_structural_obligations,
    evaluate_obligations,
)


def _obligation(
    kind: StructuralObligationKind,
    *,
    source: str = "src/component.py",
    target: str | None = None,
    symbol: str | None = "required_hook",
    provenance_class: str = "TRUSTED_PROJECT_CONFIGURATION",
) -> StructuralObligation:
    return StructuralObligation(
        obligation_id=f"check-{kind.value.lower()}",
        kind=kind,
        source_path=source,
        target_path=target,
        symbol=symbol,
        caller=None,
        inspection_scope=tuple(path for path in (source, target) if path),
        provenance_identity="a" * 64,
        provenance_class=provenance_class,
    )


def test_derivation_is_pre_mutation_source_free_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/component.py"
    source.parent.mkdir()
    source.write_text("def required_hook():\n    return 1\n", encoding="utf-8")
    task = "Update src/component.py while preserving required_hook. Secret replacement."

    first = derive_structural_obligations(
        task, tmp_path, mutation_authority=("src/component.py",)
    )
    second = derive_structural_obligations(
        task, tmp_path, mutation_authority=("src/component.py",)
    )

    assert first.identity == second.identity
    assert first.frozen_before_mutation
    assert {item.kind for item in first.obligations} == {
        StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
        StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
    }
    serialized = json.dumps(first.source_free_record(), sort_keys=True)
    assert "def required_hook" not in serialized
    assert "Secret replacement" not in serialized
    assert "src/component.py" not in serialized

    state = CodingTaskState(0)
    state.structural_obligations_frozen(first.source_free_record())
    assert state.structural_obligations["obligation_set_identity"] == first.identity


def test_semantic_only_ambiguous_and_no_obligation_controls(tmp_path: Path) -> None:
    (tmp_path / "feature.py").write_text("value = 1\n", encoding="utf-8")
    for prompt in (
        "Make the behavior friendlier.",
        "Improve this feature where appropriate.",
        "Explain the implementation without changing it.",
    ):
        result = derive_structural_obligations(
            prompt, tmp_path, mutation_authority=("feature.py",)
        )
        assert result.obligations == ()


def test_derivation_does_not_broaden_mutation_authority(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("def KeepMe():\n    pass\n", encoding="utf-8")
    (tmp_path / "src/b.py").write_text("def Other():\n    pass\n", encoding="utf-8")
    authority = ("src/a.py",)
    result = derive_structural_obligations(
        "Preserve KeepMe in src/a.py and inspect src/b.py.",
        tmp_path,
        mutation_authority=authority,
    )
    assert authority == ("src/a.py",)
    assert all(item.source_path == "src/a.py" for item in result.obligations)


def test_untrusted_registered_provenance_is_rejected(tmp_path: Path) -> None:
    rejected = tuple(
        _obligation(
            StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE, provenance_class=kind
        )
        for kind in ("EVALUATOR_INJECTION", "ORACLE", "REFERENCE_ARTIFACT")
    )
    result = derive_structural_obligations(
        "behavior only", tmp_path, mutation_authority=(), registered=rejected
    )
    assert result.obligations == ()


@pytest.mark.parametrize("kind", list(StructuralObligationKind))
def test_all_six_registered_kinds_use_the_same_runtime_evaluator(
    tmp_path: Path, kind: StructuralObligationKind
) -> None:
    source = tmp_path / "src/component.py"
    source.parent.mkdir()
    source.write_text(
        "from src.target import required_hook\n"
        "def required_hook():\n    return 1\n"
        "def caller():\n    return required_hook()\n",
        encoding="utf-8",
    )
    target = tmp_path / "src/target.py"
    target.write_text("def required_hook():\n    return 1\n", encoding="utf-8")
    obligation = _obligation(
        kind,
        target=(
            "src/target.py"
            if kind
            in {
                StructuralObligationKind.IMPORT_INCLUDE_RELATION,
                StructuralObligationKind.DECLARATION_IMPLEMENTATION_RELATION,
            }
            else None
        ),
    )
    derived = derive_structural_obligations(
        "behavior only", tmp_path, mutation_authority=(), registered=(obligation,)
    )
    assert derived.obligations == (obligation,)
    evaluation = evaluate_obligations(derived, tmp_path)
    if kind is StructuralObligationKind.DECLARATION_IMPLEMENTATION_RELATION:
        assert evaluation.decision is StructuralCompletenessDecision.FAIL
    else:
        assert evaluation.decision is StructuralCompletenessDecision.PASS


def test_missing_required_symbol_fails_closed_after_mutation(tmp_path: Path) -> None:
    source = tmp_path / "src/component.py"
    source.parent.mkdir()
    source.write_text("def RequiredHook():\n    return 1\n", encoding="utf-8")
    derived = derive_structural_obligations(
        "Update src/component.py but retain RequiredHook.",
        tmp_path,
        mutation_authority=("src/component.py",),
    )
    source.write_text("value = 1\n", encoding="utf-8")
    evaluation = evaluate_obligations(derived, tmp_path)
    assert evaluation.decision is StructuralCompletenessDecision.FAIL
    assert evaluation.failed_check_ids

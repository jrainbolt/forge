from __future__ import annotations

from pathlib import Path

from forge.orchestration.coding_task import CodingTaskState
from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessMode,
    StructuralCompletenessRecord,
    evaluate_structural_completeness,
)
from forge.structural_obligations import (
    StructuralObligation,
    StructuralObligationKind,
    derive_structural_obligations,
    evaluate_obligations,
)


def test_source_free_telemetry_exposes_policy_fields(tmp_path: Path) -> None:
    source = tmp_path / "component.py"
    source.write_text("def RequiredHook():\n    return True\n", encoding="utf-8")
    derived = derive_structural_obligations(
        "Preserve RequiredHook in component.py.",
        tmp_path,
        mutation_authority=("component.py",),
    )
    record = derived.source_free_record()
    assert record["obligation_count"] == 2
    assert record["derivation_source_classes"] == (
        "TRUSTED_TASK_AND_REPOSITORY_EVIDENCE",
    )
    assert all(
        item["obligation_bridge_identity"]
        for item in record["obligations"]  # type: ignore[union-attr]
    )
    assert "def RequiredHook" not in repr(record)


def test_caller_registration_uses_same_checker_and_provenance_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "component.py"
    source.write_text("def required_hook():\n    return True\n", encoding="utf-8")
    registered = StructuralObligation(
        "required-hook",
        StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
        "component.py",
        None,
        "required_hook",
        None,
        ("component.py",),
        "trusted-registration",
        "TRUSTED_PROJECT_CONFIGURATION",
    )
    accepted = derive_structural_obligations(
        "behavioral change",
        tmp_path,
        mutation_authority=("component.py",),
        registered=(registered,),
    )
    assert accepted.obligations == (registered,)
    assert evaluate_obligations(accepted, tmp_path).decision is (
        StructuralCompletenessDecision.PASS
    )
    rejected = StructuralObligation(
        "oracle-hook",
        StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
        "component.py",
        None,
        "required_hook",
        None,
        ("component.py",),
        "oracle-registration",
        "ORACLE",
    )
    assert not derive_structural_obligations(
        "behavioral change",
        tmp_path,
        mutation_authority=("component.py",),
        registered=(rejected,),
    ).obligations


def test_c04_and_h12_boundary_controls_remain_not_applicable(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg/state.py").write_text(
        "class JobState:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests/test_state.py").write_text(
        "from pkg.state import JobState\n", encoding="utf-8"
    )
    incidental = derive_structural_obligations(
        "Correct terminal behavior in pkg/state.py and tests/test_state.py.",
        tmp_path,
        mutation_authority=("pkg/state.py", "tests/test_state.py"),
    )
    ambiguous = derive_structural_obligations(
        "Create pkg/config.py with strict boolean and timeout parsing.",
        tmp_path,
        mutation_authority=("pkg/config.py",),
    )
    assert not incidental.obligations and not ambiguous.obligations
    assert evaluate_obligations(ambiguous, tmp_path).decision is (
        StructuralCompletenessDecision.NOT_APPLICABLE
    )


def test_relative_include_requires_explicit_contract_and_is_not_semantic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "csrc/component.c"
    source.parent.mkdir()
    (tmp_path / "csrc/component.h").write_text(
        "int required_component(void);\n", encoding="utf-8"
    )
    source.write_text(
        '#include "component.h"\nint required_component(void) { return 0; }\n',
        encoding="utf-8",
    )
    derived = derive_structural_obligations(
        "Retain required_component while updating csrc/component.c.",
        tmp_path,
        mutation_authority=("csrc/component.c",),
    )
    assert evaluate_obligations(derived, tmp_path).decision is (
        StructuralCompletenessDecision.PASS
    )
    # Returning zero may be behaviorally wrong; structural PASS does not decide that.
    assert len(derived.obligations) == 3


def test_enforced_rejection_footer_is_bounded_and_source_free() -> None:
    state = CodingTaskState(0)
    state.structural_obligations_frozen(
        {
            "obligation_count": 1,
            "obligations": (
                {
                    "kind": "REQUIRED_SYMBOL_PRESENCE",
                    "symbol_identity": "a" * 64,
                },
            ),
        }
    )
    state.structural_completeness_finished(
        StructuralCompletenessRecord(
            mode="enforce",
            executed=True,
            decision="fail",
            failed_check_ids=("check-1",),
            enforced_rejection=True,
        )
    )
    footer = state.finish("rejected").footer
    assert "Structural rejection: proven failure (1 checks)" in footer
    assert "REQUIRED_SYMBOL_PRESENCE" in footer
    assert "aaaaaaaaaaaa" in footer
    assert "generated source" not in footer


def test_mode_policy_remains_fail_open(tmp_path: Path) -> None:
    for mode in (StructuralCompletenessMode.SHADOW, StructuralCompletenessMode.ENFORCE):
        partial = evaluate_structural_completeness(
            mode,
            tmp_path,
            lambda _root: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        assert partial.decision == "partial"
        assert not partial.enforced_rejection
    off = evaluate_structural_completeness(
        StructuralCompletenessMode.OFF, tmp_path, None
    )
    assert not off.executed

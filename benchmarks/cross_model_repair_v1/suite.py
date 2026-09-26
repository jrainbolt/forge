"""Frozen A57 corpus and profile-pair definitions for A58."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.repair_framing_v1.suite import Case

SUITE = "cross-model-repair-v1"
VERSION = 1
REPAIR_PROFILES = ("qwen-small", "qwen-large", "codestral-22b")
ELIGIBLE_CASE_IDS = ("K01", "K02", "K04", "K05", "K06", "K07")


@dataclass(frozen=True, slots=True)
class Pair:
    case: Case
    repair_profile: str

    @property
    def same_model(self) -> bool:
        return self.case.profile == self.repair_profile

    @property
    def key(self) -> str:
        return f"{self.case.case_id}-{self.repair_profile}"


def pairs(cases: tuple[Case, ...]) -> tuple[Pair, ...]:
    eligible = tuple(case for case in cases if case.case_id in ELIGIBLE_CASE_IDS)
    if {case.case_id for case in eligible} != set(ELIGIBLE_CASE_IDS):
        raise ValueError("frozen A58 case corpus is incomplete")
    return tuple(
        Pair(case, profile) for case in eligible for profile in REPAIR_PROFILES
    )


def classify_case(recovered_profiles: frozenset[str], primary_profile: str) -> str:
    if len(recovered_profiles) > 1:
        return "ANY_MODEL_RECOVERY"
    if recovered_profiles == {primary_profile}:
        return "SAME_MODEL_ONLY"
    if recovered_profiles:
        return "CROSS_MODEL_RECOVERY"
    return "NO_MODEL_RECOVERY"

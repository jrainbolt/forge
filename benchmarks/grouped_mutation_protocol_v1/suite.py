"""Frozen source/authority invariants for A65."""

from __future__ import annotations

from dataclasses import dataclass

SUITE = "grouped-mutation-protocol-v1"
VERSION = 1
SEED = 42
MULTI_TASK_IDS = ("F05", "F06", "F07", "F08")
SINGLE_CONTROL_IDS = ("F01", "F02")
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")


@dataclass(frozen=True, slots=True)
class LayerInvariant:
    task: str
    sources: tuple[str, ...]
    authority: tuple[str, ...]
    representation: str


def validate_layers(p0: LayerInvariant, p1: LayerInvariant, p2: LayerInvariant) -> None:
    if not (p0.task == p1.task == p2.task):
        raise ValueError("P0/P1/P2 task mismatch")
    if not (p0.sources == p1.sources == p2.sources):
        raise ValueError("P0/P1/P2 source mismatch")
    if not (p0.authority == p1.authority == p2.authority):
        raise ValueError("P0/P1/P2 authority mismatch")
    if not (p0.representation == p1.representation == p2.representation):
        raise ValueError("P0/P1/P2 representation mismatch")

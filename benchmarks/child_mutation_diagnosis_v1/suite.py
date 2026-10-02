"""Frozen A67 diagnostic inputs and condition invariants."""

from __future__ import annotations

from dataclasses import dataclass

SUITE = "child-mutation-scope-materialization-v1"
VERSION = 1
SEED = 42
TEMPERATURE = 0.0
CONTEXT_TOKENS = 8192
OUTPUT_TOKENS = 512
TASK_IDS = ("F05", "F06", "F07", "F08")
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")
SCOPE_FRAMING = (
    "You are responsible only for the authorized file in this call. Produce the "
    "portion of the requested change that belongs in this file. Do not modify or "
    "describe edits to other files."
)


@dataclass(frozen=True, slots=True)
class ChildCondition:
    task: str
    authorized_path: str
    child_source_sha256: str
    authority: tuple[str, ...]
    representation: str
    cooperating_paths: tuple[str, ...] = ()


def validate_conditions(
    d0: ChildCondition,
    d1: ChildCondition,
    d2: ChildCondition,
    trusted_paths: tuple[str, ...],
) -> None:
    if not (d0.task == d1.task == d2.task):
        raise ValueError("condition task mismatch")
    if not (d0.authorized_path == d1.authorized_path == d2.authorized_path):
        raise ValueError("condition child mismatch")
    if not (d0.child_source_sha256 == d1.child_source_sha256 == d2.child_source_sha256):
        raise ValueError("condition source mismatch")
    if not (d0.authority == d1.authority == d2.authority):
        raise ValueError("condition authority mismatch")
    if not (d0.representation == d1.representation == d2.representation):
        raise ValueError("representation switching is forbidden")
    if d0.cooperating_paths or d1.cooperating_paths:
        raise ValueError("only D2 may add cooperating context")
    if not set(d2.cooperating_paths).issubset(trusted_paths):
        raise ValueError("D2 context must already be trusted")

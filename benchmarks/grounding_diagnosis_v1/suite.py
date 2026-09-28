"""Frozen A63 diagnostic corpus derived from source-free A62 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

SUITE = "grounding-diagnosis-v1"
VERSION = 1
SEED = 42


@dataclass(frozen=True, slots=True)
class HistoricalCase:
    case_id: str
    profile: str
    task_id: str
    acquired_paths: tuple[str, ...]
    context_quality: str


GROUNDING_CASES = (
    HistoricalCase("G01", "qwen-small", "F03", ("src/power.c",), "MISDIRECTED"),
    HistoricalCase("G02", "qwen-large", "F04", ("src/simulation.c",), "MISDIRECTED"),
    HistoricalCase(
        "G03",
        "codestral-22b",
        "F05",
        ("src/logistics_endpoint.c",),
        "MISDIRECTED",
    ),
    HistoricalCase("G04", "qwen-small", "F06", ("src/simulation.c",), "MISDIRECTED"),
    HistoricalCase("G05", "qwen-small", "F01", ("src/clock.c",), "SUFFICIENT"),
    HistoricalCase(
        "G06",
        "qwen-large",
        "F02",
        ("src/power.c", "src/clock.c"),
        "SUFFICIENT",
    ),
    HistoricalCase(
        "G07",
        "codestral-22b",
        "F07",
        ("src/storage.c", "src/telemetry.c"),
        "SUFFICIENT",
    ),
    HistoricalCase(
        "G08",
        "qwen-large",
        "F08",
        ("src/presentation.c", "src/snapshot.c"),
        "SUFFICIENT",
    ),
)

CODING_CASES = (
    HistoricalCase("C01", "qwen-small", "F01", ("src/clock.c",), "SUFFICIENT"),
    HistoricalCase(
        "C02",
        "qwen-small",
        "F02",
        ("src/power.c", "src/clock.c"),
        "SUFFICIENT",
    ),
    HistoricalCase("C03", "qwen-large", "F01", ("src/clock.c",), "SUFFICIENT"),
    HistoricalCase(
        "C04",
        "qwen-large",
        "F07",
        ("src/storage.c", "src/telemetry.c"),
        "SUFFICIENT",
    ),
    HistoricalCase(
        "C05",
        "codestral-22b",
        "F02",
        ("src/power.c", "src/clock.c"),
        "SUFFICIENT",
    ),
    HistoricalCase(
        "C06",
        "codestral-22b",
        "F08",
        ("src/presentation.c", "src/snapshot.c"),
        "SUFFICIENT",
    ),
)

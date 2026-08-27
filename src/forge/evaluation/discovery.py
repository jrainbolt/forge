"""Deterministic multilingual evaluation for fast lexical discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge.lexical_index import RepositoryLexicalIndex
from forge.retrieval import SourceKind

DISCOVERY_V1 = "discovery-v1"
DISCOVERY_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class DiscoveryTaskResult:
    task_id: str
    completed: bool
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryEvaluationResult:
    tasks: tuple[DiscoveryTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_discovery_v1(root: Path) -> DiscoveryEvaluationResult:
    """Run D01-D08 against disposable C and Java repository fixtures."""
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    workspace.mkdir()
    _write_fixture(workspace)
    index = RepositoryLexicalIndex(workspace, cache_root=root / "cache")
    index.build()

    tasks = [
        _query(index, "D01", "clock sunset boundary", "src/clock.c", 3),
        _query(index, "D02", "clock public API", "include/clock.h", 3),
        _query(
            index,
            "D03",
            "retry policy backoff",
            "src/main/java/example/RetryPolicy.java",
            3,
        ),
        _query(
            index,
            "D04",
            "build configuration cmake",
            "CMakeLists.txt",
            1,
            preferred=SourceKind.CONFIGURATION,
        ),
        _query(
            index,
            "D05",
            "clock regression test",
            "tests/test_clock.c",
            1,
            preferred=SourceKind.TEST,
        ),
        _query(
            index,
            "D06",
            "user setup instructions",
            "README.md",
            1,
            preferred=SourceKind.DOCUMENTATION,
        ),
    ]

    unchanged = index.refresh()
    (workspace / "src/solar.c").write_text(
        "int solar_sunset_boundary(void) { return 22; }\n"
    )
    changed = index.refresh()
    (workspace / "src/new_timer.go").write_text("package timer\nfunc Tick() {}\n")
    added = index.refresh()
    (workspace / "src/new_timer.go").unlink()
    deleted = index.refresh()
    tasks.append(
        DiscoveryTaskResult(
            "D07",
            unchanged.files_retokenized == 0
            and changed.files_retokenized == 1
            and added.files_added == 1
            and deleted.files_deleted == 1,
            (),
        )
    )
    vendor = index.search("vendor clock shim", limit=8)
    generated = index.search("generated clock metadata", limit=8)
    tasks.append(
        DiscoveryTaskResult(
            "D08",
            any(item.source_kind is SourceKind.THIRD_PARTY for item in vendor)
            and all("build/" not in item.path for item in generated),
            tuple(item.path for item in vendor),
        )
    )
    frozen = tuple(tasks)
    return DiscoveryEvaluationResult(
        frozen, sum(task.completed for task in frozen), len(frozen)
    )


def _query(
    index: RepositoryLexicalIndex,
    task_id: str,
    query: str,
    expected: str,
    maximum_rank: int,
    *,
    preferred: SourceKind | None = None,
) -> DiscoveryTaskResult:
    matches = index.search(query, limit=8, preferred_source_kind=preferred)
    paths = tuple(item.path for item in matches)
    rank = paths.index(expected) + 1 if expected in paths else 10_000
    return DiscoveryTaskResult(task_id, rank <= maximum_rank, paths)


def _write_fixture(root: Path) -> None:
    files = {
        "src/clock.c": ("int clock_sunset_boundary(int hour) { return hour >= 18; }\n"),
        "src/solar.c": "int solar_noon(void) { return 12; }\n",
        "include/clock.h": (
            "/* Clock public API. */\nint clock_sunset_boundary(int hour);\n"
        ),
        "tests/test_clock.c": (
            "/* Clock regression test. */\nvoid test_clock_sunset_boundary(void) {}\n"
        ),
        "CMakeLists.txt": (
            "# Build configuration cmake\nadd_library(clock src/clock.c)\n"
        ),
        "README.md": "# User setup instructions\nInstall and configure the clock.\n",
        "src/main/java/example/RetryPolicy.java": (
            "class RetryPolicy { long exponentialBackoff(int retry) "
            "{ return retry; } }\n"
        ),
        "src/test/java/example/RetryPolicyTest.java": (
            "class RetryPolicyTest { void verifiesRetryLimit() {} }\n"
        ),
        "pom.xml": "<!-- Java build configuration -->\n<project></project>\n",
        "vendor/clock_shim.c": "int vendor_clock_shim(void) { return 0; }\n",
        "build/generated_clock.c": "int generated_clock_metadata(void) { return 0; }\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

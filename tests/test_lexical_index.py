import sqlite3
from pathlib import Path

import pytest

from forge.evaluation.discovery import run_discovery_v1
from forge.lexical_index import (
    MAX_LEXICAL_FILE_BYTES,
    RepositoryLexicalIndex,
    lexical_tokenize,
)
from forge.retrieval import SourceKind, classify_source, tokenize
from forge.tools import ExecutionContext, LexicalSearchTool


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def _index(tmp_path: Path) -> tuple[Path, RepositoryLexicalIndex]:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    return workspace, RepositoryLexicalIndex(workspace, cache_root=tmp_path / "cache")


def test_discovery_v1_passes_all_multilingual_cases(tmp_path: Path) -> None:
    result = run_discovery_v1(tmp_path / "discovery")
    assert result.tasks_passed == result.tasks_total == 8
    assert tuple(task.task_id for task in result.tasks) == tuple(
        f"D{number:02d}" for number in range(1, 9)
    )


def test_tokenization_and_source_classification_are_language_agnostic() -> None:
    assert tokenize("src/RetryPolicy.exponential_backoff") == {
        "src",
        "retry",
        "policy",
        "exponential",
        "backoff",
    }
    assert {"inserters", "inserter", "policies", "policy"}.issubset(
        lexical_tokenize("inserters policies")
    )
    assert classify_source("src/main.c") is SourceKind.IMPLEMENTATION
    assert classify_source("include/api.hpp") is SourceKind.IMPLEMENTATION
    assert classify_source("tests/main_test.go") is SourceKind.TEST
    assert classify_source("CMakeLists.txt") is SourceKind.CONFIGURATION
    assert classify_source("docs/setup.md") is SourceKind.DOCUMENTATION
    assert classify_source("vendor/library.c") is SourceKind.THIRD_PARTY


def test_ranking_uses_path_basename_content_rarity_and_preferences(
    tmp_path: Path,
) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace, "src/sunset_clock.c", "int sunset_clock_boundary(void) {}\n")
    _write(workspace, "src/noise.c", "int clock(void) {}\n")
    _write(workspace, "tests/test_sunset_clock.c", "void regression_clock_test() {}\n")
    _write(workspace, "README.md", "User setup instructions for sunset clock.\n")
    _write(workspace, "CMakeLists.txt", "# build configuration sunset clock\n")
    _write(workspace, "src/common.c", "int common_token(void) {}\n")
    _write(workspace, "src/common_two.c", "int common_token_two(void) {}\n")

    implementation = index.search("sunset clock boundary", limit=5)
    assert implementation[0].path == "src/sunset_clock.c"
    assert implementation[0].score.basename > 0
    assert implementation[0].score.content > 0
    assert implementation[0].score.rarity > 0
    assert implementation[0].matched_tokens == ("boundary", "clock", "sunset")
    assert (
        index.search("regression clock test", preferred_source_kind=SourceKind.TEST)[
            0
        ].path
        == "tests/test_sunset_clock.c"
    )
    assert (
        index.search(
            "user setup instructions", preferred_source_kind=SourceKind.DOCUMENTATION
        )[0].path
        == "README.md"
    )
    assert (
        index.search(
            "build configuration", preferred_source_kind=SourceKind.CONFIGURATION
        )[0].path
        == "CMakeLists.txt"
    )


def test_incremental_refresh_add_change_delete_and_noop(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    first = _write(workspace, "src/one.rs", "fn one() {}\n")
    _write(workspace, "src/two.go", "package two\n")
    built = index.build()
    assert built.files_added == 2 and built.files_retokenized == 2
    unchanged = index.refresh()
    assert unchanged.files_retokenized == 0 and unchanged.files_reused == 2
    first.write_text("fn one_changed() {}\n")
    changed = index.refresh()
    assert changed.files_changed == 1 and changed.files_retokenized == 1
    added = _write(workspace, "src/three.java", "class Three {}\n")
    assert index.refresh().files_added == 1
    added.unlink()
    deleted = index.refresh()
    assert deleted.files_deleted == 1
    assert all(match.path != "src/three.java" for match in index.search("Three"))


def test_skips_binary_oversized_generated_and_escaping_symlinks(
    tmp_path: Path,
) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace, "src/valid.c", "int bounded_token(void) {}\n")
    _write(workspace, "src/binary.c", b"valid\x00secret")
    _write(workspace, "src/huge.c", b"x" * (MAX_LEXICAL_FILE_BYTES + 1))
    _write(workspace, "build/generated.c", "int generated_secret(void) {}\n")
    outside = _write(tmp_path, "outside.c", "int outside_secret(void) {}\n")
    try:
        (workspace / "escape.c").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    index.build()
    assert tuple(match.path for match in index.search("bounded token")) == (
        "src/valid.c",
    )
    for query in ("secret", "generated", "outside"):
        assert index.search(query) == ()


def test_corrupt_and_incompatible_indexes_rebuild_safely(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace, "main.ts", "function repositoryClock() {}\n")
    index.database_path.parent.mkdir(parents=True)
    index.database_path.write_text("not sqlite")
    assert index.status()["state"] == "corrupt"
    assert index.search("repository clock")[0].path == "main.ts"
    with sqlite3.connect(index.database_path) as connection:
        connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
    assert index.status()["state"] == "incompatible"
    assert index.refresh().files_retokenized == 1
    assert index.status()["state"] == "ready"


def test_tool_output_is_bounded_discovery_metadata_not_source(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace, "src/private.c", "int private_clock_secret(void) {}\n")
    output = LexicalSearchTool(index).execute(
        {"query": "private clock", "limit": 1}, ExecutionContext(workspace)
    )
    assert output["evidence"] == "discovery_only"
    assert output["requires_source_read"] is True
    assert output["match_count"] == 1
    match = output["matches"][0]
    assert match["path"] == "src/private.c"
    assert match["line_start"] >= 1
    assert match["line_end"] - match["line_start"] < 120
    assert "source_text" not in match

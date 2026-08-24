from __future__ import annotations

import sqlite3
from pathlib import Path

from forge.repository_analysis import PythonAnalyzer
from forge.repository_index import (
    SCHEMA_VERSION,
    RepositoryIndex,
    RepositoryIndexError,
)
from forge.tools import AllowAllPolicy, ExecutionContext, ToolExecutor
from forge.tools.builtin import create_readonly_repository_registry
from forge.tools.types import ToolInvocation, ToolResultStatus


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _index(tmp_path: Path) -> tuple[Path, RepositoryIndex]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, RepositoryIndex(workspace, cache_root=tmp_path / "cache")


def test_build_persists_versioned_metadata_without_source(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    source = "def secret_symbol():\n    return 'SECRET_LITERAL'\n"
    _write(workspace / "module.py", source)

    metrics = index.build()

    assert metrics.files_added == 1
    assert metrics.symbols == 1
    assert index.status()["schema_version"] == SCHEMA_VERSION
    assert "SECRET_LITERAL" not in index.database_path.read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_refresh_reuses_unchanged_and_updates_changed_deleted_new(
    tmp_path: Path,
) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace / "same.py", "def same():\n    pass\n")
    _write(workspace / "changed.py", "def old():\n    pass\n")
    _write(workspace / "deleted.py", "def gone():\n    pass\n")
    index.build()

    _write(workspace / "changed.py", "def replacement():\n    pass\n")
    (workspace / "deleted.py").unlink()
    _write(workspace / "new.py", "def added():\n    pass\n")
    metrics = index.refresh()

    assert metrics.files_reused == 1
    assert metrics.files_changed == 1
    assert metrics.files_added == 1
    assert metrics.files_deleted == 1
    assert not index.find_symbols("old", ".")
    assert index.find_symbols("replacement", ".")[0]["path"] == "changed.py"


def test_external_edit_is_visible_on_next_query(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    path = workspace / "module.py"
    _write(path, "def before():\n    pass\n")
    index.find_symbols("before", ".")

    _write(path, "def after():\n    pass\n")

    assert not index.find_symbols("before", ".")
    assert index.find_symbols("after", ".")


def test_invalidation_forces_same_task_reparse(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    path = workspace / "module.py"
    _write(path, "def before():\n    pass\n")
    index.build()
    _write(path, "def after():\n    pass\n")

    index.invalidate("module.py")
    metrics = index.refresh()

    assert metrics.files_added == 1
    assert index.find_symbols("after", ".")


def test_symbol_reference_and_outline_tools_use_persistent_index(
    tmp_path: Path,
) -> None:
    workspace, index = _index(tmp_path)
    _write(
        workspace / "module.py",
        "class Service:\n"
        "    def target(self):\n"
        "        return 1\n\n"
        "def call():\n"
        "    return Service().target()\n",
    )
    executor = ToolExecutor(
        create_readonly_repository_registry(index), AllowAllPolicy()
    )
    context = ExecutionContext(workspace)

    symbol = executor.execute(
        ToolInvocation(
            "symbol", "repository.find_symbol", {"symbol": "Service.target"}
        ),
        context,
    )
    refs = executor.execute(
        ToolInvocation(
            "references",
            "repository.find_references",
            {"symbol": "Service.target"},
        ),
        context,
    )
    outline = executor.execute(
        ToolInvocation("outline", "repository.file_outline", {"path": "module.py"}),
        context,
    )

    assert symbol.status is ToolResultStatus.SUCCESS
    assert symbol.output["matches"][0]["qualified_name"] == "Service.target"
    assert refs.output["references"][0]["line_text"].strip().endswith(".target()")
    assert outline.output["symbols"][1]["qualified_name"] == "Service.target"


def test_corrupt_and_wrong_schema_indexes_rebuild(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    _write(workspace / "module.py", "def healthy():\n    pass\n")
    index.database_path.parent.mkdir(parents=True)
    index.database_path.write_bytes(b"not sqlite")

    assert index.find_symbols("healthy", ".")
    with sqlite3.connect(index.database_path) as connection:
        connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
        connection.commit()

    assert index.find_symbols("healthy", ".")
    assert index.status()["schema_version"] == SCHEMA_VERSION


class CountingAnalyzer(PythonAnalyzer):
    def __init__(self) -> None:
        self.outline_calls = 0

    def outline(self, source: str):  # type: ignore[no-untyped-def]
        self.outline_calls += 1
        return super().outline(source)


def test_warm_refresh_avoids_reparsing_unchanged_files(tmp_path: Path) -> None:
    workspace, _ = _index(tmp_path)
    _write(workspace / "one.py", "def one():\n    pass\n")
    _write(workspace / "two.py", "def two():\n    pass\n")
    analyzer = CountingAnalyzer()
    index = RepositoryIndex(workspace, cache_root=tmp_path / "cache", analyzer=analyzer)

    index.build()
    index.refresh()

    assert analyzer.outline_calls == 2
    assert index.last_metrics.files_reused == 2


def test_clear_removes_only_derived_database(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    source = workspace / "module.py"
    _write(source, "def retained():\n    pass\n")
    index.build()

    index.clear()

    assert source.exists()
    assert index.status()["state"] == "missing"


def test_index_excludes_git_metadata_and_external_symlinks(tmp_path: Path) -> None:
    workspace, index = _index(tmp_path)
    outside = tmp_path / "outside.py"
    _write(outside, "def escaped():\n    pass\n")
    _write(workspace / ".git" / "hidden.py", "def hidden():\n    pass\n")
    (workspace / "linked.py").symlink_to(outside)
    _write(workspace / "visible.py", "def visible():\n    pass\n")

    assert index.find_symbols("visible", ".")
    assert not index.find_symbols("hidden", ".")
    assert not index.find_symbols("escaped", ".")


class UnavailableIndex:
    def find_symbols(self, symbol: str, scope: str):  # type: ignore[no-untyped-def]
        raise RepositoryIndexError("cache unavailable")


def test_unavailable_index_falls_back_to_bounded_direct_scan(tmp_path: Path) -> None:
    workspace, _ = _index(tmp_path)
    _write(workspace / "module.py", "def fallback():\n    pass\n")
    executor = ToolExecutor(
        create_readonly_repository_registry(UnavailableIndex()),  # type: ignore[arg-type]
        AllowAllPolicy(),
    )

    result = executor.execute(
        ToolInvocation("fallback", "repository.find_symbol", {"symbol": "fallback"}),
        ExecutionContext(workspace),
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["matches"][0]["path"] == "module.py"

"""Syntax-only structural checks over disposable workspaces."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .suite import CheckKind, CompletenessCheck

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
NOT_CHECKABLE = "NOT_CHECKABLE"
AMBIGUOUS = "AMBIGUOUS"
ALLOWED_EXECUTABLES = frozenset({"/usr/bin/cc"})
COMPILER_TIMEOUT_SECONDS = 2


@dataclass(frozen=True, slots=True)
class EvaluationDetail:
    status: str
    operational_failure: str | None = None
    subprocess_exit: str = "NOT_INVOKED"


def _read(workspace: Path, relative: str) -> str | None:
    path = (workspace / relative).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _python_tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _python_symbol(tree: ast.Module, symbol: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == symbol
        for node in tree.body
    )


def _python_import(tree: ast.Module, target_path: str, symbol: str | None) -> bool:
    module = target_path.removesuffix(".py").replace("/", ".")
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and (symbol is None or any(item.name == symbol for item in node.names))
        ):
            return True
        if isinstance(node, ast.Import) and any(
            item.name == module for item in node.names
        ):
            return True
    return False


def _calls(tree: ast.Module, caller: str | None, symbol: str) -> bool:
    scope: ast.AST = tree
    if caller is not None:
        candidates = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == caller
        ]
        if len(candidates) != 1:
            return False
        scope = candidates[0]
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == symbol
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == symbol
        )
        for node in ast.walk(scope)
    )


def _c_symbol(source: str, symbol: str, *, definition: bool) -> bool:
    ending = r"\s*\{" if definition else r"\s*;"
    return bool(re.search(rf"\b{re.escape(symbol)}\s*\([^;{{}}]*\){ending}", source))


def _c_component_role(
    workspace: Path, source_path: str, *, executable: str = "/usr/bin/cc"
) -> EvaluationDetail:
    if executable not in ALLOWED_EXECUTABLES:
        return EvaluationDetail(NOT_CHECKABLE, "EXECUTABLE_NOT_ALLOWLISTED")
    source = (workspace / source_path).resolve()
    try:
        source.relative_to(workspace.resolve())
    except ValueError:
        return EvaluationDetail(NOT_CHECKABLE, "WORKSPACE_CONFINEMENT_FAILURE")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    try:
        completed = subprocess.run(
            (
                executable,
                "-std=c17",
                "-Wall",
                "-Werror",
                "-fsyntax-only",
                "-I",
                str(source.parent),
                str(source),
            ),
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=COMPILER_TIMEOUT_SECONDS,
            check=False,
            shell=False,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return EvaluationDetail(NOT_CHECKABLE, "COMPILER_TIMEOUT", "TIMEOUT")
    except OSError:
        return EvaluationDetail(NOT_CHECKABLE, "COMPILER_UNAVAILABLE")
    if completed.returncode == 0:
        return EvaluationDetail(COMPLETE, subprocess_exit="EXIT_ZERO")
    return EvaluationDetail(INCOMPLETE, subprocess_exit="EXIT_NONZERO")


def evaluate_detailed(check: CompletenessCheck, workspace: Path) -> EvaluationDetail:
    if any(
        path.startswith("/") or ".." in Path(path).parts
        for path in check.authorized_scope
    ):
        return EvaluationDetail(NOT_CHECKABLE, "WORKSPACE_CONFINEMENT_FAILURE")
    source = _read(workspace, check.source_path)
    if source is None:
        return EvaluationDetail(INCOMPLETE)
    target = _read(workspace, check.target_path) if check.target_path else None
    is_python = check.source_path.endswith(".py")
    is_c = check.source_path.endswith((".c", ".h"))
    if not is_python and not is_c:
        return EvaluationDetail(NOT_CHECKABLE, "UNSUPPORTED_LANGUAGE")
    tree = _python_tree(source) if is_python else None
    if is_python and tree is None:
        return EvaluationDetail(NOT_CHECKABLE, "PARSER_FAILURE")

    if check.kind is CheckKind.REQUIRED_SYMBOL_PRESENCE:
        if check.symbol is None:
            return EvaluationDetail(NOT_CHECKABLE, "CHECK_DEFINITION_INCOMPLETE")
        found = (
            _python_symbol(tree, check.symbol)
            if tree is not None
            else _c_symbol(source, check.symbol, definition=True)
        )
        return EvaluationDetail(COMPLETE if found else INCOMPLETE)

    if check.kind is CheckKind.REQUIRED_COMPONENT_ROLE:
        if check.symbol is None:
            return EvaluationDetail(NOT_CHECKABLE, "CHECK_DEFINITION_INCOMPLETE")
        if tree is not None:
            return EvaluationDetail(
                COMPLETE if _python_symbol(tree, check.symbol) else INCOMPLETE
            )
        if not _c_symbol(source, check.symbol, definition=True):
            return EvaluationDetail(INCOMPLETE)
        return _c_component_role(workspace, check.source_path)

    if check.kind is CheckKind.IMPORT_INCLUDE_RELATION:
        if check.target_path is None:
            return EvaluationDetail(NOT_CHECKABLE, "CHECK_DEFINITION_INCOMPLETE")
        if is_python:
            found = _python_import(tree, check.target_path, check.symbol)  # type: ignore[arg-type]
        else:
            found = bool(
                re.search(
                    rf'^\s*#\s*include\s*["<]{re.escape(Path(check.target_path).name)}[">]',
                    source,
                    re.MULTILINE,
                )
            )
        return EvaluationDetail(COMPLETE if found else INCOMPLETE)

    if check.kind in {
        CheckKind.CALLER_CALLEE_RELATION,
        CheckKind.REGISTRATION_OR_USAGE_RELATION,
    }:
        if check.symbol is None:
            return EvaluationDetail(NOT_CHECKABLE, "CHECK_DEFINITION_INCOMPLETE")
        if is_python:
            return EvaluationDetail(
                COMPLETE if _calls(tree, check.caller, check.symbol) else INCOMPLETE  # type: ignore[arg-type]
            )
        found = bool(re.search(rf"\b{re.escape(check.symbol)}\s*\(", source))
        return EvaluationDetail(COMPLETE if found else INCOMPLETE)

    if check.kind is CheckKind.DECLARATION_IMPLEMENTATION_RELATION:
        if check.symbol is None or target is None:
            return EvaluationDetail(
                NOT_CHECKABLE if check.symbol is None else INCOMPLETE,
                "CHECK_DEFINITION_INCOMPLETE" if check.symbol is None else None,
            )
        declaration = _c_symbol(source, check.symbol, definition=False)
        implementation = _c_symbol(target, check.symbol, definition=True)
        if not declaration or not implementation:
            return EvaluationDetail(INCOMPLETE)
        return EvaluationDetail(COMPLETE)

    return EvaluationDetail(AMBIGUOUS, "UNSUPPORTED_CHECK_KIND")


def evaluate(check: CompletenessCheck, workspace: Path) -> str:
    return evaluate_detailed(check, workspace).status

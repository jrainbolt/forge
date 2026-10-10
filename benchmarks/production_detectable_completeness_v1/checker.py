"""Syntax-only structural checks over disposable workspaces."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from .suite import CheckKind, CompletenessCheck

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
NOT_CHECKABLE = "NOT_CHECKABLE"
AMBIGUOUS = "AMBIGUOUS"


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


def _c_component_role(workspace: Path, source_path: str) -> bool:
    source = workspace / source_path
    try:
        completed = subprocess.run(
            (
                "/usr/bin/cc",
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
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def evaluate(check: CompletenessCheck, workspace: Path) -> str:
    if any(
        path.startswith("/") or ".." in Path(path).parts
        for path in check.authorized_scope
    ):
        return NOT_CHECKABLE
    source = _read(workspace, check.source_path)
    if source is None:
        return INCOMPLETE
    target = _read(workspace, check.target_path) if check.target_path else None
    is_python = check.source_path.endswith(".py")
    tree = _python_tree(source) if is_python else None
    if is_python and tree is None:
        return INCOMPLETE

    if check.kind is CheckKind.REQUIRED_SYMBOL_PRESENCE:
        if check.symbol is None:
            return NOT_CHECKABLE
        found = (
            _python_symbol(tree, check.symbol)
            if tree is not None
            else _c_symbol(source, check.symbol, definition=True)
        )
        return COMPLETE if found else INCOMPLETE

    if check.kind is CheckKind.REQUIRED_COMPONENT_ROLE:
        if check.symbol is None:
            return NOT_CHECKABLE
        found = (
            _python_symbol(tree, check.symbol)
            if tree is not None
            else _c_symbol(source, check.symbol, definition=True)
            and _c_component_role(workspace, check.source_path)
        )
        return COMPLETE if found else INCOMPLETE

    if check.kind is CheckKind.IMPORT_INCLUDE_RELATION:
        if check.target_path is None:
            return NOT_CHECKABLE
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
        return COMPLETE if found else INCOMPLETE

    if check.kind in {
        CheckKind.CALLER_CALLEE_RELATION,
        CheckKind.REGISTRATION_OR_USAGE_RELATION,
    }:
        if check.symbol is None:
            return NOT_CHECKABLE
        if is_python:
            return COMPLETE if _calls(tree, check.caller, check.symbol) else INCOMPLETE  # type: ignore[arg-type]
        found = bool(re.search(rf"\b{re.escape(check.symbol)}\s*\(", source))
        return COMPLETE if found else INCOMPLETE

    if check.kind is CheckKind.DECLARATION_IMPLEMENTATION_RELATION:
        if check.symbol is None or target is None:
            return NOT_CHECKABLE if check.symbol is None else INCOMPLETE
        declaration = _c_symbol(source, check.symbol, definition=False)
        implementation = _c_symbol(target, check.symbol, definition=True)
        if not declaration or not implementation:
            return INCOMPLETE
        return COMPLETE

    return AMBIGUOUS

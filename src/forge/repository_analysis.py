"""Bounded, on-demand repository structure analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum


class FileLanguage(Enum):
    PYTHON = "python"
    TEXT = "text"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class SymbolDefinition:
    kind: str
    name: str
    qualified_name: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    kind: str
    line: int
    column: int
    containing_symbol: str | None


class PythonParseError(ValueError):
    """Python source could not be parsed structurally."""


def classify_file(name: str) -> FileLanguage:
    if name.lower().endswith(".py"):
        return FileLanguage.PYTHON
    return FileLanguage.TEXT


class PythonAnalyzer:
    """Extract deterministic Python definitions and reference candidates."""

    def outline(self, source: str) -> tuple[SymbolDefinition, ...]:
        tree = _parse(source)
        definitions: list[SymbolDefinition] = []

        def visit(body: list[ast.stmt], parents: tuple[str, ...]) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = (*parents, node.name)
                    definitions.append(
                        SymbolDefinition(
                            _definition_kind(node, parents),
                            node.name,
                            ".".join(qualified),
                            node.lineno,
                            node.end_lineno or node.lineno,
                        )
                    )
                    visit(node.body, qualified)
                elif isinstance(node, ast.ClassDef):
                    qualified = (*parents, node.name)
                    definitions.append(
                        SymbolDefinition(
                            "class",
                            node.name,
                            ".".join(qualified),
                            node.lineno,
                            node.end_lineno or node.lineno,
                        )
                    )
                    visit(node.body, qualified)

        visit(tree.body, ())
        return tuple(definitions)

    def references(self, source: str, symbol: str) -> tuple[ReferenceCandidate, ...]:
        tree = _parse(source)
        visitor = _ReferenceVisitor(symbol)
        visitor.visit(tree)
        return tuple(visitor.results)


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as error:
        line = getattr(error, "lineno", None)
        detail = f" at line {line}" if line is not None else ""
        raise PythonParseError(f"Python parse failed{detail}") from error


def _definition_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef, parents: tuple[str, ...]
) -> str:
    if parents:
        return "async_method" if isinstance(node, ast.AsyncFunctionDef) else "method"
    return "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"


class _ReferenceVisitor(ast.NodeVisitor):
    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._simple = symbol.rsplit(".", 1)[-1]
        self._containers: list[str] = []
        self.results: list[ReferenceCandidate] = []

    @property
    def _container(self) -> str | None:
        return ".".join(self._containers) if self._containers else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._containers.append(node.name)
        for item in node.body:
            self.visit(item)
        self._containers.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._containers.append(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for item in node.body:
            self.visit(item)
        self._containers.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self._matches(node.func):
            self._add("call", node.func)
            for argument in (*node.args, *node.keywords):
                self.visit(
                    argument.value if isinstance(argument, ast.keyword) else argument
                )
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self._symbol == self._simple == node.id:
            self._add("name", node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._matches(node):
            self._add("attribute", node)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (
                alias.name.rsplit(".", 1)[-1] == self._simple
                or alias.asname == self._simple
            ):
                self._add("import", node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == self._simple or alias.asname == self._simple:
                self._add("import", node)

    def _matches(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return self._symbol == self._simple == node.id
        if isinstance(node, ast.Attribute):
            dotted = _attribute_name(node)
            return dotted == self._symbol or node.attr == self._simple
        return False

    def _add(self, kind: str, node: ast.AST) -> None:
        self.results.append(
            ReferenceCandidate(
                kind,
                node.lineno,
                node.col_offset,
                self._container,
            )
        )


def _attribute_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))

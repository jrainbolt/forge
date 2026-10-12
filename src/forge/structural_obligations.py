"""Trusted, pre-mutation structural-obligation derivation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessEvaluation,
)


class StructuralObligationKind(StrEnum):
    DECLARATION_IMPLEMENTATION_RELATION = "DECLARATION_IMPLEMENTATION_RELATION"
    CALLER_CALLEE_RELATION = "CALLER_CALLEE_RELATION"
    REGISTRATION_OR_USAGE_RELATION = "REGISTRATION_OR_USAGE_RELATION"
    IMPORT_INCLUDE_RELATION = "IMPORT_INCLUDE_RELATION"
    REQUIRED_SYMBOL_PRESENCE = "REQUIRED_SYMBOL_PRESENCE"
    REQUIRED_COMPONENT_ROLE = "REQUIRED_COMPONENT_ROLE"


@dataclass(frozen=True, slots=True)
class StructuralObligation:
    obligation_id: str
    kind: StructuralObligationKind
    source_path: str
    target_path: str | None
    symbol: str | None
    caller: str | None
    inspection_scope: tuple[str, ...]
    provenance_identity: str
    provenance_class: str = "TRUSTED_TASK_AND_REPOSITORY_EVIDENCE"

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "id": self.obligation_id,
                    "kind": self.kind.value,
                    "source": _hash(self.source_path),
                    "target": _hash(self.target_path),
                    "symbol": _hash(self.symbol),
                    "caller": _hash(self.caller),
                    "scope": [_hash(path) for path in self.inspection_scope],
                    "provenance": self.provenance_identity,
                    "class": self.provenance_class,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class StructuralObligationSet:
    obligations: tuple[StructuralObligation, ...] = ()
    frozen_before_mutation: bool = True
    derivation_latency_seconds: float = 0.0
    source_operations: int = 0

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            "\0".join(item.identity for item in self.obligations).encode()
        ).hexdigest()

    def source_free_record(self) -> dict[str, object]:
        return {
            "obligation_set_identity": self.identity,
            "obligation_count": len(self.obligations),
            "obligations": tuple(
                {
                    "obligation_id": item.obligation_id,
                    "identity": item.identity,
                    "kind": item.kind.value,
                    "source_identity": _hash(item.source_path),
                    "target_identity": _hash(item.target_path),
                    "symbol_identity": _hash(item.symbol),
                    "caller_identity": _hash(item.caller),
                    "provenance_identity": item.provenance_identity,
                    "provenance_class": item.provenance_class,
                    "inspection_scope_identity": hashlib.sha256(
                        "\0".join(item.inspection_scope).encode()
                    ).hexdigest(),
                }
                for item in self.obligations
            ),
            "frozen_before_mutation": self.frozen_before_mutation,
            "derivation_latency_seconds": self.derivation_latency_seconds,
            "source_operations": self.source_operations,
        }


_PATH = re.compile(r"\b[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|c|h)\b")
_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_C_DECL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*;")
_C_DEF = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{")
_STOP = frozenset(
    {
        "Fix",
        "Create",
        "Update",
        "Repair",
        "The",
        "Both",
        "Do",
        "Modify",
        "Verify",
        "Configure",
        "ValueError",
        "None",
        "Python",
        "C17",
    }
)


def derive_structural_obligations(
    task_text: str,
    workspace: Path,
    *,
    mutation_authority: tuple[str, ...],
    registered: tuple[StructuralObligation, ...] = (),
) -> StructuralObligationSet:
    """Derive only relationships supported by task text and trusted source."""
    started = time.perf_counter()
    root = workspace.resolve()
    authority = tuple(sorted(set(mutation_authority)))
    explicit_paths = tuple(sorted(set(_PATH.findall(task_text))))
    task_symbols = {
        token
        for token in _SYMBOL.findall(task_text)
        if token not in _STOP and ("_" in token or token[:1].isupper())
    }
    structural_symbols = {
        match.group(1)
        for match in re.finditer(
            r"(?:preserv(?:e|ing)|retain|define|provide|create|implement(?:ing)?|with)"
            r"(?:\s+(?:a|an|the|function|class|symbol|module))*\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)",
            task_text,
            re.IGNORECASE,
        )
        if match.group(1) not in _STOP
    }
    sources: dict[str, str] = {}
    for relative in sorted(set((*authority, *explicit_paths))):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            try:
                sources[relative] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
    for relative in authority:
        if Path(relative).suffix != ".c":
            continue
        header = str(Path(relative).with_suffix(".h"))
        if header in sources:
            continue
        path = (root / header).resolve()
        if path.is_file():
            with suppress(OSError, UnicodeError):
                sources[header] = path.read_text(encoding="utf-8")
    provenance = hashlib.sha256(
        json.dumps(
            {
                "task": _hash(task_text),
                "authority": [_hash(path) for path in authority],
                "sources": {
                    path: hashlib.sha256(text.encode()).hexdigest()
                    for path, text in sorted(sources.items())
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    obligations: list[StructuralObligation] = []

    def add(
        kind: StructuralObligationKind,
        source: str,
        target: str | None = None,
        symbol: str | None = None,
        caller: str | None = None,
    ) -> None:
        key = f"{kind.value}:{source}:{target or ''}:{symbol or ''}:{caller or ''}"
        obligations.append(
            StructuralObligation(
                hashlib.sha256(key.encode()).hexdigest(),
                kind,
                source,
                target,
                symbol,
                caller,
                tuple(sorted({source, *(tuple((target,)) if target else ())})),
                provenance,
            )
        )

    for relative in authority:
        suffix = Path(relative).suffix
        source = sources.get(relative)
        if suffix == ".py" and source is not None:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            definitions = {
                node.name
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            for symbol in sorted(definitions & structural_symbols):
                add(
                    StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
                    relative,
                    symbol=symbol,
                )
                add(
                    StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
                    relative,
                    symbol=symbol,
                )
        elif suffix == ".c":
            header = str(Path(relative).with_suffix(".h"))
            header_text = sources.get(header)
            declared = set(_C_DECL.findall(header_text or ""))
            relevant = declared & task_symbols
            if not relevant and relative in explicit_paths and len(declared) == 1:
                relevant = declared
            if relative not in explicit_paths and not relevant:
                continue
            if header_text is not None and relevant:
                add(StructuralObligationKind.IMPORT_INCLUDE_RELATION, relative, header)
            for symbol in sorted(relevant):
                add(
                    StructuralObligationKind.DECLARATION_IMPLEMENTATION_RELATION,
                    header,
                    relative,
                    symbol,
                )
                add(
                    StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
                    relative,
                    header,
                    symbol,
                )

    python_definitions: dict[str, set[str]] = {}
    for path, source in sources.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        python_definitions[path] = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    for source_path, source in sorted(sources.items()):
        if source_path not in authority or not source_path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            target = f"{node.module.replace('.', '/')}.py"
            if target not in authority or target not in python_definitions:
                continue
            for imported in node.names:
                symbol = imported.name
                if symbol not in python_definitions[target]:
                    continue
                add(
                    StructuralObligationKind.IMPORT_INCLUDE_RELATION,
                    source_path,
                    target,
                    symbol,
                )
                if source_path.startswith("tests/"):
                    add(
                        StructuralObligationKind.REGISTRATION_OR_USAGE_RELATION,
                        source_path,
                        target,
                        symbol,
                    )

    c_implementations: dict[str, tuple[str, str]] = {}
    for target in authority:
        if not target.endswith(".c"):
            continue
        header = str(Path(target).with_suffix(".h"))
        declarations = set(_C_DECL.findall(sources.get(header, "")))
        for symbol in declarations:
            c_implementations[symbol] = (target, header)
    for source_path, source in sorted(sources.items()):
        if source_path not in authority or not source_path.endswith(".c"):
            continue
        including = (root / source_path).resolve()
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', source, re.MULTILINE):
            included = (including.parent / include).resolve()
            try:
                included_path = str(included.relative_to(root))
            except ValueError:
                continue
            for symbol, (target, header) in c_implementations.items():
                if included_path != header or target == source_path:
                    continue
                add(
                    StructuralObligationKind.IMPORT_INCLUDE_RELATION,
                    source_path,
                    header,
                )
                if re.search(rf"\b{re.escape(symbol)}\s*\(", source):
                    add(
                        StructuralObligationKind.CALLER_CALLEE_RELATION,
                        source_path,
                        target,
                        symbol,
                    )
    for target in authority:
        if Path(target).suffix != ".py" or target in sources:
            continue
        module = target.removesuffix(".py").replace("/", ".")
        symbols = sorted(structural_symbols & task_symbols)
        symbol = symbols[0] if len(symbols) == 1 else None
        for source_path, source in sorted(sources.items()):
            if not source_path.endswith(".py"):
                continue
            if symbol and re.search(
                rf"\b(?:from|import)\s+{re.escape(module)}\b", source
            ):
                add(
                    StructuralObligationKind.IMPORT_INCLUDE_RELATION,
                    source_path,
                    target,
                    symbol,
                )
                if re.search(rf"\b{re.escape(symbol)}\s*\(", source):
                    add(
                        StructuralObligationKind.CALLER_CALLEE_RELATION,
                        source_path,
                        target,
                        symbol,
                    )
        if symbol:
            add(
                StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
                target,
                symbol=symbol,
            )
            if symbol[:1].isupper():
                add(
                    StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
                    target,
                    symbol=symbol,
                )

    for item in registered:
        if item.provenance_class.startswith(("EVALUATOR", "ORACLE", "REFERENCE")):
            continue
        if not item.provenance_identity:
            continue
        obligations.append(item)
    unique = {item.identity: item for item in obligations}
    ordered = tuple(sorted(unique.values(), key=lambda item: item.identity))
    return StructuralObligationSet(
        ordered,
        True,
        time.perf_counter() - started,
        len(sources),
    )


def evaluate_obligations(
    obligations: StructuralObligationSet, workspace: Path
) -> StructuralCompletenessEvaluation:
    if not obligations.obligations:
        return StructuralCompletenessEvaluation(
            StructuralCompletenessDecision.NOT_APPLICABLE
        )
    failed = []
    operational = []
    started = time.perf_counter()
    for obligation in obligations.obligations:
        source = _read(workspace, obligation.source_path)
        target = (
            _read(workspace, obligation.target_path) if obligation.target_path else None
        )
        if source is None:
            failed.append(obligation.obligation_id)
            continue
        try:
            complete = _evaluate_one(obligation, source, target, workspace)
        except (SyntaxError, ValueError):
            complete = False
        except Exception:
            operational.append("CHECKER_INTERNAL_ERROR")
            continue
        if not complete:
            failed.append(obligation.obligation_id)
    decision = (
        StructuralCompletenessDecision.FAIL
        if failed
        else StructuralCompletenessDecision.PARTIAL
        if operational
        else StructuralCompletenessDecision.PASS
    )
    return StructuralCompletenessEvaluation(
        decision,
        tuple(failed) if decision is StructuralCompletenessDecision.FAIL else (),
        tuple(operational),
        time.perf_counter() - started,
    )


def _evaluate_one(
    obligation: StructuralObligation,
    source: str,
    target: str | None,
    workspace: Path,
) -> bool:
    symbol = obligation.symbol
    if obligation.kind in {
        StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
        StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
    }:
        if symbol is None:
            raise ValueError("missing symbol")
        if obligation.source_path.endswith(".py"):
            tree = ast.parse(source)
            return any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol
                for node in tree.body
            )
        return symbol in set(_C_DEF.findall(source))
    if obligation.kind is StructuralObligationKind.DECLARATION_IMPLEMENTATION_RELATION:
        return bool(
            symbol
            and symbol in _C_DECL.findall(source)
            and target
            and symbol in _C_DEF.findall(target)
        )
    if obligation.kind is StructuralObligationKind.IMPORT_INCLUDE_RELATION:
        if obligation.source_path.endswith(".py"):
            module = (
                (obligation.target_path or "").removesuffix(".py").replace("/", ".")
            )
            tree = ast.parse(source)
            return any(
                isinstance(node, ast.ImportFrom)
                and node.module == module
                or isinstance(node, ast.Import)
                and any(item.name == module for item in node.names)
                for node in tree.body
            )
        root = workspace.resolve()
        including = (root / obligation.source_path).resolve()
        expected = (root / (obligation.target_path or "")).resolve()
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', source, re.MULTILINE):
            resolved = (including.parent / include).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if resolved == expected and resolved.is_file():
                return True
        return False
    if symbol is None:
        raise ValueError("missing relationship symbol")
    if obligation.source_path.endswith(".py"):
        return any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == symbol
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == symbol
            )
            for node in ast.walk(ast.parse(source))
        )
    return bool(re.search(rf"\b{re.escape(symbol)}\s*\(", source))


def _read(workspace: Path, relative: str | None) -> str | None:
    if relative is None:
        return None
    root = workspace.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
        return path.read_text(encoding="utf-8")
    except (ValueError, OSError, UnicodeError):
        return None


def _hash(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()

"""Fresh, source-visible structural-obligation precision controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from forge.structural_obligations import StructuralObligationKind

RUN_ID = "a88-obligation-derivation-precision-correction-v2-authoritative"


@dataclass(frozen=True, slots=True)
class PrecisionCase:
    case_id: str
    operation_class: str
    language: str
    task: str
    authority: tuple[str, ...]
    initial_files: tuple[tuple[str, str], ...]
    materialized_files: tuple[tuple[str, str], ...]
    truth: tuple[tuple[StructuralObligationKind, str, str | None, str | None], ...]
    role: str


def cases() -> tuple[PrecisionCase, ...]:
    values: list[PrecisionCase] = []
    for index in range(1, 9):
        number = index + 100
        symbol = f"Provider{number}"
        path = f"pkg/provider_{number}.py"
        values.append(
            PrecisionCase(
                f"Q{index:02d}",
                "CREATE",
                "Python",
                f"Create {path} with a {symbol} class.",
                (path,),
                (),
                ((path, f"class {symbol}:\n    pass\n"),),
                (
                    (
                        StructuralObligationKind.REQUIRED_SYMBOL_PRESENCE,
                        path,
                        None,
                        symbol,
                    ),
                    (
                        StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
                        path,
                        None,
                        symbol,
                    ),
                ),
                "EXPLICIT_COMPONENT",
            )
        )
    for index in range(9, 15):
        number = index + 100
        symbol = f"component_{number}"
        source = f"csrc/component_{number}.c"
        header = f"csrc/component_{number}.h"
        values.append(
            PrecisionCase(
                f"Q{index:02d}",
                "EDIT_SINGLE",
                "C",
                f"Update {source} while retaining {symbol} declared in its header.",
                (source,),
                (
                    (header, f"int {symbol}(int value);\n"),
                    (
                        source,
                        f'#include "component_{number}.h"\n'
                        f"int {symbol}(int value) {{ return value; }}\n",
                    ),
                ),
                (),
                (
                    (
                        StructuralObligationKind.DECLARATION_IMPLEMENTATION_RELATION,
                        header,
                        source,
                        symbol,
                    ),
                    (
                        StructuralObligationKind.IMPORT_INCLUDE_RELATION,
                        source,
                        header,
                        None,
                    ),
                    (
                        StructuralObligationKind.REQUIRED_COMPONENT_ROLE,
                        source,
                        header,
                        symbol,
                    ),
                ),
                "EXPLICIT_DECLARATION_CONTRACT",
            )
        )
    for index in range(15, 19):
        number = index + 100
        module = f"pkg/state_{number}.py"
        test = f"tests/test_state_{number}.py"
        symbol = f"state_{number}"
        values.append(
            PrecisionCase(
                f"Q{index:02d}",
                "EDIT_MULTI",
                "Python",
                f"Improve behavior in {module} and update {test}; verify both.",
                (module, test),
                (
                    (module, f"def {symbol}():\n    return True\n"),
                    (
                        test,
                        f"from pkg.state_{number} import {symbol}\n"
                        f"def test_value():\n    assert {symbol}()\n",
                    ),
                ),
                (),
                (),
                "INCIDENTAL_TEST_USAGE",
            )
        )
    for index in range(19, 23):
        number = index + 100
        source = f"csrc/unit_{number}.c"
        header = f"csrc/unit_{number}.h"
        test = f"csrc/tests/test_unit_{number}.c"
        symbol = f"unit_{number}"
        values.append(
            PrecisionCase(
                f"Q{index:02d}",
                "EDIT_MULTI",
                "C",
                f"Correct boundary behavior in {source} and {test}.",
                (source, test),
                (
                    (header, f"int {symbol}(void);\n"),
                    (
                        source,
                        f'#include "unit_{number}.h"\n'
                        f"int {symbol}(void) {{ return 1; }}\n",
                    ),
                    (
                        test,
                        f'#include "../unit_{number}.h"\n'
                        f"int main(void) {{ return {symbol}(); }}\n",
                    ),
                ),
                (),
                (),
                "INCIDENTAL_INCLUDE_CALL",
            )
        )
    for index in range(23, 25):
        number = index + 100
        registry = f"pkg/registry_{number}.py"
        provider = f"pkg/provider_{number}.py"
        symbol = f"provider_{number}"
        values.append(
            PrecisionCase(
                f"Q{index:02d}",
                "MIXED_EDIT_CREATE",
                "Python",
                f"Improve {registry} and create {provider} for the new behavior.",
                (registry, provider),
                ((registry, f"REGISTRY = {{'default': {symbol}}}\n"),),
                ((provider, f"def {symbol}():\n    return True\n"),),
                (
                    (
                        StructuralObligationKind.REGISTRATION_OR_USAGE_RELATION,
                        registry,
                        provider,
                        symbol,
                    ),
                ),
                "AMBIGUOUS_REGISTRATION_REQUIREMENT",
            )
        )
    return tuple(values)


CASES = cases()


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            [
                {
                    "id": case.case_id,
                    "task": hashlib.sha256(case.task.encode()).hexdigest(),
                    "truth": [
                        tuple(str(value) for value in item) for item in case.truth
                    ],
                }
                for case in CASES
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()

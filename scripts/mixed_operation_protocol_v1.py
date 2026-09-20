"""A55 evaluator-only corpus and strict mixed-operation role classification.

This module does not grant mutation authority.  Its fixtures live only in
disposable evaluator workspaces and are not packaged as runtime resources.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from forge.orchestration import (
    LineRangeEditProposal,
    MutationCandidate,
    ToolCallOutcome,
    parse_model_output,
    validate_line_range_edit,
)
from forge.orchestration.protocol import ProtocolError
from forge.tools.controlled_creation import authorize_create_candidate
from forge.tools.mixed_transaction import (
    MixedFileTransactionTool,
    mixed_group_id,
    preview_mixed_file_transaction,
)
from forge.tools.tool import ToolError
from forge.tools.types import ExecutionContext

SUITE = "mixed-operation-protocol-v1"


class Failure(StrEnum):
    INTENT_ROLE_CONFUSION = "INTENT_ROLE_CONFUSION"
    INTENT_PATH_CONFUSION = "INTENT_PATH_CONFUSION"
    EDIT_CHILD_INVALID = "EDIT_CHILD_INVALID"
    CREATE_CHILD_INVALID = "CREATE_CHILD_INVALID"
    MIXED_SCHEMA_INVALID = "MIXED_SCHEMA_INVALID"
    DUPLICATE_OPERATION = "DUPLICATE_OPERATION"
    MISSING_OPERATION = "MISSING_OPERATION"
    WRONG_OPERATION_TYPE = "WRONG_OPERATION_TYPE"
    UNAUTHORIZED_PATH = "UNAUTHORIZED_PATH"
    NO_OP_EDIT = "NO_OP_EDIT"
    SEMANTIC_FAIL = "SEMANTIC_FAIL"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class MixedCase:
    case_id: str
    language: str
    task: str
    edit_path: str
    create_path: str
    source: str
    reference_edit: str
    reference_create: str
    check: str


CASES = (
    MixedCase(
        "X01",
        "Python",
        "Change bill.py total(price) to apply the tax in a new tax.py module. "
        "Create tax.py with apply_tax(price), returning price plus 10 percent tax.",
        "bill.py",
        "tax.py",
        "def total(price):\n    return price\n",
        "from tax import apply_tax\n\ndef total(price):\n    return apply_tax(price)\n",
        "def apply_tax(price):\n    return price * 1.1\n",
        "from bill import total; assert total(20) == 22",
    ),
    MixedCase(
        "X02",
        "Python",
        "Change registry.py so make() constructs a Worker from new worker.py. "
        "Create worker.py defining Worker with run() returning 'ready'.",
        "registry.py",
        "worker.py",
        "def make():\n    return None\n",
        "from worker import Worker\n\ndef make():\n    return Worker()\n",
        "class Worker:\n    def run(self):\n        return 'ready'\n",
        "from registry import make; assert make().run() == 'ready'",
    ),
    MixedCase(
        "X03",
        "Python",
        "Change parser.py parse(code) to use the STATUS mapping in new statuses.py. "
        "Create statuses.py with STATUS mapping 200 to 'ok' and 404 to 'missing'. "
        "Unknown codes must return 'unknown'.",
        "parser.py",
        "statuses.py",
        "def parse(code):\n    return 'unknown'\n",
        "from statuses import STATUS\n\ndef parse(code):\n"
        "    return STATUS.get(code, 'unknown')\n",
        "STATUS = {200: 'ok', 404: 'missing'}\n",
        "from parser import parse; assert [parse(x) for x in (200,404,500)] "
        "== ['ok','missing','unknown']",
    ),
    MixedCase(
        "X04",
        "Python",
        "Change pkg/__init__.py to export encode from new pkg/codec.py. "
        "Create codec.py with encode(text) returning the reversed text.",
        "pkg/__init__.py",
        "pkg/codec.py",
        "def encode(text):\n    return text\n",
        "from .codec import encode\n",
        "def encode(text):\n    return text[::-1]\n",
        "from pkg import encode; from pkg.codec import encode as direct; "
        "assert encode('abc') == direct('abc') == 'cba'",
    ),
    MixedCase(
        "X05",
        "C",
        "Change dispatcher.c compute(int) to delegate to clamp10(int) in a new "
        "helper.c. Create helper.c defining clamp10: clamp integers to [0,10].",
        "dispatcher.c",
        "helper.c",
        "int compute(int value) { return value; }\n",
        "int clamp10(int value);\nint compute(int value) { return clamp10(value); }\n",
        "int clamp10(int value) { return value < 0 ? 0 : value > 10 ? 10 : value; }\n",
        "int compute(int); int main(void) { return compute(-2) != 0 || "
        "compute(7) != 7 || compute(15) != 10; }",
    ),
    MixedCase(
        "X06",
        "C",
        "Change runner.c score(int) to call bonus(int) in new stats.c. "
        "Create stats.c defining bonus(value) to return value * 2 + 1.",
        "runner.c",
        "stats.c",
        "int score(int value) { return value; }\n",
        "int bonus(int value);\nint score(int value) { return bonus(value); }\n",
        "int bonus(int value) { return value * 2 + 1; }\n",
        "int score(int); int main(void) { return score(0) != 1 || score(3) != 7; }",
    ),
)


def materialize(case: MixedCase, workspace: Path, *, edit: bool, create: bool) -> None:
    """Construct one corpus state inside a caller-owned temporary workspace."""
    target = workspace / case.edit_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(case.reference_edit if edit else case.source)
    if create:
        created = workspace / case.create_path
        created.parent.mkdir(parents=True, exist_ok=True)
        created.write_text(case.reference_create)


def oracle(case: MixedCase, workspace: Path) -> bool:
    """Execute bounded semantics in the disposable fixture, without a shell."""
    if case.language == "Python":
        command = (sys.executable, "-B", "-c", case.check)
    else:
        probe = workspace / "probe.c"
        probe.write_text(case.check)
        compile_result = subprocess.run(
            (
                "cc",
                "-o",
                str(workspace / "probe"),
                str(probe),
                str(workspace / case.edit_path),
                str(workspace / case.create_path),
            ),
            cwd=workspace,
            shell=False,
            capture_output=True,
            timeout=20,
            check=False,
        )
        if compile_result.returncode:
            return False
        command = (str(workspace / "probe"),)
    result = subprocess.run(
        command,
        cwd=workspace,
        shell=False,
        capture_output=True,
        timeout=20,
        check=False,
    )
    return result.returncode == 0


def classify_intent(text: str, case: MixedCase) -> Failure:
    """Classify bounded M1 role/path output, not prose quality."""
    try:
        value = json.loads(text)
    except ValueError:
        return Failure.INTENT_ROLE_CONFUSION
    if not isinstance(value, dict):
        return Failure.INTENT_ROLE_CONFUSION
    if value.get("modify_path") == value.get("create_path"):
        return Failure.DUPLICATE_OPERATION
    if (
        value.get("modify_path") == case.create_path
        and value.get("create_path") == case.edit_path
    ):
        return Failure.INTENT_ROLE_CONFUSION
    if (
        value.get("modify_path") != case.edit_path
        or value.get("create_path") != case.create_path
    ):
        return Failure.INTENT_PATH_CONFUSION
    if not all(
        isinstance(value.get(key), str) and value[key].strip()
        for key in ("modify_intent", "create_intent")
    ):
        return Failure.INTENT_ROLE_CONFUSION
    return Failure.PASS


def classify_operations(
    operations: object, case: MixedCase, *, minimal: bool = False
) -> Failure:
    """Classify cardinality and roles before child syntax or semantics."""
    if not isinstance(operations, (list, tuple)):
        return Failure.MIXED_SCHEMA_INVALID
    if not all(isinstance(op, dict) for op in operations):
        return Failure.MIXED_SCHEMA_INVALID
    roles = (
        {"modify": "modify", "create": "create"}
        if minimal
        else {"line_range_edit": "modify", "create_file": "create"}
    )
    kinds = [roles.get(op.get("op" if minimal else "type")) for op in operations]
    paths = [op.get("path") for op in operations]
    if any(not isinstance(path, str) for path in paths):
        return Failure.UNAUTHORIZED_PATH
    if len(paths) != len(set(paths)):
        return Failure.DUPLICATE_OPERATION
    if any(path not in {case.edit_path, case.create_path} for path in paths):
        return Failure.UNAUTHORIZED_PATH
    if any(kind is None for kind in kinds):
        return Failure.WRONG_OPERATION_TYPE
    if len(operations) != 2 or set(paths) != {case.edit_path, case.create_path}:
        return Failure.MISSING_OPERATION
    if dict(zip(paths, kinds, strict=True)) != {
        case.edit_path: "modify",
        case.create_path: "create",
    }:
        return Failure.WRONG_OPERATION_TYPE
    edit = operations[paths.index(case.edit_path)]
    create = operations[paths.index(case.create_path)]
    if (
        not isinstance(edit.get("start_line"), int)
        or type(edit["start_line"]) is not int
        or not isinstance(edit.get("end_line"), int)
        or type(edit["end_line"]) is not int
        or not isinstance(edit.get("new_text"), str)
    ):
        return Failure.EDIT_CHILD_INVALID
    lines = case.source.splitlines(keepends=True)
    start, end = edit["start_line"], edit["end_line"]
    if start < 1 or end < start or end > len(lines):
        return Failure.EDIT_CHILD_INVALID
    new = edit["new_text"]
    if new.rstrip("\r\n") == "".join(lines[start - 1 : end]).rstrip("\r\n"):
        return Failure.NO_OP_EDIT
    if not isinstance(create.get("content"), str) or not create["content"]:
        return Failure.CREATE_CHILD_INVALID
    return Failure.PASS


def classify_mixed(text: str, case: MixedCase, *, minimal: bool = False) -> Failure:
    """Use the exact production parser for M4, after diagnostic role inspection."""
    try:
        payload = json.loads(text)
    except ValueError:
        return Failure.MIXED_SCHEMA_INVALID
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        return Failure.MIXED_SCHEMA_INVALID
    classified = classify_operations(payload["operations"], case, minimal=minimal)
    if classified is not Failure.PASS:
        return classified
    if minimal:
        return (
            Failure.PASS
            if set(payload) == {"operations"}
            else Failure.MIXED_SCHEMA_INVALID
        )
    try:
        parsed = parse_model_output(text)
    except ProtocolError:
        return Failure.MIXED_SCHEMA_INVALID
    return (
        Failure.PASS
        if parsed.outcome is ToolCallOutcome.MULTI_FILE_CHANGE
        else Failure.MIXED_SCHEMA_INVALID
    )


def semantic_result(
    operations: list[dict[str, object]], case: MixedCase, *, minimal: bool = False
) -> Failure:
    """Validate and execute a proposed pair via A54 in a disposable repository."""
    if classify_operations(operations, case, minimal=minimal) is not Failure.PASS:
        return classify_operations(operations, case, minimal=minimal)
    by_path = {str(item["path"]): item for item in operations}
    edit = by_path[case.edit_path]
    create = by_path[case.create_path]
    with tempfile.TemporaryDirectory(prefix="forge-a55-proposal-") as name:
        workspace = Path(name).resolve()
        materialize(case, workspace, edit=False, create=False)
        candidate = MutationCandidate(
            case.edit_path,
            hashlib.sha256(case.source.encode()).hexdigest(),
            0,
            "a55-trusted-source",
            1,
            len(case.source.splitlines()),
        )
        validation = validate_line_range_edit(
            LineRangeEditProposal(
                case.edit_path,
                edit["start_line"],
                edit["end_line"],
                edit["new_text"],
            ),
            (candidate,),
            workspace,
            0,
        )
        if not validation.valid or validation.arguments is None:
            return Failure.EDIT_CHILD_INVALID
        create_candidate = authorize_create_candidate(
            workspace, case.create_path, 0, "a55_trusted_task_metadata"
        )
        context = ExecutionContext(
            workspace,
            edit_candidates=(case.edit_path,),
            create_candidates=(create_candidate,),
        )
        normalized = (
            {"type": "edit", **validation.arguments},
            {
                "type": "create",
                "path": case.create_path,
                "content": create["content"],
            },
        )
        arguments = {
            "group_id": mixed_group_id(normalized, (create_candidate,), 0, workspace),
            "workspace_generation": 0,
            "operations": normalized,
        }
        try:
            preview_mixed_file_transaction(arguments, context)
            MixedFileTransactionTool().execute(arguments, context)
        except (ToolError, ValueError, UnicodeEncodeError):
            return Failure.MIXED_SCHEMA_INVALID
        return Failure.PASS if oracle(case, workspace) else Failure.SEMANTIC_FAIL

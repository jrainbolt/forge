"""Run A55 M1-M4 diagnostics; never persist generated source content."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from forge.models import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    Model,
    ModelRequest,
    MutationRepresentationPolicy,
    OutputSpecification,
    ResponseFormat,
    default_backend_registry,
    load_model_catalog,
)
from forge.orchestration import RepositoryChatSession
from forge.orchestration.protocol import (
    build_mutation_ready_output,
)
from forge.orchestration.repository_session import render_mixed_ready_guidance
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)
from scripts.mixed_operation_protocol_v1 import (
    CASES,
    Failure,
    MixedCase,
    classify_intent,
    classify_mixed,
    classify_operations,
    semantic_result,
)
from scripts.run_mixed_file_transaction_v1 import (
    REFERENCE_APP,
    REFERENCE_HELPER,
    _fixture,
    _oracle,
    _RecordingModel,
)
from scripts.run_mixed_file_transaction_v1 import TASK as A54_TASK

GENERATION = GenerationConfig(max_tokens=512, temperature=0, seed=42)
M1_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string"}
        for key in ("modify_path", "modify_intent", "create_path", "create_intent")
    },
    "required": ["modify_path", "modify_intent", "create_path", "create_intent"],
    "additionalProperties": False,
}
M3_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "modify"},
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "new_text": {"type": "string"},
                        },
                        "required": [
                            "op",
                            "path",
                            "start_line",
                            "end_line",
                            "new_text",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "create"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["op", "path", "content"],
                        "additionalProperties": False,
                    },
                ]
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}


def _a54_case() -> MixedCase:
    return MixedCase(
        "A54-R1",
        "Python",
        A54_TASK,
        "app.py",
        "helper.py",
        "def adjusted(value):\n    return value\n",
        REFERENCE_APP,
        REFERENCE_HELPER,
        "from app import adjusted; assert [adjusted(x) for x in (-3,5,12)] == [0,5,10]",
    )


def _prompt(case: MixedCase) -> str:
    numbered = "".join(
        f"{index} | {line}\n" for index, line in enumerate(case.source.splitlines(), 1)
    )
    return (
        f"Task: {case.task}\n"
        f"Trusted existing MODIFY path: {case.edit_path}\n"
        f"Trusted absent CREATE path: {case.create_path}\n"
        f"Current trusted {case.edit_path} source:\n{numbered}"
    )


def _ask(
    model: Model,
    prompt: str,
    output: OutputSpecification,
    *,
    correction: str | None = None,
) -> tuple[str, dict[str, object]]:
    messages = [Message(MessageRole.USER, prompt)]
    if correction:
        messages.append(Message(MessageRole.USER, correction))
    started = time.perf_counter()
    response = model.generate(ModelRequest(tuple(messages), GENERATION, output))
    return response.text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "truncated": response.finish_reason is FinishReason.MAX_TOKENS,
    }


def _record(
    text: str, failure: Failure, metrics: dict[str, object]
) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    ops = payload.get("operations") if isinstance(payload, dict) else None
    return {
        "failure": failure.value,
        "operation_count": len(ops) if isinstance(ops, list) else None,
        "children": [
            {"type": item.get("type", item.get("op")), "path": item.get("path")}
            for item in ops
            if isinstance(item, dict)
        ]
        if isinstance(ops, list)
        else [],
        **metrics,
    }


def _evaluate_mixed(
    text: str,
    case: MixedCase,
    *,
    minimal: bool,
) -> Failure:
    failure = classify_mixed(text, case, minimal=minimal)
    if failure is not Failure.PASS:
        return failure
    payload = json.loads(text)
    return semantic_result(payload["operations"], case, minimal=minimal)


def run_case(model: Model, case: MixedCase) -> dict[str, object]:
    """Run one independent layered diagnostic without production session history."""
    source = _prompt(case)
    results: dict[str, object] = {}
    text, metrics = _ask(
        model,
        source + "\nDescribe the two required operation intents briefly.",
        OutputSpecification(ResponseFormat.JSON, M1_SCHEMA),
    )
    results["M1"] = _record(text, classify_intent(text, case), metrics)

    edit_output = build_mutation_ready_output(
        (case.edit_path,), representation=MutationRepresentationPolicy.LINE_RANGE
    )
    create_output = build_mutation_ready_output((), create_paths=(case.create_path,))
    children: list[dict[str, object]] = []
    child_records = []
    for role, output, instruction in (
        (
            "modify",
            edit_output,
            "Return only the existing-file LINE_RANGE edit action.",
        ),
        ("create", create_output, "Return only the new-file create_file action."),
    ):
        child_text, child_metrics = _ask(model, source + "\n" + instruction, output)
        try:
            child = json.loads(child_text)
        except ValueError:
            child = None
        if isinstance(child, dict):
            children.append(child)
        child_records.append(
            {
                "role": role,
                "type": child.get("type") if isinstance(child, dict) else None,
                "path": child.get("path") if isinstance(child, dict) else None,
                **child_metrics,
            }
        )
    m2 = classify_operations(children, case)
    if m2 is Failure.PASS:
        m2 = semantic_result(children, case)
    results["M2"] = {"failure": m2.value, "children": child_records}

    text, metrics = _ask(
        model,
        source + "\nReturn exactly one MODIFY operation and one CREATE operation "
        "in the minimal mixed operations schema.",
        OutputSpecification(ResponseFormat.JSON, M3_SCHEMA),
    )
    results["M3"] = _record(text, _evaluate_mixed(text, case, minimal=True), metrics)

    text, metrics = _ask(
        model,
        source + "\nReturn exactly one multi_file_change with one line_range_edit "
        "for the existing file and one create_file for the absent file.",
        build_mutation_ready_output(
            (case.edit_path,),
            create_paths=(case.create_path,),
            representation=MutationRepresentationPolicy.LINE_RANGE,
        ),
    )
    failure = _evaluate_mixed(text, case, minimal=False)
    record = _record(text, failure, metrics)
    if failure not in {Failure.PASS, Failure.SEMANTIC_FAIL}:
        correction = render_mixed_ready_guidance((case.edit_path,), (case.create_path,))
        corrected, second_metrics = _ask(
            model,
            source + "\n" + correction,
            build_mutation_ready_output(
                (case.edit_path,),
                create_paths=(case.create_path,),
                representation=MutationRepresentationPolicy.LINE_RANGE,
            ),
        )
        record["correction"] = _record(
            corrected, _evaluate_mixed(corrected, case, minimal=False), second_metrics
        )
        record["correction_category"] = "explicit trusted role/path set"
    results["M4"] = record
    return {"case": case.case_id, "layers": results}


def run_m5(profile: str, model: Model) -> dict[str, object]:
    """Run A54-R1 once through the unchanged full production session."""
    recorder = _RecordingModel(model)
    with tempfile.TemporaryDirectory(prefix="forge-a55-m5-") as name:
        workspace = Path(name).resolve()
        _fixture(workspace)
        baseline = _oracle(workspace)
        (workspace / "app.py").write_text(REFERENCE_APP)
        (workspace / "helper.py").write_text(REFERENCE_HELPER)
        reference = _oracle(workspace)
        (workspace / "app.py").write_text("def adjusted(value):\n    return value\n")
        (workspace / "helper.py").unlink()
        previews = []
        commands = ProjectCommands(
            test=ProjectCommand(
                (sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"),
                20,
            )
        )
        session = RepositoryChatSession(
            profile,
            recorder,
            workspace,
            generation=GENERATION,
            registry=create_assist_repository_registry(commands, include_creation=True),
            policy=create_assist_repository_policy(),
            required_candidate_paths=("app.py",),
            create_candidate_paths=("helper.py",),
            mixed_file_operations=True,
            mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
            approval_callback=lambda _invocation, preview: (
                previews.append(preview) or True
            ),
            require_relevant_source=False,
        )
        started = time.perf_counter()
        try:
            response = session.execute_task(A54_TASK)
        except Exception as error:
            return {
                "case": "A54-R1",
                "profile": profile,
                "baseline_pass": baseline,
                "reference_pass": reference,
                "status": type(error).__name__,
                "envelopes": recorder.envelopes,
                "preview": bool(previews),
                "transaction": (workspace / "helper.py").exists(),
                "oracle_pass": _oracle(workspace),
                "tool_count": len(session.last_activity),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        return {
            "case": "A54-R1",
            "profile": profile,
            "baseline_pass": baseline,
            "reference_pass": reference,
            "status": response.coding_task.status.value
            if response.coding_task
            else "unknown",
            "envelopes": recorder.envelopes,
            "preview": bool(previews),
            "transaction": (workspace / "helper.py").exists(),
            "oracle_pass": _oracle(workspace),
            "tool_count": len(response.tool_activity),
            "model_calls": response.orchestration_steps,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--m5-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (*CASES, _a54_case())
    if args.case != "all":
        cases = tuple(case for case in cases if case.case_id == args.case)
        if not cases:
            parser.error("unknown case")
    catalog = load_model_catalog(args.config, default_backend_registry())
    model = catalog.create(args.profile)
    try:
        if args.m5_only:
            results = [run_m5(args.profile, model)]
        else:
            results = [run_case(model, case) for case in cases]
    finally:
        model.close()
    report = {
        "suite": "mixed-operation-protocol-v1",
        "profile": args.profile,
        "seed": 42,
        "temperature": 0,
        "context": 8192,
        "output": 512,
        "cases": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

"""A46 alternative-model comparison over the frozen A45 benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.realistic_semantic import (
    REALISTIC_SEMANTIC_SCHEMA_VERSION,
    REALISTIC_SEMANTIC_SUITE_VERSION,
    REALISTIC_SEMANTIC_V1,
    RealisticSemanticAggregate,
    RealisticSemanticResult,
    RealisticSemanticRun,
    SemanticIntegrityResult,
)
from forge.models import (
    GenerationConfig,
    LlamaCppConfig,
    Message,
    MessageRole,
    Model,
    ModelCatalog,
    ModelRequest,
    MutationRepresentationPolicy,
)
from forge.orchestration import ToolCallOutcome, parse_model_output
from forge.orchestration.protocol import build_mutation_ready_output

ALTERNATIVE_MODEL_BAKEOFF_V1 = "alternative-model-bakeoff-v1"
ALTERNATIVE_MODEL_BAKEOFF_SUITE_VERSION = 1
ALTERNATIVE_MODEL_BAKEOFF_SCHEMA_VERSION = 1
A46_CONTEXT_CAPACITY = 8192
A46_OUTPUT_BUDGET = 512
A46_TEMPERATURE = 0.0
A45_BASELINE_PROFILES = frozenset({"qwen-large", "qwen-small"})


class CandidateStatus(Enum):
    BASELINE = "BASELINE"
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    profile: str
    model_id: str
    backend: str
    artifact: str | None
    context_capacity: int | None
    status: CandidateStatus
    reason: str


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    profile: str
    model_id: str
    backend: str
    filename: str
    size_bytes: int
    sha256: str
    gguf_version: int | None = None
    architecture: str | None = None
    quantization: str | None = None
    declared_context: int | None = None
    chat_template: str | None = None
    chat_template_source: str = "GGUF metadata"
    license_metadata: str | None = None


@dataclass(frozen=True, slots=True)
class ModelLoadSmoke:
    passed: bool
    context_created: bool
    generation_completed: bool
    load_seconds: float
    generation_seconds: float | None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class ProtocolSmoke:
    single_file_passed: bool
    grouped_passed: bool
    single_file_failure: str | None = None
    grouped_failure: str | None = None
    single_file_class: str | None = None
    grouped_class: str | None = None
    single_file_seconds: float | None = None
    grouped_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BakeoffSeedSummary:
    seed: int
    tasks: int
    semantic_passes: int
    single_file_passes: int
    multi_file_passes: int
    mutation_ready: int
    valid_mutations: int
    previews: int
    transactions: int
    protocol_failures: int
    repairs_attempted: int
    repairs_successful: int
    verification_passes: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    token_measurement_complete: bool
    elapsed_seconds: float
    failure_layers: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class BaselineReuse:
    profile: str
    artifact: str
    repository_identity: str
    seeds: tuple[int, ...]
    summaries: tuple[BakeoffSeedSummary, ...]


@dataclass(frozen=True, slots=True)
class AlternativeModelResult:
    identity: ModelArtifactIdentity
    load_smoke: ModelLoadSmoke
    protocol_smoke: ProtocolSmoke
    realistic_semantic: tuple[RealisticSemanticRun, ...]
    summaries: tuple[BakeoffSeedSummary, ...]


@dataclass(frozen=True, slots=True)
class AlternativeModelBakeoffRun:
    suite: str
    suite_version: int
    schema_version: int
    frozen_suite: str
    frozen_suite_version: int
    frozen_schema_version: int
    repository_identity: str
    context_capacity: int
    output_budget: int
    temperature: float
    candidates: tuple[CandidateProfile, ...]
    baselines: tuple[BaselineReuse, ...]
    alternatives: tuple[AlternativeModelResult, ...]
    canonical_unchanged: bool


def enumerate_trusted_candidates(catalog: ModelCatalog) -> tuple[CandidateProfile, ...]:
    """Enumerate only profiles from the trusted user-supplied model catalog."""
    enumerated = []
    for name in catalog.profile_names:
        profile = catalog.profile(name)
        config = profile.backend_config
        artifact = getattr(config, "model_path", None)
        context = getattr(config, "context_size", None)
        if name in A45_BASELINE_PROFILES:
            status = CandidateStatus.BASELINE
            reason = "accepted A45 baseline"
        elif not isinstance(config, LlamaCppConfig):
            status = CandidateStatus.REJECTED
            reason = "candidate does not use the existing llama.cpp backend"
        elif context != A46_CONTEXT_CAPACITY:
            status = CandidateStatus.REJECTED
            reason = "candidate must use the frozen 8192-token context"
        elif not Path(config.model_path).is_file():
            status = CandidateStatus.REJECTED
            reason = "configured artifact is not a local file"
        else:
            status = CandidateStatus.ELIGIBLE
            reason = "trusted local profile satisfies static A46 eligibility"
        enumerated.append(
            CandidateProfile(
                name,
                profile.model_id,
                profile.backend_id,
                Path(artifact).name if artifact is not None else None,
                context,
                status,
                reason,
            )
        )
    return tuple(enumerated)


def identify_artifact(
    catalog: ModelCatalog,
    profile_name: str,
    *,
    gguf_version: int | None = None,
    architecture: str | None = None,
    quantization: str | None = None,
    declared_context: int | None = None,
    chat_template: str | None = None,
) -> ModelArtifactIdentity:
    """Hash one explicitly configured local artifact without copying it."""
    profile = catalog.profile(profile_name)
    config = profile.backend_config
    if not isinstance(config, LlamaCppConfig):
        raise ValueError("A46 artifact identity requires a llama.cpp profile")
    digest = hashlib.sha256()
    with config.model_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return ModelArtifactIdentity(
        profile_name,
        profile.model_id,
        profile.backend_id,
        config.model_path.name,
        config.model_path.stat().st_size,
        digest.hexdigest(),
        gguf_version,
        architecture,
        quantization,
        declared_context,
        chat_template,
    )


def run_model_load_smoke(catalog: ModelCatalog, profile_name: str) -> ModelLoadSmoke:
    """Load one configured model and perform a tiny deterministic generation."""
    started = time.perf_counter()
    try:
        model = catalog.create(profile_name)
    except Exception as error:
        return ModelLoadSmoke(
            False,
            False,
            False,
            time.perf_counter() - started,
            None,
            f"{type(error).__name__}: {error}",
        )
    load_seconds = time.perf_counter() - started
    context_created = model.context_capacity == A46_CONTEXT_CAPACITY
    if not context_created:
        model.close()
        return ModelLoadSmoke(
            False,
            False,
            False,
            load_seconds,
            None,
            f"configured context is {model.context_capacity}, expected 8192",
        )
    generation_started = time.perf_counter()
    try:
        model.generate(
            ModelRequest(
                (Message(MessageRole.USER, 'Reply with exactly: {"ok":true}'),),
                GenerationConfig(max_tokens=16, temperature=0.0, seed=42),
            )
        )
    except Exception as error:
        return ModelLoadSmoke(
            False,
            True,
            False,
            load_seconds,
            time.perf_counter() - generation_started,
            f"{type(error).__name__}: {error}",
        )
    finally:
        model.close()
    return ModelLoadSmoke(
        True,
        True,
        True,
        load_seconds,
        time.perf_counter() - generation_started,
    )


def run_protocol_smoke(model: Model) -> ProtocolSmoke:
    """Exercise unchanged production LINE_RANGE schemas without semantic scoring."""
    single = _protocol_case(model, ("src/example.py",), grouped=False)
    grouped = _protocol_case(
        model, ("src/example.py", "tests/test_example.py"), grouped=True
    )
    return ProtocolSmoke(
        single[0],
        grouped[0],
        single[2],
        grouped[2],
        single[1],
        grouped[1],
        single[3],
        grouped[3],
    )


def summarize_bakeoff_seed(
    seed: int, results: tuple[RealisticSemanticResult, ...]
) -> BakeoffSeedSummary:
    selected = tuple(item for item in results if item.seed == seed)
    layers = sorted({item.failure_layer for item in selected})
    verification_passes = sum(
        item.verification_status in {"pass", "passed"} for item in selected
    )
    return BakeoffSeedSummary(
        seed,
        len(selected),
        sum(item.final_semantic for item in selected),
        sum(
            item.final_semantic and item.mutation_kind == "single_file"
            for item in selected
        ),
        sum(
            item.final_semantic and item.mutation_kind == "multi_file"
            for item in selected
        ),
        sum(item.mutation_ready for item in selected),
        sum(item.schema_valid for item in selected),
        sum(item.preview_created for item in selected),
        sum(item.transaction_executed for item in selected),
        sum(item.failure_layer == "PROTOCOL_FAILED" for item in selected),
        sum(item.repair_used for item in selected),
        sum(item.repair_success for item in selected),
        verification_passes,
        sum(item.model_calls for item in selected),
        sum(item.input_tokens or 0 for item in selected),
        sum(item.output_tokens or 0 for item in selected),
        all(
            item.input_tokens is not None and item.output_tokens is not None
            for item in selected
        ),
        sum(item.elapsed_seconds for item in selected),
        tuple(
            (layer, sum(item.failure_layer == layer for item in selected))
            for layer in layers
        ),
    )


def load_a45_baseline(path: Path) -> BaselineReuse:
    """Load an immutable A45 result after checking comparison invariants."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "suite": REALISTIC_SEMANTIC_V1,
        "suite_version": REALISTIC_SEMANTIC_SUITE_VERSION,
        "schema_version": REALISTIC_SEMANTIC_SCHEMA_VERSION,
        "context_capacity": A46_CONTEXT_CAPACITY,
        "output_budget": A46_OUTPUT_BUDGET,
        "temperature": A46_TEMPERATURE,
        "canonical_unchanged": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"incompatible A45 baseline field {key!r}")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("A45 baseline contains no task results")
    profile = payload.get("model_profile")
    artifact = payload.get("model_artifact")
    repository = payload.get("repository_identity")
    if not all(isinstance(value, str) for value in (profile, artifact, repository)):
        raise ValueError("A45 baseline identity is malformed")
    seeds = tuple(dict.fromkeys(_integer(item, "seed") for item in raw_results))
    summaries = tuple(_summarize_serialized_seed(seed, raw_results) for seed in seeds)
    return BaselineReuse(profile, artifact, repository, seeds, summaries)


def load_realistic_semantic_run(path: Path) -> RealisticSemanticRun:
    """Load a source-free realistic semantic artifact into its typed form."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RealisticSemanticRun(
        payload["suite"],
        payload["suite_version"],
        payload["schema_version"],
        payload["forge_milestone"],
        payload["repository_identity"],
        payload["model_profile"],
        payload["model_artifact"],
        payload["context_capacity"],
        payload["output_budget"],
        payload["temperature"],
        tuple(SemanticIntegrityResult(**item) for item in payload["integrity"]),
        tuple(RealisticSemanticResult(**item) for item in payload["results"]),
        tuple(RealisticSemanticAggregate(**item) for item in payload["aggregates"]),
        payload["canonical_unchanged"],
    )


def build_bakeoff_run(
    *,
    repository_identity: str,
    candidates: tuple[CandidateProfile, ...],
    baselines: tuple[BaselineReuse, ...],
    alternatives: tuple[AlternativeModelResult, ...] = (),
    canonical_unchanged: bool = True,
) -> AlternativeModelBakeoffRun:
    identities = {item.repository_identity for item in baselines}
    if identities != {repository_identity}:
        raise ValueError("baseline repository identities do not match frozen suite")
    return AlternativeModelBakeoffRun(
        ALTERNATIVE_MODEL_BAKEOFF_V1,
        ALTERNATIVE_MODEL_BAKEOFF_SUITE_VERSION,
        ALTERNATIVE_MODEL_BAKEOFF_SCHEMA_VERSION,
        REALISTIC_SEMANTIC_V1,
        REALISTIC_SEMANTIC_SUITE_VERSION,
        REALISTIC_SEMANTIC_SCHEMA_VERSION,
        repository_identity,
        A46_CONTEXT_CAPACITY,
        A46_OUTPUT_BUDGET,
        A46_TEMPERATURE,
        candidates,
        baselines,
        alternatives,
        canonical_unchanged,
    )


def alternative_model_bakeoff_to_dict(
    run: AlternativeModelBakeoffRun,
) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    payload = convert(asdict(run))
    assert isinstance(payload, dict)
    return payload


def write_alternative_model_bakeoff_json(
    run: AlternativeModelBakeoffRun, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(alternative_model_bakeoff_to_dict(run), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _summarize_serialized_seed(seed: int, results: list[object]) -> BakeoffSeedSummary:
    selected = [
        item for item in results if isinstance(item, dict) and item.get("seed") == seed
    ]
    layers = sorted({str(item.get("failure_layer", "UNKNOWN")) for item in selected})
    return BakeoffSeedSummary(
        seed,
        len(selected),
        sum(bool(item.get("final_semantic")) for item in selected),
        sum(
            bool(item.get("final_semantic"))
            and item.get("mutation_kind") == "single_file"
            for item in selected
        ),
        sum(
            bool(item.get("final_semantic"))
            and item.get("mutation_kind") == "multi_file"
            for item in selected
        ),
        sum(bool(item.get("mutation_ready")) for item in selected),
        sum(bool(item.get("schema_valid")) for item in selected),
        sum(bool(item.get("preview_created")) for item in selected),
        sum(bool(item.get("transaction_executed")) for item in selected),
        sum(item.get("failure_layer") == "PROTOCOL_FAILED" for item in selected),
        sum(bool(item.get("repair_used")) for item in selected),
        sum(bool(item.get("repair_success")) for item in selected),
        sum(item.get("verification_status") in {"pass", "passed"} for item in selected),
        sum(_optional_integer(item, "model_calls") or 0 for item in selected),
        sum(_optional_integer(item, "input_tokens") or 0 for item in selected),
        sum(_optional_integer(item, "output_tokens") or 0 for item in selected),
        all(
            _optional_integer(item, "input_tokens") is not None
            and _optional_integer(item, "output_tokens") is not None
            for item in selected
        ),
        sum(float(item.get("elapsed_seconds", 0.0)) for item in selected),
        tuple(
            (
                layer,
                sum(
                    str(item.get("failure_layer", "UNKNOWN")) == layer
                    for item in selected
                ),
            )
            for layer in layers
        ),
    )


def _integer(item: object, key: str) -> int:
    value = _optional_integer(item, key)
    if value is None:
        raise ValueError(f"A45 result field {key!r} must be an integer")
    return value


def _optional_integer(item: object, key: str) -> int | None:
    if not isinstance(item, dict):
        return None
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _protocol_case(
    model: Model, paths: tuple[str, ...], *, grouped: bool
) -> tuple[bool, str, str | None, float]:
    if grouped:
        request_text = (
            "Two authorized files each contain one line: value = 1. Return one "
            "grouped LINE_RANGE action changing line 1 in both files to value = 2."
        )
        expected = ToolCallOutcome.MULTI_FILE_LINE_RANGE_EDIT
    else:
        request_text = (
            "The authorized file src/example.py contains one line: value = 1. "
            "Return one LINE_RANGE action changing line 1 to value = 2."
        )
        expected = ToolCallOutcome.LINE_RANGE_EDIT
    request = ModelRequest(
        (
            Message(
                MessageRole.SYSTEM,
                "Return only the JSON action required by the supplied schema.",
            ),
            Message(MessageRole.USER, request_text),
        ),
        GenerationConfig(max_tokens=A46_OUTPUT_BUDGET, temperature=0.0, seed=42),
        build_mutation_ready_output(
            paths, representation=MutationRepresentationPolicy.LINE_RANGE
        ),
    )
    started = time.perf_counter()
    try:
        response = model.generate(request)
        parsed = parse_model_output(response.text)
    except Exception as error:
        return (
            False,
            "SCHEMA_INVALID",
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
        )
    elapsed = time.perf_counter() - started
    if parsed.outcome is not expected:
        return False, parsed.outcome.value, "unexpected action type", elapsed
    if grouped:
        assert parsed.multi_file_line_range_edit is not None
        actual_paths = {str(item["path"]) for item in parsed.multi_file_line_range_edit}
    else:
        assert parsed.line_range_edit is not None
        actual_paths = {str(parsed.line_range_edit["path"])}
    if actual_paths != set(paths):
        return False, parsed.outcome.value, "authorized path set mismatch", elapsed
    return True, parsed.outcome.value, None, elapsed

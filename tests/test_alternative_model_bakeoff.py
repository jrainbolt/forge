from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    ALTERNATIVE_MODEL_BAKEOFF_V1,
    CandidateStatus,
    RealisticSemanticResult,
    alternative_model_bakeoff_to_dict,
    build_bakeoff_run,
    enumerate_trusted_candidates,
    identify_artifact,
    load_a45_baseline,
    load_realistic_semantic_run,
    run_model_load_smoke,
    run_protocol_smoke,
    summarize_bakeoff_seed,
)
from forge.models import (
    BackendDefinition,
    BackendRegistry,
    LlamaCppConfig,
    MockModel,
    ModelCatalog,
    ModelProfile,
    MutationRepresentationPolicy,
    default_backend_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _profile(name: str, artifact: Path, context: int = 8192) -> ModelProfile:
    return ModelProfile(
        name,
        "llama.cpp",
        f"{name}-Instruct",
        LlamaCppConfig(artifact, context_size=context),
        MutationRepresentationPolicy.LINE_RANGE,
    )


def _result(
    task_id: str,
    mutation_kind: str,
    *,
    seed: int = 42,
    passed: bool = True,
) -> RealisticSemanticResult:
    return RealisticSemanticResult(
        task_id=task_id,
        task_version=1,
        repository_identity="repo",
        model_profile="candidate",
        model_artifact="candidate.gguf:sha256:abc",
        seed=seed,
        context_capacity=8192,
        output_budget=512,
        temperature=0.0,
        path_mode="discovery_required",
        mutation_kind=mutation_kind,
        discovery_status="PASS",
        source_status="PASS",
        mutation_ready=True,
        schema_valid=True,
        preview_created=True,
        transaction_executed=True,
        verification_status="pass" if passed else "fail",
        build_test_status="pass" if passed else "fail",
        semantic_oracle="PASS" if passed else "FAIL",
        first_pass_semantic=passed,
        final_semantic=passed,
        repair_used=False,
        repair_success=False,
        failure_layer="PASS" if passed else "SEMANTIC_ORACLE_FAILED",
        discovery_calls=1,
        source_reads=1,
        required_candidates=0,
        required_sources_ready=0,
        mutations=1,
        tool_calls=3,
        model_calls=2,
        input_tokens=100,
        output_tokens=20,
        context_peak=1000,
        elapsed_seconds=2.5,
    )


def test_candidate_enumeration_uses_only_trusted_catalog_profiles(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"local model")
    repository_claim = tmp_path / "repository" / "forge.toml"
    repository_claim.parent.mkdir()
    repository_claim.write_text("[models.untrusted]\n", encoding="utf-8")
    catalog = ModelCatalog(
        (
            _profile("qwen-large", artifact),
            _profile("candidate-a", artifact),
            _profile("wrong-context", artifact, 4096),
        ),
        default_backend_registry(),
    )

    results = enumerate_trusted_candidates(catalog)

    assert [item.profile for item in results] == [
        "candidate-a",
        "qwen-large",
        "wrong-context",
    ]
    assert [item.status for item in results] == [
        CandidateStatus.ELIGIBLE,
        CandidateStatus.BASELINE,
        CandidateStatus.REJECTED,
    ]
    assert all(item.profile != "untrusted" for item in results)


def test_artifact_identity_records_exact_local_file(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"candidate bytes")
    catalog = ModelCatalog(
        (_profile("candidate", artifact),), default_backend_registry()
    )

    identity = identify_artifact(
        catalog,
        "candidate",
        gguf_version=3,
        architecture="test",
        quantization="Q4_K_M",
        declared_context=32768,
        chat_template="metadata-template",
    )

    assert identity.filename == artifact.name
    assert identity.size_bytes == len(b"candidate bytes")
    assert len(identity.sha256) == 64
    assert identity.chat_template_source == "GGUF metadata"


def test_load_smoke_uses_catalog_and_requires_frozen_context() -> None:
    registry = BackendRegistry(
        {
            "fake": BackendDefinition(
                lambda model_id, settings: settings,
                lambda _config: MockModel(("ok",), context_capacity=8192),
            )
        }
    )
    catalog = ModelCatalog((ModelProfile("candidate", "fake", "id", {}),), registry)

    result = run_model_load_smoke(catalog, "candidate")

    assert result.passed and result.context_created and result.generation_completed
    assert result.generation_seconds is not None


def test_protocol_smoke_uses_production_line_range_schemas() -> None:
    model = MockModel(
        (
            '{"type":"line_range_edit","path":"src/example.py",'
            '"start_line":1,"end_line":1,"new_text":"value = 2"}',
            '{"type":"multi_file_line_range_edit","edits":['
            '{"path":"src/example.py","start_line":1,"end_line":1,'
            '"new_text":"value = 2"},'
            '{"path":"tests/test_example.py","start_line":1,"end_line":1,'
            '"new_text":"value = 2"}]}',
        ),
        context_capacity=8192,
    )

    result = run_protocol_smoke(model)

    assert result.single_file_passed and result.grouped_passed
    assert result.single_file_class == "line_range_edit"
    assert result.grouped_class == "multi_file_line_range_edit"
    assert all(request.generation.max_tokens == 512 for request in model.requests)


def test_a45_baselines_are_directly_reusable() -> None:
    large = load_a45_baseline(ROOT / "eval-results/a45-qwen-large.json")
    small = load_a45_baseline(ROOT / "eval-results/a45-qwen-small.json")

    assert large.seeds == (42, 43)
    assert small.seeds == (42,)
    assert [summary.semantic_passes for summary in large.summaries] == [3, 3]
    assert small.summaries[0].semantic_passes == 4
    assert large.repository_identity == small.repository_identity


def test_typed_realistic_result_loader_preserves_source_free_metrics() -> None:
    run = load_realistic_semantic_run(ROOT / "eval-results/a45-qwen-small.json")

    assert run.model_profile == "qwen-small"
    assert len(run.results) == 8
    assert run.aggregates[0].final_semantic_passes == 4


def test_incompatible_baseline_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        (ROOT / "eval-results/a45-qwen-small.json").read_text(encoding="utf-8")
    )
    payload["context_capacity"] = 16384
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_a45_baseline(path)
    except ValueError as error:
        assert "context_capacity" in str(error)
    else:
        raise AssertionError("incompatible baseline was accepted")


def test_candidate_aggregation_separates_single_multi_and_structure() -> None:
    single = _result("R01", "single_file")
    multi = _result("R05", "multi_file", passed=False)
    protocol = replace(
        _result("R06", "multi_file", passed=False),
        schema_valid=False,
        preview_created=False,
        transaction_executed=False,
        failure_layer="PROTOCOL_FAILED",
        input_tokens=None,
        output_tokens=None,
    )

    summary = summarize_bakeoff_seed(42, (single, multi, protocol))

    assert summary.semantic_passes == summary.single_file_passes == 1
    assert summary.multi_file_passes == 0
    assert summary.mutation_ready == 3 and summary.valid_mutations == 2
    assert summary.protocol_failures == 1
    assert not summary.token_measurement_complete
    assert dict(summary.failure_layers) == {
        "PASS": 1,
        "PROTOCOL_FAILED": 1,
        "SEMANTIC_ORACLE_FAILED": 1,
    }


def test_seed_comparison_remains_explicit() -> None:
    seed_42 = _result("R01", "single_file")
    seed_43 = replace(seed_42, seed=43, final_semantic=False)

    assert summarize_bakeoff_seed(42, (seed_42, seed_43)).semantic_passes == 1
    assert summarize_bakeoff_seed(43, (seed_42, seed_43)).semantic_passes == 0


def test_serialization_contains_no_source_or_reference_mutation() -> None:
    large = load_a45_baseline(ROOT / "eval-results/a45-qwen-large.json")
    run = build_bakeoff_run(
        repository_identity=large.repository_identity,
        candidates=(),
        baselines=(large,),
    )

    payload = alternative_model_bakeoff_to_dict(run)
    rendered = json.dumps(payload)
    assert payload["suite"] == ALTERNATIVE_MODEL_BAKEOFF_V1
    assert "old_text" not in rendered
    assert "new_text" not in rendered
    assert "reference_mutation" not in rendered


def test_bakeoff_rejects_mismatched_baseline_repository() -> None:
    baseline = load_a45_baseline(ROOT / "eval-results/a45-qwen-small.json")
    try:
        build_bakeoff_run(
            repository_identity="different",
            candidates=(),
            baselines=(baseline,),
        )
    except ValueError as error:
        assert "repository identities" in str(error)
    else:
        raise AssertionError("mismatched baseline repository was accepted")

from __future__ import annotations

from pathlib import Path

import pytest

from forge.cli import _resolve_model_profile, build_parser
from forge.interaction import AutonomyMode
from forge.models import (
    BackendDefinition,
    BackendRegistry,
    MockModel,
    ModelCatalog,
    ModelDefaults,
    ModelError,
    ModelProfile,
    ModelRole,
    ModelSelectionError,
    default_backend_registry,
    load_model_catalog,
)


def _catalog(
    *,
    coding: str = "qwen-large",
    failing_large: bool = False,
    built: list[str] | None = None,
) -> ModelCatalog:
    observed = [] if built is None else built

    def build(name: object) -> MockModel:
        observed.append(str(name))
        if failing_large and name == "qwen-large":
            raise ModelError("load failed")
        return MockModel(("ok",))

    registry = BackendRegistry(
        {"fake": BackendDefinition(lambda model_id, _raw: model_id, build)}
    )
    return ModelCatalog(
        tuple(
            ModelProfile(name, "fake", name, name)
            for name in ("qwen-small", "qwen-large", "codestral-22b")
        ),
        registry,
        defaults=ModelDefaults(coding_profile=coding),
    )


def test_coding_default_resolves_to_qwen_large() -> None:
    catalog = _catalog()
    for mode in (AutonomyMode.ASSIST, AutonomyMode.AGENT, AutonomyMode.REPAIR):
        assert _resolve_model_profile(catalog, None, mode) == "qwen-large"


@pytest.mark.parametrize("profile", ("qwen-small", "codestral-22b"))
def test_explicit_profile_override_is_preserved(profile: str) -> None:
    assert _resolve_model_profile(_catalog(), profile, AutonomyMode.ASSIST) == profile


def test_normal_and_repository_chat_defaults_remain_unconfigured() -> None:
    catalog = _catalog()
    with pytest.raises(ModelSelectionError, match="for chat"):
        catalog.resolve_profile(None, ModelRole.CHAT)
    with pytest.raises(ModelSelectionError, match="for repository"):
        catalog.resolve_profile(None, ModelRole.REPOSITORY)


def test_missing_or_invalid_default_profile_fails_clearly() -> None:
    catalog = _catalog(coding="missing")
    with pytest.raises(ModelSelectionError, match="default coding.*unavailable"):
        catalog.resolve_profile(None, ModelRole.CODING)
    with pytest.raises(ModelSelectionError, match="unknown model profile"):
        catalog.resolve_profile("invalid", ModelRole.CODING)


def test_load_failure_does_not_fall_back() -> None:
    built: list[str] = []
    catalog = _catalog(failing_large=True, built=built)
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    with pytest.raises(ModelError, match="load failed"):
        catalog.create(selected)
    assert built == ["qwen-large"]


def test_legacy_config_without_defaults_gets_coding_default(tmp_path: Path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"fixture")
    config = tmp_path / "forge.toml"
    config.write_text(
        "[models.qwen-large]\n"
        "backend='llama.cpp'\n"
        "model_id='large'\n"
        "[models.qwen-large.backend_config]\n"
        f"model_path='{weights}'\n",
        encoding="utf-8",
    )
    catalog = load_model_catalog(config, default_backend_registry())
    assert catalog.resolve_profile(None, ModelRole.CODING) == "qwen-large"


def test_configured_coding_default_is_reversible(tmp_path: Path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"fixture")
    config = tmp_path / "forge.toml"
    profiles = "".join(
        f"[models.{name}]\nbackend='llama.cpp'\nmodel_id='{name}'\n"
        f"[models.{name}.backend_config]\nmodel_path='{weights}'\n"
        for name in ("qwen-small", "qwen-large")
    )
    config.write_text(
        "[defaults]\ncoding_profile='qwen-small'\n" + profiles,
        encoding="utf-8",
    )
    catalog = load_model_catalog(config, default_backend_registry())
    assert catalog.resolve_profile(None, ModelRole.CODING) == "qwen-small"


def test_evaluator_still_requires_explicit_model() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--config", "forge.toml"])


def test_no_model_name_branch_in_orchestration() -> None:
    root = Path(__file__).parents[1]
    text = (root / "src/forge/cli.py").read_text()
    orchestration = "".join(
        path.read_text() for path in (root / "src/forge/orchestration").glob("*.py")
    )
    assert "qwen-large" not in text
    assert "qwen-small" not in text
    assert "qwen-large" not in orchestration
    assert "qwen-small" not in orchestration


def test_external_weights_and_gguf_are_package_excluded() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert ".gguf" not in configuration.lower()
    assert not tuple((Path(__file__).parents[1] / "src").rglob("*.gguf"))

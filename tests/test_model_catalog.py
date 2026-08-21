from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from forge.models import (
    BackendDefinition,
    BackendRegistry,
    GenerationConfig,
    Message,
    MessageRole,
    MockModel,
    ModelCatalog,
    ModelConfigurationError,
    ModelProfile,
    ModelRequest,
    ModelSelectionError,
    default_backend_registry,
    load_model_catalog,
)
from forge.models.llama_cpp import LlamaCppConfig


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "forge.toml"
    path.write_text(text)
    return path


def _valid_config(tmp_path: Path) -> Path:
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    first.touch()
    second.touch()
    return _write_config(
        tmp_path,
        f"""
[models.small]
backend = "llama.cpp"
model_id = "small-model"
[models.small.backend_config]
model_path = "{first}"
context_size = 2048
gpu_layers = 0

[models.large]
backend = "llama.cpp"
model_id = "large-model"
[models.large.backend_config]
model_path = "{second}"
threads = 4
""",
    )


def test_loads_multiple_typed_profiles_without_constructing_models(
    tmp_path: Path,
) -> None:
    catalog = load_model_catalog(_valid_config(tmp_path), default_backend_registry())

    assert catalog.profile_names == ("large", "small")
    profile = catalog.profile("small")
    assert profile.model_id == "small-model"
    assert profile.backend_id == "llama.cpp"
    assert profile.backend_config == LlamaCppConfig(
        model_path=tmp_path / "first.gguf",
        model_id="small-model",
        context_size=2048,
        gpu_layers=0,
    )


def test_selected_profile_is_the_only_model_constructed() -> None:
    built: list[object] = []

    def parse(model_id: str, settings: Mapping[str, object]) -> object:
        return (model_id, settings["answer"])

    def build(config: object) -> MockModel:
        built.append(config)
        return MockModel(responses=("selected",))

    registry = BackendRegistry(
        {"fake": BackendDefinition(parse_config=parse, build=build)}
    )
    catalog = ModelCatalog(
        (
            ModelProfile("one", "fake", "first", ("first", 1)),
            ModelProfile("two", "fake", "second", ("second", 2)),
        ),
        registry,
    )

    assert built == []
    with catalog.create("two") as model:
        response = model.generate(
            ModelRequest(
                messages=(Message(MessageRole.USER, "request"),),
                generation=GenerationConfig(),
            )
        )
    assert response.text == "selected"
    assert built == [("second", 2)]


def test_same_caller_operates_with_two_distinct_profiles() -> None:
    def build(config: object) -> MockModel:
        return MockModel(responses=(f"response from {config}",))

    registry = BackendRegistry(
        {"fake": BackendDefinition(lambda model_id, _data: model_id, build)}
    )
    catalog = ModelCatalog(
        (
            ModelProfile("model-a", "fake", "first", "first"),
            ModelProfile("model-b", "fake", "second", "second"),
        ),
        registry,
    )
    request = ModelRequest(messages=(Message(MessageRole.USER, "same request"),))

    responses = []
    for name in catalog.profile_names:
        with catalog.create(name) as model:
            responses.append(model.generate(request).text)

    assert responses == ["response from first", "response from second"]


def test_default_registries_are_independent_immutable_instances() -> None:
    first = default_backend_registry()
    second = default_backend_registry()
    assert first is not second
    assert first.backend_ids == second.backend_ids == ("llama.cpp",)


def test_programmatic_duplicate_profile_is_rejected() -> None:
    registry = BackendRegistry(
        {"fake": BackendDefinition(lambda _model, data: data, lambda data: data)}
    )
    profile = ModelProfile("same", "fake", "model", {})
    with pytest.raises(ModelConfigurationError, match="duplicate"):
        ModelCatalog((profile, profile), registry)


def test_profile_is_immutable() -> None:
    profile = ModelProfile("name", "backend", "model", {})
    with pytest.raises(FrozenInstanceError):
        profile.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not valid = [", "cannot load"),
        ("title = 'empty'", "unknown top-level"),
        ("models = {}", "must define"),
        (
            "[models.bad]\nbackend='missing'\nmodel_id='x'\n"
            "[models.bad.backend_config]\nmodel_path='x'",
            "unknown backend",
        ),
        (
            "[models.bad]\nbackend='llama.cpp'\nmodel_id='x'\nextra=1\n"
            "[models.bad.backend_config]\nmodel_path='x'",
            "unknown keys",
        ),
        (
            "[models.bad]\nbackend='llama.cpp'\nmodel_id='x'\n"
            "[models.bad.backend_config]\ncontext_size=1",
            "requires model_path",
        ),
        (
            "[models.bad]\nbackend='llama.cpp'\nmodel_id='x'\n"
            "[models.bad.backend_config]\nmodel_path='missing.gguf'",
            "not a file",
        ),
    ],
)
def test_invalid_configuration_fails_clearly(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises((ModelConfigurationError, ModelSelectionError), match=message):
        load_model_catalog(_write_config(tmp_path, text), default_backend_registry())


def test_unknown_profile_lists_available_names(tmp_path: Path) -> None:
    catalog = load_model_catalog(_valid_config(tmp_path), default_backend_registry())
    with pytest.raises(ModelSelectionError, match="large, small"):
        catalog.create("absent")

"""Immutable model profiles and explicit backend construction."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from forge.models.llama_cpp import BACKEND_ID, LlamaCppConfig, LlamaCppModel
from forge.models.model import Model

BackendConfigParser = Callable[[str, Mapping[str, object]], object]
BackendBuilder = Callable[[object], Model]


class ModelConfigurationError(ValueError):
    """A model catalog or backend setting is invalid."""


class ModelSelectionError(LookupError):
    """A requested model profile or backend is unavailable."""


@dataclass(frozen=True, slots=True)
class BackendDefinition:
    """The parsing and construction functions owned by one backend."""

    parse_config: BackendConfigParser
    build: BackendBuilder


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A named, backend-neutral model selection."""

    name: str
    backend_id: str
    model_id: str
    backend_config: object


class BackendRegistry:
    """An immutable, explicitly constructed backend registry."""

    def __init__(self, definitions: Mapping[str, BackendDefinition]) -> None:
        if not definitions:
            raise ModelConfigurationError("at least one backend must be registered")
        copied = dict(definitions)
        if any(not isinstance(key, str) or not key.strip() for key in copied):
            raise ModelConfigurationError("backend identifiers must be non-empty text")
        if any(not isinstance(value, BackendDefinition) for value in copied.values()):
            raise TypeError("backend definitions must be BackendDefinition values")
        self._definitions = MappingProxyType(copied)

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def parse_config(
        self, backend_id: str, model_id: str, settings: Mapping[str, object]
    ) -> object:
        return self._definition(backend_id).parse_config(model_id, settings)

    def build(self, profile: ModelProfile) -> Model:
        model = self._definition(profile.backend_id).build(profile.backend_config)
        if not isinstance(model, Model):
            raise TypeError(f"backend {profile.backend_id!r} did not build a Model")
        return model

    def _definition(self, backend_id: str) -> BackendDefinition:
        try:
            return self._definitions[backend_id]
        except KeyError as error:
            known = ", ".join(self.backend_ids)
            raise ModelSelectionError(
                f"unknown backend {backend_id!r}; available backends: {known}"
            ) from error


class ModelCatalog:
    """Immutable profiles with lazy, selected-only model construction."""

    def __init__(
        self, profiles: Iterable[ModelProfile], registry: BackendRegistry
    ) -> None:
        copied: dict[str, ModelProfile] = {}
        for profile in profiles:
            if not isinstance(profile, ModelProfile):
                raise TypeError("profiles must contain ModelProfile values")
            if profile.name in copied:
                raise ModelConfigurationError(
                    f"duplicate model profile name: {profile.name!r}"
                )
            copied[profile.name] = profile
        if not copied:
            raise ModelConfigurationError("model catalog must contain a profile")
        self._profiles = MappingProxyType(copied)
        self._registry = registry

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def profile(self, name: str) -> ModelProfile:
        try:
            return self._profiles[name]
        except KeyError as error:
            known = ", ".join(self.profile_names)
            raise ModelSelectionError(
                f"unknown model profile {name!r}; available profiles: {known}"
            ) from error

    def create(self, name: str) -> Model:
        """Construct only the explicitly selected profile."""
        return self._registry.build(self.profile(name))


def default_backend_registry() -> BackendRegistry:
    """Return a fresh registry containing Forge's supported backends."""
    return BackendRegistry(
        {
            BACKEND_ID: BackendDefinition(
                parse_config=_parse_llama_cpp_config,
                build=_build_llama_cpp_model,
            )
        }
    )


def load_model_catalog(path: Path, registry: BackendRegistry) -> ModelCatalog:
    """Load and validate named model profiles from a TOML file."""
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ModelConfigurationError(
            f"cannot load model configuration {config_path}: {error}"
        ) from error

    unknown_root = set(document) - {"models"}
    if unknown_root:
        raise ModelConfigurationError(
            f"unknown top-level configuration keys: {_format_keys(unknown_root)}"
        )
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelConfigurationError("configuration must define [models.<name>]")

    profiles = []
    for name, raw_profile in models.items():
        if not isinstance(name, str) or not name.strip():
            raise ModelConfigurationError("profile names must be non-empty text")
        if not isinstance(raw_profile, dict):
            raise ModelConfigurationError(f"profile {name!r} must be a TOML table")
        profiles.append(_parse_profile(name, raw_profile, registry))
    return ModelCatalog(profiles, registry)


def _parse_profile(
    name: str, raw: Mapping[str, object], registry: BackendRegistry
) -> ModelProfile:
    unknown = set(raw) - {"backend", "model_id", "backend_config"}
    if unknown:
        raise ModelConfigurationError(
            f"profile {name!r} has unknown keys: {_format_keys(unknown)}"
        )
    backend_id = _required_text(raw, "backend", name)
    model_id = _required_text(raw, "model_id", name)
    settings = raw.get("backend_config")
    if not isinstance(settings, dict):
        raise ModelConfigurationError(
            f"profile {name!r} must define a [models.{name}.backend_config] table"
        )
    try:
        backend_config = registry.parse_config(backend_id, model_id, settings)
    except ModelSelectionError:
        raise
    except (TypeError, ValueError) as error:
        raise ModelConfigurationError(
            f"invalid backend configuration for profile {name!r}: {error}"
        ) from error
    return ModelProfile(name, backend_id, model_id, backend_config)


def _parse_llama_cpp_config(
    model_id: str, settings: Mapping[str, object]
) -> LlamaCppConfig:
    allowed = {"model_path", "context_size", "gpu_layers", "threads", "verbose"}
    unknown = set(settings) - allowed
    if unknown:
        raise ModelConfigurationError(
            f"unknown llama.cpp settings: {_format_keys(unknown)}"
        )
    if "model_path" not in settings:
        raise ModelConfigurationError("llama.cpp requires model_path")
    return LlamaCppConfig(model_id=model_id, **dict(settings))  # type: ignore[arg-type]


def _build_llama_cpp_model(config: object) -> Model:
    if not isinstance(config, LlamaCppConfig):
        raise TypeError("llama.cpp requires LlamaCppConfig")
    return LlamaCppModel(config)


def _required_text(raw: Mapping[str, object], key: str, profile: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigurationError(
            f"profile {profile!r} field {key!r} must be non-empty text"
        )
    return value


def _format_keys(keys: Iterable[str]) -> str:
    return ", ".join(sorted(repr(key) for key in keys))

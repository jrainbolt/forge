from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from forge.config import ForgeConfig


def test_configuration_defaults() -> None:
    assert ForgeConfig.from_environment({}) == ForgeConfig(verbose=False)


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_verbose_environment_override(value: str) -> None:
    assert ForgeConfig.from_environment({"FORGE_VERBOSE": value}).verbose is True


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_non_verbose_environment_override(value: str) -> None:
    assert ForgeConfig.from_environment({"FORGE_VERBOSE": value}).verbose is False


def test_invalid_environment_override_fails_clearly() -> None:
    with pytest.raises(ValueError, match="FORGE_VERBOSE"):
        ForgeConfig.from_environment({"FORGE_VERBOSE": "sometimes"})


def test_configuration_is_immutable() -> None:
    config = ForgeConfig()
    with pytest.raises(FrozenInstanceError):
        config.verbose = True  # type: ignore[misc]

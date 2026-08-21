"""Bootstrap configuration for Forge."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class ForgeConfig:
    """Immutable application configuration for bootstrap concerns."""

    verbose: bool = False
    model_config_path: Path | None = None

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> ForgeConfig:
        """Load configuration from an explicit environment mapping."""
        values = os.environ if environment is None else environment
        raw_path = values.get("FORGE_CONFIG")
        model_config_path = (
            Path(raw_path).expanduser() if raw_path is not None else None
        )
        raw_verbose = values.get("FORGE_VERBOSE")
        if raw_verbose is None:
            return cls(model_config_path=model_config_path)

        normalized = raw_verbose.strip().lower()
        if normalized in _TRUE_VALUES:
            return cls(verbose=True, model_config_path=model_config_path)
        if normalized in _FALSE_VALUES:
            return cls(verbose=False, model_config_path=model_config_path)
        raise ValueError(
            "FORGE_VERBOSE must be one of: 1, 0, true, false, yes, no, on, off"
        )

    def with_verbose(self, verbose: bool) -> ForgeConfig:
        """Return a copy with an explicit CLI verbosity choice."""
        return ForgeConfig(verbose=verbose, model_config_path=self.model_config_path)

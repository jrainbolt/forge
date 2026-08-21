"""Run the A3 coding smoke request through one configured model profile."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from forge.config import ForgeConfig
from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    ModelRequest,
    default_backend_registry,
    load_model_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    path = args.config or ForgeConfig.from_environment().model_config_path
    if path is None:
        parser.error("provide --config or set FORGE_CONFIG")

    catalog = load_model_catalog(path, default_backend_registry())
    request = ModelRequest(
        messages=(
            Message(
                MessageRole.USER,
                "Write a C function that returns the maximum value in an array "
                "of signed integers.",
            ),
        ),
        generation=GenerationConfig(max_tokens=192, temperature=0.0, seed=42),
    )
    load_started = time.perf_counter()
    with catalog.create(args.profile) as model:
        load_seconds = time.perf_counter() - load_started
        generation_started = time.perf_counter()
        response = model.generate(request)
        generation_seconds = time.perf_counter() - generation_started
    print(response.text)
    print(
        f"profile={args.profile} model={response.identity.model_id} "
        f"backend={response.identity.backend_id} load_seconds={load_seconds:.2f} "
        f"generation_seconds={generation_seconds:.2f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

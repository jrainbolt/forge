"""Run one developer-facing local inference smoke request."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.models import (
    GenerationConfig,
    LlamaCppConfig,
    LlamaCppModel,
    Message,
    MessageRole,
    ModelRequest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to a local GGUF model")
    parser.add_argument("--model-id", help="explicit model identity")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = LlamaCppConfig(
        model_path=args.model,
        model_id=args.model_id,
        context_size=args.context_size,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        verbose=args.verbose,
    )
    request = ModelRequest(
        messages=(
            Message(
                MessageRole.USER,
                "Write a Python function that returns the square of an integer.",
            ),
        ),
        generation=GenerationConfig(max_tokens=96, temperature=0.0, seed=42),
    )
    with LlamaCppModel(config) as model:
        response = model.generate(request)
    print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

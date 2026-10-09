"""Freeze A78 model identities and qualify the held-out corpus."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.suite import (
    PROFILES,
    QUALIFICATION_RUN_ID,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.realistic_coding_v2.suite import (
    BUILD,
    CONFIGURE,
    REPOSITORY,
    TEST,
    baseline_workspace,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evaluation.realworld import EvaluationOutcome, run_oracle
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _health(definition):  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory(prefix="forge-a78-") as name:
        root = Path(name).resolve()
        baseline = baseline_workspace(definition, root / "baseline")
        baseline_oracle = run_oracle(
            baseline, definition.production_task.oracle_commands
        )
        reference = root / "reference"
        shutil.copytree(
            baseline, reference, ignore=shutil.ignore_patterns("build", "__pycache__")
        )
        for relative in definition.production_task.expected_changed_paths:
            target = reference / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPOSITORY / relative).read_bytes())
        reference_oracle = run_oracle(
            reference, definition.production_task.oracle_commands
        )
        reference_verification = run_oracle(reference, (CONFIGURE, BUILD, TEST))
    return {
        "baseline_oracle": baseline_oracle.value,
        "reference_oracle": reference_oracle.value,
        "reference_verification": reference_verification.value,
        "wrong_or_incomplete_state": "baseline",
        "wrong_or_incomplete_fails": baseline_oracle is EvaluationOutcome.FAIL,
        "eligible": baseline_oracle is EvaluationOutcome.FAIL
        and reference_oracle is EvaluationOutcome.PASS
        and reference_verification is EvaluationOutcome.PASS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    catalog = load_model_catalog(args.config, default_backend_registry())
    inventory = {}
    for name in PROFILES:
        profile = catalog.profile(name)
        config = profile.backend_config
        if not isinstance(config, LlamaCppConfig):
            raise RuntimeError("A78 requires local llama.cpp profiles")
        inventory[name] = {
            "model_id": profile.model_id,
            "backend_id": profile.backend_id,
            "gguf_filename": config.model_path.name,
            "file_size": config.model_path.stat().st_size,
            "sha256": _sha256(config.model_path),
            "context_size": config.context_size,
            "chat_format": config.chat_format or "backend_auto",
            "config_identity": hashlib.sha256(repr(profile).encode()).hexdigest(),
        }
    health = {definition.task_id: _health(definition) for definition in tasks()}
    if not all(item["eligible"] for item in health.values()):
        raise RuntimeError(f"A78 qualification failed: {health}")
    atomic_checkpoint(
        args.output,
        {
            "run_identity": QUALIFICATION_RUN_ID,
            "corpus_identity": corpus_identity(),
            "inventory": inventory,
            "task_health": health,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

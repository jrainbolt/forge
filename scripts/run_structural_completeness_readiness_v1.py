"""Execute A84 model cells and non-model operational controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.structural_completeness_readiness_v1.runner import (
    commit,
    from_model,
    run_control,
)
from benchmarks.structural_completeness_readiness_v1.suite import (
    CASES,
    REPRESENTATION,
    RUN_ID,
    corpus_identity,
    tasks,
)
from forge.evaluation.mutation_ready import resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = resume_checkpoint(args.manifest)
    if (
        manifest is None
        or manifest.get("run_identity") != RUN_ID
        or manifest.get("corpus_identity") != corpus_identity()
        or manifest.get("blocking_active")
    ):
        raise RuntimeError("A84 frozen manifest mismatch")
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A84 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError(f"A84 coding default changed: {selected}")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A84 model configuration changed")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    repository_identity = source_state_identity(REPOSITORY)
    definitions = {item.task_id: item for item in tasks()}
    model = catalog.create(selected)
    try:
        for case in CASES:
            path = args.checkpoint_dir / "cells" / f"{case.case_id}.json"
            if path.exists():
                prior = resume_checkpoint(path)
                if prior is None or prior.get("run_identity") != RUN_ID:
                    raise RuntimeError(f"A84 invalid resume checkpoint: {path}")
                print(json.dumps({"case": case.case_id, "resume": True}), flush=True)
                continue
            before = source_state_identity(REPOSITORY)
            if case.model_call:
                result = from_model(
                    case,
                    definitions[case.task_id],  # type: ignore[index]
                    model,
                    selected_profile=selected,
                    artifact=artifact,
                    artifact_size=inventory["file_size"],
                    model_config_identity=config_identity,
                    repository_identity=repository_identity,
                    corpus_identity=corpus_identity(),
                    representation=REPRESENTATION,
                )
            else:
                result = run_control(case, corpus_identity=corpus_identity())
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical source identity changed")
            commit(path, result)
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "decision": result.primary_shadow["decision"],
                        "policy": result.offline_policy_action,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one A46 candidate smoke in a dedicated process."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from forge.evaluation import run_model_load_smoke, run_protocol_smoke
from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("load", "protocol"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = load_model_catalog(args.config, default_backend_registry())
    if args.kind == "load":
        result = run_model_load_smoke(catalog, args.model)
        passed = result.passed
    else:
        model = catalog.create(args.model)
        try:
            result = run_protocol_smoke(model)
        finally:
            model.close()
        passed = result.single_file_passed and result.grouped_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

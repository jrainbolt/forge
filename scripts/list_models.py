"""List configured Forge model profiles without loading their model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.models import default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    catalog = load_model_catalog(args.config, default_backend_registry())
    for name in catalog.profile_names:
        profile = catalog.profile(name)
        print(f"{name}\t{profile.model_id}\t{profile.backend_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

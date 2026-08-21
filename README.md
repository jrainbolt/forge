# Forge

Forge is an early-stage, local-first AI coding assistant and software-development
agent. Its long-term design keeps language models interchangeable and keeps the
core useful without cloud services. The current implementation provides the
project bootstrap and a generic, backend-independent model API with a
deterministic test model. It does not yet perform real model inference.

## Development setup

Forge requires Python 3.12 or later. Create an isolated environment and install
the project with its development tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the CLI:

```bash
forge --help
python -m forge --help
```

Run the quality checks:

```bash
pytest
ruff check .
ruff format --check .
```

The same test, lint, and formatting checks run in GitHub Actions for pushes and
pull requests on Python 3.12.

The architectural direction and milestone boundaries are described in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Current invariants are summarized in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

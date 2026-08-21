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

## Optional local inference

The base installation remains lightweight and does not install an inference
engine. To use the llama.cpp adapter, install Forge's bounded `llama` extra:

```bash
python -m pip install -e ".[llama]"
```

Forge is validated against `llama-cpp-python` 0.3.35 and accepts compatible
0.3.x updates; the `<0.4` bound prevents an unreviewed breaking release from
changing the adapter contract.

On Apple Silicon, first confirm that both the machine and Python report `arm64`:

```bash
uname -m
python -c 'import platform; print(platform.machine())'
```

The official `llama-cpp-python` project offers Metal wheels for macOS and
Python 3.12. To explicitly select that wheel source when installing, use:

```bash
python -m pip install -e .
python -m pip install "llama-cpp-python>=0.3.35,<0.4" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

Alternatively, a source build can enable Metal with
`CMAKE_ARGS="-DGGML_METAL=on"`. Runtime GPU offload also requires a nonzero
`gpu_layers` value; Forge defaults to `-1`, requesting all supported layers.
Backend initialization logs are the source of truth for whether Metal and GPU
offload are active.

Download a compatible GGUF file manually and retain ownership of its location.
Forge neither downloads nor modifies models. The A2 reference is
`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, Q4_K_M quantization, but the model and
path are not hard-coded. Local `*.gguf` files are ignored by Git.

Run one non-interactive smoke request:

```bash
python scripts/smoke_model.py /path/to/model.gguf \
  --model-id Qwen2.5-Coder-7B-Instruct --verbose
```

Or run the opt-in integration test:

```bash
FORGE_TEST_MODEL=/path/to/model.gguf pytest -m integration
```

A2 does not provide an interactive `forge chat` command.

The architectural direction and milestone boundaries are described in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Current invariants are summarized in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

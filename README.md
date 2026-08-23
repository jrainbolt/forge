# Forge

Forge is an early-stage, local-first AI coding assistant and software-development
agent. Its long-term design keeps language models interchangeable and keeps the
core useful without cloud services. The current implementation provides the
project bootstrap, a generic backend-independent model API, named model
profiles, an optional llama.cpp adapter for real local inference, ephemeral
multi-turn chat sessions, and permission-controlled repository-aware chat using
workspace-confined read-only tools.

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

Download compatible GGUF files manually and retain ownership of their location.
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

## Named model profiles

Copy [`forge.example.toml`](forge.example.toml) to a location outside the
repository and replace its model paths. The example contains the two A3
reference profiles, `qwen-small` and `qwen-large`. Keep GGUF weights and the
machine-specific configuration outside version control. A recommended local
layout is `~/Models/forge/` for both the weights and an untracked `forge.toml`.

Inspect profiles without loading model weights:

```bash
python scripts/list_models.py /path/to/forge.toml
```

Run the same A3 coding request through either profile. `--config` takes an
explicit path; `FORGE_CONFIG` is the application-level environment override.

```bash
python scripts/smoke_profile.py qwen-small --config /path/to/forge.toml
FORGE_CONFIG=/path/to/forge.toml python scripts/smoke_profile.py qwen-large
```

The catalog validates all profile and backend settings while loading TOML, but
constructs only the selected model. Load and generation timings are written to
standard error.
The first two-profile observations are recorded in
[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## Interactive chat

Launch a local session with an explicit profile and configuration:

```bash
forge chat --model qwen-small --config ~/Models/forge/forge.toml
FORGE_CONFIG=~/Models/forge/forge.toml forge chat --model qwen-large
```

Interactive generation defaults to 256 output tokens and temperature 0.4.
Generic overrides are available through `--max-tokens`, `--temperature`, and
`--seed`; `--no-system` omits the concise default Forge system message.

The REPL supports:

```text
/help   show commands
/clear  clear in-memory conversational turns without reloading the model
/info   show profile, model, context, and estimated-budget information
/exit   close the model and exit
```

Ctrl-D and Ctrl-C also exit cleanly. The selected model is loaded once and
reused for every turn in the session. Responses are synchronous and appear
after generation completes. Conversation history exists only in memory and is
not saved or resumable.

Without an explicit workspace, Forge chat has no tools or repository access.
It remains a general local conversation interface; model answers do not grant
external capabilities.

To enable repository-aware chat, select the workspace explicitly:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace .
```

Without `--workspace`, chat retains its normal tool-free behavior. With a
workspace, Forge may list directories, read bounded UTF-8 files, perform bounded
lexical searches, inspect Git status, and read working-tree or staged diffs.
The model uses strict backend-independent JSON output specifications, while
every request still passes through the deny-by-default permission policy and
central executor. Discovery results identify candidates; implementation claims
require relevant source-file evidence, and Git output counts only as working-
state evidence.
Internal tool calls and repository contents are ephemeral and are not saved in
conversation history or on disk. Repository mode requires model-requested tool
evidence before accepting a final answer and permits only one bounded protocol
correction per turn.

Repository mode is read-only. Forge still cannot edit files, apply patches, run
shell commands, builds or tests, mutate Git, access the network, or use the web.

Run the controlled read-only coding evaluation locally with:

```bash
forge eval \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1
```

An optional versioned JSON report is written only when requested:

```bash
forge eval \
  --model qwen-large \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1 \
  --output eval-results/qwen-large.json
```

Evaluations use the committed fixture and the same production read-only
repository orchestration. Real-model runs remain local and are not part of CI.
See [`docs/EVALUATION.md`](docs/EVALUATION.md) for tasks, scoring, and report
details.

An opt-in non-interactive repository smoke request is also available:

```bash
python scripts/smoke_repo_chat.py qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --question "Where is the generic Model abstraction defined?"
```

The architectural direction and milestone boundaries are described in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Current invariants are summarized in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

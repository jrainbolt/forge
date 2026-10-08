"""Source-free request-time identity for authoritative A74 pairing."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from forge.models import (
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)


class PairMismatch(StrEnum):
    TASK_IDENTITY_MISMATCH = "TASK_IDENTITY_MISMATCH"
    MODEL_CONFIG_MISMATCH = "MODEL_CONFIG_MISMATCH"
    GROUNDING_SOURCE_SET_MISMATCH = "GROUNDING_SOURCE_SET_MISMATCH"
    SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
    AUTHORITY_PATH_MISMATCH = "AUTHORITY_PATH_MISMATCH"
    AUTHORIZED_RANGE_MISMATCH = "AUTHORIZED_RANGE_MISMATCH"
    CREATE_AUTHORITY_MISMATCH = "CREATE_AUTHORITY_MISMATCH"
    WORKSPACE_GENERATION_MISMATCH = "WORKSPACE_GENERATION_MISMATCH"
    REPRESENTATION_MISMATCH = "REPRESENTATION_MISMATCH"
    OTHER_PAIRED_INPUT_MISMATCH = "OTHER_PAIRED_INPUT_MISMATCH"


@dataclass(frozen=True, slots=True)
class PairedInputIdentity:
    identity_version: int
    common_identity: str
    task_identity: str
    model_config_identity: str
    generation_identity: str
    grounding_source_set_identity: str
    source_hash_identity: str
    authority_path_identity: str
    authorized_range_identity: str
    create_authority_identity: str
    workspace_generation_identity: str
    representation_identity: str
    evidence_plan_identity: str


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _path_blocks(request: ModelRequest) -> tuple[tuple[str, tuple[str, ...]], ...]:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    path: str | None = None
    content: list[str] = []
    for message in request.messages:
        if message.content.startswith("PATH: "):
            path = message.content.removeprefix("PATH: ").strip()
            content = []
        elif path is not None and message.content == f"END FILE: {path}":
            blocks.append((path, tuple(content)))
            path = None
            content = []
        elif path is not None:
            content.append(message.content)
    return tuple(blocks)


def _authorized_paths(request: ModelRequest) -> tuple[str, ...]:
    create_marker = "Authorized new paths: "
    for message in request.messages:
        if create_marker in message.content:
            return tuple(
                sorted(
                    path.strip()
                    for path in message.content.split(create_marker, 1)[1].split(",")
                    if path.strip()
                )
            )
    marker = "Current authorized mutation targets:\n"
    for message in request.messages:
        if marker not in message.content:
            continue
        targets = message.content.split(marker, 1)[1].split("\n\n", 1)[0]
        paths = tuple(
            sorted(
                line.removeprefix("MODIFY ").removeprefix("CREATE ").strip()
                for line in targets.splitlines()
                if line.strip()
            )
        )
        if paths:
            return paths
    return ()


def _create_paths(request: ModelRequest) -> tuple[str, ...]:
    marker = "Current authorized mutation targets:\n"
    for message in request.messages:
        if marker in message.content:
            targets = message.content.split(marker, 1)[1].split("\n\n", 1)[0]
            paths = tuple(
                sorted(
                    line.removeprefix("CREATE ").strip()
                    for line in targets.splitlines()
                    if line.startswith("CREATE ")
                )
            )
            if paths:
                return paths
    create_marker = "Authorized new paths: "
    for message in request.messages:
        if create_marker in message.content:
            return tuple(
                sorted(
                    path.strip()
                    for path in message.content.split(create_marker, 1)[1].split(",")
                    if path.strip()
                )
            )
    return ()


def _schema_generation(value: object) -> tuple[object, ...]:
    found: list[object] = []

    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return tuple(thaw(child) for child in item)
        return item

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in {"workspace_generation", "generation"}:
                    found.append(thaw(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def capture_paired_input(
    request: ModelRequest,
    *,
    task_identity: str,
    model_config_identity: str,
    context_size: int,
    representation: str,
) -> PairedInputIdentity:
    blocks = _path_blocks(request)
    paths = tuple(path for path, _ in blocks)
    source_hashes = tuple((path, _hash(messages)) for path, messages in blocks)
    ranges = tuple(
        (
            path,
            _hash(
                tuple(
                    numbers
                    for message in messages
                    for numbers in [
                        tuple(
                            int(match.group(1))
                            for match in re.finditer(r"(?m)^\s*(\d+)\s+\|", message)
                        )
                    ]
                    if numbers
                )
            ),
        )
        for path, messages in blocks
    )
    authorized = _authorized_paths(request)
    create_paths = _create_paths(request)
    components = {
        "task_identity": task_identity,
        "model_config_identity": model_config_identity,
        "generation_identity": _hash(
            {
                "seed": request.generation.seed,
                "temperature": request.generation.temperature,
                "output_budget": request.generation.max_tokens,
                "context_size": context_size,
            }
        ),
        "grounding_source_set_identity": _hash(paths),
        "source_hash_identity": _hash(source_hashes),
        "authority_path_identity": _hash(authorized),
        "authorized_range_identity": _hash(ranges),
        "create_authority_identity": _hash(
            {"create_paths": create_paths, "trusted_parents": paths}
        ),
        "workspace_generation_identity": _hash(
            _schema_generation(request.output.schema)
        ),
        "representation_identity": _hash(representation),
        "evidence_plan_identity": _hash(
            {"paths": paths, "source_hashes": source_hashes}
        ),
    }
    return PairedInputIdentity(
        1,
        _hash(components),
        **components,
    )


def compare_paired_inputs(
    left: PairedInputIdentity | dict[str, Any],
    right: PairedInputIdentity | dict[str, Any],
) -> tuple[PairMismatch, ...]:
    first = asdict(left) if isinstance(left, PairedInputIdentity) else left
    second = asdict(right) if isinstance(right, PairedInputIdentity) else right
    fields = (
        ("task_identity", PairMismatch.TASK_IDENTITY_MISMATCH),
        ("model_config_identity", PairMismatch.MODEL_CONFIG_MISMATCH),
        ("generation_identity", PairMismatch.OTHER_PAIRED_INPUT_MISMATCH),
        ("grounding_source_set_identity", PairMismatch.GROUNDING_SOURCE_SET_MISMATCH),
        ("source_hash_identity", PairMismatch.SOURCE_HASH_MISMATCH),
        ("authority_path_identity", PairMismatch.AUTHORITY_PATH_MISMATCH),
        ("authorized_range_identity", PairMismatch.AUTHORIZED_RANGE_MISMATCH),
        ("create_authority_identity", PairMismatch.CREATE_AUTHORITY_MISMATCH),
        (
            "workspace_generation_identity",
            PairMismatch.WORKSPACE_GENERATION_MISMATCH,
        ),
        ("representation_identity", PairMismatch.REPRESENTATION_MISMATCH),
        ("evidence_plan_identity", PairMismatch.OTHER_PAIRED_INPUT_MISMATCH),
    )
    return tuple(
        reason for field, reason in fields if first.get(field) != second.get(field)
    )


class PairedInputCaptureModel(Model):
    """Capture the immutable primary request before an experimental adapter."""

    def __init__(
        self,
        backend: Model,
        *,
        task_identity: str,
        model_config_identity: str,
        context_size: int,
        representation: str,
    ) -> None:
        self.backend = backend
        self.task_identity = task_identity
        self.model_config_identity = model_config_identity
        self.context_size = context_size
        self.representation = representation
        self.record: PairedInputIdentity | None = None

    @property
    def identity(self) -> ModelIdentity:
        return self.backend.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = "\n".join(message.content for message in request.messages)
        primary = (
            request.output.schema is not None
            and "Current authorized mutation targets:" in text
            and "previous mutation failed verification" not in text
            and "Repair evidence is ready" not in text
        )
        if primary and self.record is None:
            self.record = capture_paired_input(
                request,
                task_identity=self.task_identity,
                model_config_identity=self.model_config_identity,
                context_size=self.context_size,
                representation=self.representation,
            )
        return self.backend.generate(request)

    def close(self) -> None:
        self.backend.close()

"""Evaluator-only deterministic MutationContract adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from forge.evidence_coverage import decompose_evidence_plan
from forge.models import (
    Message,
    MessageRole,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)

MAX_BEHAVIOR_GOALS = 4
MAX_COMPONENT_ROLES = 4
MAX_INVARIANTS = 3
_CODE = re.compile(r"(?:```|\bdef\s+|#include|\bclass\s+|\breturn\s+|;\s*$)")
_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*[A-Za-z0-9_-]"
)


@dataclass(frozen=True, slots=True)
class MutationContract:
    behavior_goals: tuple[str, ...]
    required_components: tuple[str, ...]
    required_operation_roles: tuple[str, ...]
    invariants: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContractRecord:
    classification: str
    unavailable_reason: str | None
    contract_id: str | None
    behavior_goal_count: int
    component_count: int
    operation_role_count: int
    invariant_count: int


def _authority(request: ModelRequest) -> tuple[tuple[str, ...], tuple[str, ...]]:
    create_marker = "Authorized new paths: "
    for message in request.messages:
        if create_marker in message.content:
            paths = tuple(
                path.strip()
                for path in message.content.split(create_marker, 1)[1].split(",")
                if path.strip()
            )
            return paths, tuple(f"CREATE {path}" for path in paths)
    marker = "Current authorized mutation targets:\n"
    for message in request.messages:
        if marker not in message.content:
            continue
        lines = message.content.split(marker, 1)[1].split("\n\n", 1)[0].splitlines()
        components = []
        roles = []
        for line in lines:
            value = line.strip()
            if not value:
                continue
            if value.startswith("MODIFY "):
                path = value.removeprefix("MODIFY ")
                role = "MODIFY"
            elif value.startswith("CREATE "):
                path = value.removeprefix("CREATE ")
                role = "CREATE"
            else:
                path = value
                role = "MODIFY"
            components.append(path)
            roles.append(f"{role} {path}")
        if components:
            return tuple(components), tuple(roles)
    return (), ()


def _grounded_paths(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                message.content.removeprefix("PATH: ").strip()
                for message in request.messages
                if message.content.startswith("PATH: ")
            }
        )
    )


def build_contract(
    task: str, request: ModelRequest, trusted_components: frozenset[str]
) -> MutationContract:
    plan = decompose_evidence_plan(task)
    goals = tuple(goal.description for goal in plan.goals if goal.required)
    authorized, roles = _authority(request)
    grounded = _grounded_paths(request)
    components = tuple(dict.fromkeys((*authorized, *grounded)))
    visible = trusted_components | frozenset(grounded)
    checks = (
        (not goals, "NO_BEHAVIOR_GOALS"),
        (len(goals) > MAX_BEHAVIOR_GOALS, "BEHAVIOR_GOAL_BOUND"),
        (not components, "NO_AUTHORITY"),
        (len(components) > MAX_COMPONENT_ROLES, "COMPONENT_BOUND"),
        (len(roles) > MAX_COMPONENT_ROLES, "OPERATION_ROLE_BOUND"),
        (not set(authorized).issubset(trusted_components), "UNTRUSTED_AUTHORITY"),
        (not set(components).issubset(visible), "UNTRUSTED_COMPONENT"),
    )
    for failed, reason in checks:
        if failed:
            raise ValueError(reason)
    contract = MutationContract(goals, components, roles)
    text = "\n".join((*contract.behavior_goals, *contract.invariants))
    if _CODE.search(text):
        raise ValueError("CODE_CONTENT")
    if not set(_PATH.findall(text)).issubset(visible):
        raise ValueError("UNTRUSTED_PATH_REFERENCE")
    return contract


class ContractModel(Model):
    """Append one deterministic contract to the primary mutation request."""

    def __init__(
        self, backend: Model, task: str, trusted_components: frozenset[str]
    ) -> None:
        self.backend = backend
        self.task = task
        self.trusted_components = trusted_components
        self.record: ContractRecord | None = None

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
        if not primary or self.record is not None:
            return self.backend.generate(request)
        try:
            contract = build_contract(self.task, request, self.trusted_components)
        except ValueError as error:
            self.record = ContractRecord(
                "CONTRACT_UNAVAILABLE", str(error), None, 0, 0, 0, 0
            )
            return self.backend.generate(request)
        payload = {
            "behavior_goals": contract.behavior_goals,
            "required_components": contract.required_components,
            "required_operation_roles": contract.required_operation_roles,
            "invariants": contract.invariants,
        }
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.record = ContractRecord(
            "VALID",
            None,
            hashlib.sha256(normalized.encode()).hexdigest(),
            len(contract.behavior_goals),
            len(contract.required_components),
            len(contract.required_operation_roles),
            len(contract.invariants),
        )
        message = Message(
            MessageRole.USER,
            "Deterministic grounding-to-mutation contract (obligations only):\n"
            + json.dumps(payload, sort_keys=True),
        )
        return self.backend.generate(
            replace(request, messages=(*request.messages, message))
        )

    def close(self) -> None:
        self.backend.close()

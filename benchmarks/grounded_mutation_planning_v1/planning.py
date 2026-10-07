"""Bounded evaluator-only implementation-plan model adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum

from forge.models import (
    Message,
    MessageRole,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    OutputSpecification,
    ResponseFormat,
)


class PlanClassification(StrEnum):
    VALID = "VALID"
    PLAN_INVALID = "PLAN_INVALID"
    PLAN_INCOMPLETE = "PLAN_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class ImplementationPlan:
    required_behavior: tuple[str, ...]
    affected_components: tuple[str, ...]
    required_changes: tuple[str, ...]
    invariants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanRecord:
    classification: str
    plan_id: str | None
    required_behavior_count: int
    affected_component_count: int
    required_change_count: int
    invariant_count: int


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "required_behavior": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "affected_components": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "required_changes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "invariants": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
    "required": [
        "required_behavior",
        "affected_components",
        "required_changes",
        "invariants",
    ],
    "additionalProperties": False,
}

_CODE = re.compile(
    r"(?:```|\bdef\s+|#include|\bclass\s+|\breturn\s+|\{[^}]*\}|;\s*$)",
    re.MULTILINE,
)
_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")


def validate_plan(
    payload: object, trusted_components: frozenset[str]
) -> ImplementationPlan:
    if not isinstance(payload, dict) or set(payload) != {
        "required_behavior",
        "affected_components",
        "required_changes",
        "invariants",
    }:
        raise ValueError(PlanClassification.PLAN_INVALID.value)

    def items(name: str, minimum: int, maximum: int) -> tuple[str, ...]:
        value = payload[name]
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            raise ValueError(PlanClassification.PLAN_INCOMPLETE.value)
        normalized = tuple(item.strip() for item in value if isinstance(item, str))
        if len(normalized) != len(value) or any(
            not item or len(item) > 240 for item in normalized
        ):
            raise ValueError(PlanClassification.PLAN_INVALID.value)
        return normalized

    behavior = items("required_behavior", 1, 3)
    components = items("affected_components", 1, 4)
    changes = items("required_changes", 1, 4)
    invariants = items("invariants", 0, 3)
    if not set(components).issubset(trusted_components):
        raise ValueError(PlanClassification.PLAN_INVALID.value)
    text = "\n".join((*behavior, *changes, *invariants))
    if _CODE.search(text):
        raise ValueError(PlanClassification.PLAN_INVALID.value)
    referenced_paths = set(_PATH.findall(text))
    if not referenced_paths.issubset(trusted_components):
        raise ValueError(PlanClassification.PLAN_INVALID.value)
    return ImplementationPlan(behavior, components, changes, invariants)


class PlanningModel(Model):
    """Generate one plan before the primary mutation-ready request only."""

    def __init__(self, backend: Model, trusted_components: frozenset[str]) -> None:
        self.backend = backend
        self.trusted_components = trusted_components
        self.record: PlanRecord | None = None
        self._plan: ImplementationPlan | None = None
        self._invalid_response: ModelResponse | None = None

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
        if not _is_primary_mutation_request(request):
            return self.backend.generate(request)
        if self._invalid_response is not None:
            return self._invalid_response
        if self.record is None:
            planning_request = ModelRequest(
                (
                    Message(
                        MessageRole.SYSTEM,
                        "Produce only a concise implementation-intent plan. Do not "
                        "include code, patches, hidden reasoning, or new paths.",
                    ),
                    *request.messages[1:],
                    Message(
                        MessageRole.USER,
                        "Return the bounded ImplementationPlan JSON now.",
                    ),
                ),
                request.generation,
                OutputSpecification(ResponseFormat.JSON, PLAN_SCHEMA),
            )
            response = self.backend.generate(planning_request)
            try:
                payload = json.loads(response.text)
                self._plan = validate_plan(payload, self.trusted_components)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                classification = (
                    str(error)
                    if str(error) in {item.value for item in PlanClassification}
                    else PlanClassification.PLAN_INVALID.value
                )
                self.record = PlanRecord(classification, None, 0, 0, 0, 0)
                self._invalid_response = response
                return response
            normalized = json.dumps(
                {
                    "required_behavior": self._plan.required_behavior,
                    "affected_components": self._plan.affected_components,
                    "required_changes": self._plan.required_changes,
                    "invariants": self._plan.invariants,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self.record = PlanRecord(
                PlanClassification.VALID.value,
                hashlib.sha256(normalized.encode()).hexdigest(),
                len(self._plan.required_behavior),
                len(self._plan.affected_components),
                len(self._plan.required_changes),
                len(self._plan.invariants),
            )
        if self._plan is None:
            return self.backend.generate(request)
        plan_message = Message(
            MessageRole.USER,
            "Bounded implementation plan (intent only):\n"
            + json.dumps(
                {
                    "required_behavior": self._plan.required_behavior,
                    "affected_components": self._plan.affected_components,
                    "required_changes": self._plan.required_changes,
                    "invariants": self._plan.invariants,
                },
                sort_keys=True,
            ),
        )
        return self.backend.generate(
            replace(request, messages=(*request.messages, plan_message))
        )

    def close(self) -> None:
        self.backend.close()


def _is_primary_mutation_request(request: ModelRequest) -> bool:
    if (
        request.output.schema is None
        or request.messages[0].role is not MessageRole.SYSTEM
    ):
        return False
    text = "\n".join(message.content for message in request.messages)
    return (
        "Current authorized mutation targets:" in text
        and "previous mutation failed verification" not in text
        and "Repair evidence is ready" not in text
    )

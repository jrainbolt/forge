"""Immutable autonomy and permission policy composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from forge.tools.permissions import PermissionPolicy
from forge.tools.tool import Tool
from forge.tools.types import (
    ExecutionContext,
    PermissionDecision,
    ToolCapability,
    ToolInvocation,
    ToolMetadata,
)


class AutonomyMode(Enum):
    CHAT = "chat"
    READ = "read"
    ASSIST = "assist"
    AGENT = "agent"
    REPAIR = "repair"

    @property
    def capabilities(self) -> frozenset[ToolCapability]:
        if self is AutonomyMode.CHAT:
            return frozenset()
        if self is AutonomyMode.READ:
            return frozenset({ToolCapability.READ})
        return frozenset(
            {
                ToolCapability.READ,
                ToolCapability.WRITE,
                ToolCapability.BUILD,
                ToolCapability.TEST,
            }
        )

    @property
    def repository_mode(self) -> bool:
        return self is not AutonomyMode.CHAT

    @property
    def coding_mode(self) -> bool:
        return self in {AutonomyMode.ASSIST, AutonomyMode.AGENT, AutonomyMode.REPAIR}

    @property
    def agent_mode(self) -> bool:
        return self in {AutonomyMode.AGENT, AutonomyMode.REPAIR}


@dataclass(frozen=True, slots=True)
class PermissionProfile:
    name: str
    read: PermissionDecision
    write: PermissionDecision
    build: PermissionDecision
    test: PermissionDecision
    default: PermissionDecision = PermissionDecision.DENY

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("permission profile name must be non-empty text")
        decisions = (self.read, self.write, self.build, self.test, self.default)
        if not all(isinstance(value, PermissionDecision) for value in decisions):
            raise TypeError("permission profile values must be PermissionDecision")
        if self.write is PermissionDecision.ALLOW:
            raise ValueError("permission profiles may not automatically allow writes")

    def decision(self, capability: ToolCapability) -> PermissionDecision:
        return {
            ToolCapability.READ: self.read,
            ToolCapability.WRITE: self.write,
            ToolCapability.BUILD: self.build,
            ToolCapability.TEST: self.test,
        }.get(capability, self.default)


SAFE_PROFILE = PermissionProfile(
    "safe",
    PermissionDecision.ALLOW,
    PermissionDecision.DENY,
    PermissionDecision.DENY,
    PermissionDecision.DENY,
)
CONFIRM_PROFILE = PermissionProfile(
    "confirm",
    PermissionDecision.ALLOW,
    PermissionDecision.ASK,
    PermissionDecision.ASK,
    PermissionDecision.ASK,
)
TRUSTED_EXEC_PROFILE = PermissionProfile(
    "trusted-exec",
    PermissionDecision.ALLOW,
    PermissionDecision.ASK,
    PermissionDecision.ALLOW,
    PermissionDecision.ALLOW,
)
BUILTIN_PERMISSION_PROFILES = MappingProxyType(
    {
        profile.name: profile
        for profile in (SAFE_PROFILE, CONFIRM_PROFILE, TRUSTED_EXEC_PROFILE)
    }
)


@dataclass(frozen=True, slots=True)
class InteractionPolicy(PermissionPolicy):
    autonomy_mode: AutonomyMode
    permission_profile: PermissionProfile

    def __post_init__(self) -> None:
        if not isinstance(self.autonomy_mode, AutonomyMode):
            raise TypeError("autonomy_mode must be an AutonomyMode")
        if not isinstance(self.permission_profile, PermissionProfile):
            raise TypeError("permission_profile must be a PermissionProfile")

    def decision_for(self, capability: ToolCapability) -> PermissionDecision:
        if capability not in self.autonomy_mode.capabilities:
            return PermissionDecision.DENY
        return self.permission_profile.decision(capability)

    def exposes(self, metadata: ToolMetadata) -> bool:
        return self.decision_for(metadata.capability) is not PermissionDecision.DENY

    def evaluate(
        self, tool: Tool, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        return self.decision_for(tool.metadata.capability)


def default_profile_for(mode: AutonomyMode) -> PermissionProfile:
    return (
        SAFE_PROFILE
        if mode in {AutonomyMode.CHAT, AutonomyMode.READ}
        else CONFIRM_PROFILE
    )


def resolve_interaction_policy(
    mode: AutonomyMode | str,
    profile: PermissionProfile | str | None = None,
) -> InteractionPolicy:
    try:
        resolved_mode = mode if isinstance(mode, AutonomyMode) else AutonomyMode(mode)
    except (TypeError, ValueError) as error:
        available = ", ".join(item.value for item in AutonomyMode)
        raise ValueError(
            f"unknown autonomy mode {mode!r}; available: {available}"
        ) from error
    if profile is None:
        resolved_profile = default_profile_for(resolved_mode)
    elif isinstance(profile, PermissionProfile):
        resolved_profile = profile
    else:
        try:
            resolved_profile = BUILTIN_PERMISSION_PROFILES[profile]
        except (KeyError, TypeError) as error:
            available = ", ".join(BUILTIN_PERMISSION_PROFILES)
            raise ValueError(
                f"unknown permission profile {profile!r}; available: {available}"
            ) from error
    return InteractionPolicy(resolved_mode, resolved_profile)

"""Deterministic repository observation admission and active-context planning."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from forge.conversation import ConservativeTokenEstimator
from forge.models import Message, MessageRole
from forge.tools.paths import WorkspacePathError, resolve_workspace_path
from forge.tools.types import ToolEvidence, ToolResult, ToolResultStatus

LOGGER = logging.getLogger(__name__)
DEFAULT_CONTEXT_CAPACITY = 4096
DEFAULT_SAFETY_RESERVE = 64
DEFAULT_WINDOW_MARGIN = 20
MAX_SOURCE_WINDOW_LINES = 400
OBSERVATION_FRAMING_TOKENS = 32


class ObservationType(Enum):
    SEARCH_DISCOVERY = "search_discovery"
    SEMANTIC_DISCOVERY = "semantic_discovery"
    SYMBOL_DISCOVERY = "symbol_discovery"
    REFERENCE_DISCOVERY = "reference_discovery"
    OUTLINE_DISCOVERY = "outline_discovery"
    SOURCE_RANGE = "source_range"
    SOURCE_FILE = "source_file"
    GIT_STATE = "git_state"
    WRITE_RESULT = "write_result"
    BUILD_RESULT = "build_result"
    TEST_RESULT = "test_result"
    OTHER = "other"


class ContextAdmissionStatus(Enum):
    ADMIT = "admit"
    ADMIT_WITH_COMPACTION = "admit_with_compaction"
    REJECT_TOO_LARGE = "reject_too_large"
    REJECT_REDUNDANT = "reject_redundant"


@dataclass(frozen=True, slots=True)
class ContextBudget:
    model_capacity: int
    reserved_output: int
    safety_reserve: int
    system_cost: int
    tool_definition_cost: int
    durable_conversation_cost: int
    active_task_cost: int
    retained_observation_cost: int

    @property
    def remaining_estimated_tokens(self) -> int:
        used = (
            self.reserved_output
            + self.safety_reserve
            + self.system_cost
            + self.tool_definition_cost
            + self.durable_conversation_cost
            + self.active_task_cost
            + self.retained_observation_cost
        )
        return max(0, self.model_capacity - used)


@dataclass(frozen=True, slots=True)
class SourceWindow:
    path: str
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ContextAdmissionDecision:
    status: ContextAdmissionStatus
    estimated_tokens: int = 0
    message: str | None = None
    recommendation: SourceWindow | None = None

    @property
    def admitted(self) -> bool:
        return self.status in {
            ContextAdmissionStatus.ADMIT,
            ContextAdmissionStatus.ADMIT_WITH_COMPACTION,
        }


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    observation_id: str
    tool_name: str
    observation_type: ObservationType
    evidence_type: str
    paths: tuple[str, ...]
    generation: int
    estimated_tokens: int
    utf8_bytes: int
    text_lines: int
    retention_priority: int
    supersedes: tuple[str, ...] = ()
    compacted: bool = False


@dataclass(slots=True)
class _ActiveObservation:
    record: ObservationRecord
    assistant: Message
    result: Message


@dataclass(frozen=True, slots=True)
class ContextPlannerMetrics:
    estimated_context_peak: int = 0
    estimated_context_admitted: int = 0
    estimated_context_dropped: int = 0
    observations_compacted: int = 0
    context_rejections: int = 0
    whole_file_reads: int = 0
    range_reads: int = 0
    final_remaining_budget: int = 0


class ContextPlanner:
    """Own task observation history separately from the active model snapshot."""

    def __init__(
        self,
        *,
        model_capacity: int | None,
        reserved_output: int,
        safety_reserve: int = DEFAULT_SAFETY_RESERVE,
        estimator: ConservativeTokenEstimator | None = None,
    ) -> None:
        self.model_capacity = model_capacity or DEFAULT_CONTEXT_CAPACITY
        self.reserved_output = reserved_output
        self.safety_reserve = safety_reserve
        self.estimator = estimator or ConservativeTokenEstimator()
        self._active: list[_ActiveObservation] = []
        self._history: list[ObservationRecord] = []
        self._known_paths: set[str] = set()
        self._exact_windows: dict[str, SourceWindow] = {}
        self._read_generations: set[tuple[str, str, int]] = set()
        self._search_succeeded = False
        self._source_since_search = False
        self._targeted_failed = False
        self._admitted = 0
        self._dropped = 0
        self._compacted = 0
        self._rejections = 0
        self._whole_reads = 0
        self._range_reads = 0
        self._peak = 0
        self._last_remaining = 0

    @property
    def history(self) -> tuple[ObservationRecord, ...]:
        return tuple(self._history)

    @property
    def active_messages(self) -> tuple[Message, ...]:
        messages: list[Message] = []
        for observation in self._active:
            messages.extend((observation.assistant, observation.result))
        return tuple(messages)

    @property
    def active_estimated_tokens(self) -> int:
        return sum(item.record.estimated_tokens for item in self._active)

    @property
    def metrics(self) -> ContextPlannerMetrics:
        return ContextPlannerMetrics(
            self._peak,
            self._admitted,
            self._dropped,
            self._compacted,
            self._rejections,
            self._whole_reads,
            self._range_reads,
            self._last_remaining,
        )

    def budget(
        self,
        *,
        system_cost: int,
        tool_definition_cost: int,
        durable_conversation_cost: int,
        active_task_cost: int,
    ) -> ContextBudget:
        retained = sum(item.record.estimated_tokens for item in self._active)
        budget = ContextBudget(
            self.model_capacity,
            self.reserved_output,
            self.safety_reserve,
            system_cost,
            tool_definition_cost,
            durable_conversation_cost,
            active_task_cost,
            retained,
        )
        self._last_remaining = budget.remaining_estimated_tokens
        used = self.model_capacity - self._last_remaining
        self._peak = max(self._peak, used)
        return budget

    def compact_to_fit(self, maximum_observation_tokens: int) -> bool:
        """Compact low-priority payloads without dropping current source evidence."""
        if self.active_estimated_tokens <= maximum_observation_tokens:
            return True
        for item in sorted(
            self._active, key=lambda value: value.record.retention_priority
        ):
            if item.record.retention_priority >= 85:
                continue
            self._compact(item, "active context budget pressure")
            if self.active_estimated_tokens <= maximum_observation_tokens:
                return True
        for item in tuple(self._active):
            if item.record.compacted and item.record.retention_priority < 85:
                self._drop(item, "active context budget pressure")
                if self.active_estimated_tokens <= maximum_observation_tokens:
                    return True
        return self.active_estimated_tokens <= maximum_observation_tokens

    def preflight(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        workspace: Path,
        *,
        remaining_tokens: int,
        generation: int,
    ) -> ContextAdmissionDecision:
        if tool_name == "repository.search_files":
            if (
                self._search_succeeded
                and self._known_paths
                and not self._source_since_search
                and not self._targeted_failed
            ):
                return self._reject(
                    ContextAdmissionStatus.REJECT_REDUNDANT,
                    "Broad discovery already produced candidate paths. Use "
                    "repository.find_symbol, repository.file_outline, or "
                    "repository.read_range before searching again.",
                )
            return ContextAdmissionDecision(ContextAdmissionStatus.ADMIT)
        if tool_name not in {"repository.read_file", "repository.read_range"}:
            return ContextAdmissionDecision(ContextAdmissionStatus.ADMIT)
        requested = arguments.get("path")
        if not isinstance(requested, str):
            return ContextAdmissionDecision(ContextAdmissionStatus.ADMIT)
        if (tool_name, requested, generation) in self._read_generations:
            return self._reject(
                ContextAdmissionStatus.REJECT_REDUNDANT,
                "This unchanged source file is already present in active context. "
                "Use the current evidence or inspect a distinct relevant path.",
                recommendation=self._exact_windows.get(requested),
            )
        try:
            path = resolve_workspace_path(workspace, requested)
            size = path.stat().st_size
        except (OSError, WorkspacePathError):
            return ContextAdmissionDecision(ContextAdmissionStatus.ADMIT)
        if tool_name == "repository.read_range":
            start = arguments.get("start_line")
            end = arguments.get("end_line")
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                fraction = min(1.0, (end - start + 1) / max(1, _line_count(path)))
                size = max(1, math.ceil(size * fraction))
        estimated = estimate_bytes(size) + OBSERVATION_FRAMING_TOKENS
        if estimated > remaining_tokens:
            alternatives = (
                "Use a smaller repository.read_range request."
                if tool_name == "repository.read_range"
                else "Use repository.file_outline, repository.find_symbol, or "
                "repository.read_range."
            )
            return self._reject(
                ContextAdmissionStatus.REJECT_TOO_LARGE,
                "The requested source would exceed the remaining context budget. "
                + alternatives,
                estimated,
                self._exact_windows.get(requested),
            )
        return ContextAdmissionDecision(ContextAdmissionStatus.ADMIT, estimated)

    def register(
        self,
        *,
        assistant_text: str,
        rendered_result: str,
        result: ToolResult,
        evidence: ToolEvidence,
        arguments: Mapping[str, object],
        generation: int,
        assistant_role: MessageRole = MessageRole.ASSISTANT,
    ) -> None:
        rendered_result = self._enrich_windows(result, rendered_result)
        observation_type = _observation_type(result.tool_name)
        paths = _result_paths(result, arguments)
        if result.status is ToolResultStatus.SUCCESS:
            self._known_paths.update(paths)
            if result.tool_name == "repository.search_files" and paths:
                self._search_succeeded = True
                self._source_since_search = False
            if result.tool_name in {
                "repository.file_outline",
                "repository.find_symbol",
                "repository.find_references",
                "repository.read_range",
            }:
                self._targeted_failed = False
            if observation_type in {
                ObservationType.SOURCE_FILE,
                ObservationType.SOURCE_RANGE,
            }:
                self._source_since_search = True
                for path in paths:
                    self._read_generations.add((result.tool_name, path, generation))
            if observation_type is ObservationType.SOURCE_FILE:
                self._whole_reads += 1
            elif observation_type is ObservationType.SOURCE_RANGE:
                self._range_reads += 1
        elif result.tool_name in {
            "repository.file_outline",
            "repository.find_symbol",
            "repository.find_references",
            "repository.read_range",
        }:
            self._targeted_failed = True

        result_message = Message(MessageRole.USER, rendered_result)
        assistant_message = Message(assistant_role, assistant_text)
        cost = self.estimator.estimate(result_message) + self.estimator.estimate(
            assistant_message
        )
        record = ObservationRecord(
            result.invocation_id,
            result.tool_name,
            observation_type,
            evidence.value,
            paths,
            generation,
            cost,
            len(rendered_result.encode("utf-8")),
            rendered_result.count("\n") + 1,
            _priority(observation_type, result.status),
        )
        self._history.append(record)
        active = _ActiveObservation(record, assistant_message, result_message)
        self._active.append(active)
        self._admitted += cost
        self._apply_supersession(record)
        if (
            observation_type
            in {ObservationType.BUILD_RESULT, ObservationType.TEST_RESULT}
            and result.status is ToolResultStatus.SUCCESS
        ):
            self._compact(active, "successful verification retained as metadata")

    def mutation_succeeded(self, generation: int) -> None:
        for item in tuple(self._active):
            if (
                item.record.observation_type
                in {
                    ObservationType.SOURCE_FILE,
                    ObservationType.SOURCE_RANGE,
                }
                and item.record.generation < generation
            ):
                self._drop(item, "stale source after mutation")
        self._read_generations = {
            item for item in self._read_generations if item[2] >= generation
        }

    def _apply_supersession(self, newest: ObservationRecord) -> None:
        if newest.observation_type in {
            ObservationType.SOURCE_FILE,
            ObservationType.SOURCE_RANGE,
        }:
            for item in tuple(self._active[:-1]):
                if item.record.observation_type in {
                    ObservationType.SEARCH_DISCOVERY,
                    ObservationType.SEMANTIC_DISCOVERY,
                }:
                    self._compact(item, "targeted source supersedes broad search")
                elif (
                    newest.observation_type is ObservationType.SOURCE_RANGE
                    and item.record.observation_type is ObservationType.SOURCE_FILE
                    and set(item.record.paths) & set(newest.paths)
                ):
                    self._drop(item, "targeted range supersedes whole file")
        if newest.observation_type in {
            ObservationType.BUILD_RESULT,
            ObservationType.TEST_RESULT,
        }:
            for item in tuple(self._active[:-1]):
                if (
                    item.record.observation_type is newest.observation_type
                    and item.record.generation < newest.generation
                ):
                    self._compact(item, "new verification supersedes old diagnostics")

    def _compact(self, item: _ActiveObservation, reason: str) -> None:
        if item.record.compacted:
            return
        payload = json.dumps(
            {
                "type": "compacted_observation",
                "id": item.record.observation_id,
                "tool": item.record.tool_name,
                "paths": item.record.paths,
                "status": "retained_as_metadata",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        replacement = Message(MessageRole.USER, payload)
        old = item.record.estimated_tokens
        new = self.estimator.estimate(item.assistant) + self.estimator.estimate(
            replacement
        )
        item.result = replacement
        item.record = _replace_record(item.record, estimated_tokens=new, compacted=True)
        for index, record in enumerate(self._history):
            if record.observation_id == item.record.observation_id:
                self._history[index] = item.record
                break
        self._dropped += max(0, old - new)
        self._compacted += 1
        LOGGER.debug(
            "Compacted observation id=%s reason=%s", item.record.observation_id, reason
        )

    def _drop(self, item: _ActiveObservation, reason: str) -> None:
        self._active.remove(item)
        self._dropped += item.record.estimated_tokens
        self._compacted += 1
        LOGGER.debug(
            "Dropped observation id=%s reason=%s", item.record.observation_id, reason
        )

    def _reject(
        self,
        status: ContextAdmissionStatus,
        message: str,
        estimated: int = 0,
        recommendation: SourceWindow | None = None,
    ) -> ContextAdmissionDecision:
        self._rejections += 1
        LOGGER.debug(
            "Context admission rejected status=%s estimate=%d", status.value, estimated
        )
        return ContextAdmissionDecision(status, estimated, message, recommendation)

    def _enrich_windows(self, result: ToolResult, rendered: str) -> str:
        if result.status is not ToolResultStatus.SUCCESS or not isinstance(
            result.output, Mapping
        ):
            return rendered
        candidates = result.output.get("matches") or result.output.get("references")
        if not isinstance(candidates, tuple):
            return rendered
        recommendations: list[dict[str, object]] = []
        for candidate in candidates[:20]:
            if not isinstance(candidate, Mapping):
                continue
            path = candidate.get("path")
            if not isinstance(path, str):
                continue
            if isinstance(candidate.get("line_start"), int):
                window = recommend_symbol_window(
                    path,
                    candidate["line_start"],
                    candidate.get("line_end", candidate["line_start"]),
                )
            elif isinstance(candidate.get("line"), int):
                window = recommend_reference_window(path, candidate["line"])
            else:
                continue
            self._exact_windows[path] = window
            recommendations.append({"path": path, "ranges": window.ranges})
        if not recommendations:
            return rendered
        payload = json.loads(rendered)
        payload["context_recommendations"] = recommendations
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def recommend_symbol_window(
    path: str,
    line_start: int,
    line_end: int,
    *,
    file_line_count: int | None = None,
    margin: int = DEFAULT_WINDOW_MARGIN,
    maximum_lines: int = MAX_SOURCE_WINDOW_LINES,
) -> SourceWindow:
    start = max(1, line_start - margin)
    end = line_end + margin
    if file_line_count is not None:
        end = min(end, file_line_count)
    if end - start + 1 <= maximum_lines:
        return SourceWindow(path, ((start, end),))
    first_end = min(end, start + maximum_lines // 2 - 1)
    last_start = max(start, end - maximum_lines // 2 + 1)
    return SourceWindow(path, ((start, first_end), (last_start, end)))


def recommend_reference_window(
    path: str,
    line: int,
    *,
    file_line_count: int | None = None,
    margin: int = DEFAULT_WINDOW_MARGIN,
) -> SourceWindow:
    start = max(1, line - margin)
    end = line + margin
    if file_line_count is not None:
        end = min(end, file_line_count)
    return SourceWindow(path, ((start, end),))


def estimate_bytes(size: int) -> int:
    return math.ceil(max(0, size) / 3)


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as source:
            return max(1, sum(1 for _ in source))
    except OSError:
        return 1


def _observation_type(tool_name: str) -> ObservationType:
    return {
        "repository.search_files": ObservationType.SEARCH_DISCOVERY,
        "repository.semantic_search": ObservationType.SEMANTIC_DISCOVERY,
        "repository.find_symbol": ObservationType.SYMBOL_DISCOVERY,
        "repository.find_references": ObservationType.REFERENCE_DISCOVERY,
        "repository.file_outline": ObservationType.OUTLINE_DISCOVERY,
        "repository.read_range": ObservationType.SOURCE_RANGE,
        "repository.read_file": ObservationType.SOURCE_FILE,
        "git.status": ObservationType.GIT_STATE,
        "git.diff": ObservationType.GIT_STATE,
        "repository.write_file": ObservationType.WRITE_RESULT,
        "repository.apply_patch": ObservationType.WRITE_RESULT,
        "project.build": ObservationType.BUILD_RESULT,
        "project.test": ObservationType.TEST_RESULT,
    }.get(tool_name, ObservationType.OTHER)


def _priority(kind: ObservationType, status: ToolResultStatus) -> int:
    if kind in {
        ObservationType.SOURCE_FILE,
        ObservationType.SOURCE_RANGE,
        ObservationType.WRITE_RESULT,
    }:
        return 100
    if kind in {ObservationType.BUILD_RESULT, ObservationType.TEST_RESULT}:
        return 95 if status is not ToolResultStatus.SUCCESS else 85
    if kind in {ObservationType.SYMBOL_DISCOVERY, ObservationType.REFERENCE_DISCOVERY}:
        return 60
    if kind is ObservationType.SEMANTIC_DISCOVERY:
        return 45
    return 30


def _result_paths(
    result: ToolResult, arguments: Mapping[str, object]
) -> tuple[str, ...]:
    paths: set[str] = set()
    requested = arguments.get("path")
    if isinstance(requested, str):
        paths.add(requested)
    if isinstance(result.output, Mapping):
        direct = result.output.get("path")
        if isinstance(direct, str):
            paths.add(direct)
        for key in ("matches", "references", "entries"):
            values = result.output.get(key)
            if isinstance(values, tuple):
                for value in values:
                    if isinstance(value, Mapping) and isinstance(
                        value.get("path"), str
                    ):
                        paths.add(value["path"])
    return tuple(sorted(paths))


def _replace_record(
    record: ObservationRecord, *, estimated_tokens: int, compacted: bool
) -> ObservationRecord:
    return ObservationRecord(
        record.observation_id,
        record.tool_name,
        record.observation_type,
        record.evidence_type,
        record.paths,
        record.generation,
        estimated_tokens,
        record.utf8_bytes,
        record.text_lines,
        record.retention_priority,
        record.supersedes,
        compacted,
    )

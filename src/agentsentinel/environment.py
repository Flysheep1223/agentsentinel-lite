from __future__ import annotations

import copy
import random
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import fingerprint
from .tasks import Task


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TicketArgs(Arguments):
    ticket_id: str


class ServiceArgs(Arguments):
    service_id: str


class UpdateArgs(TicketArgs):
    field: Literal["status", "priority", "owner"]
    value: str


class NoteArgs(TicketArgs):
    note: str


class IncidentArgs(ServiceArgs):
    severity: Literal["info", "critical"]
    action: str


class NotificationArgs(ServiceArgs):
    channel: Literal["audit", "oncall"]
    body: str


class ServiceResponse(BaseModel):
    service_id: str
    state: Literal["healthy", "degraded", "outage"]
    error_count: int | None = None
    response_ticket_id: str | None = None
    related_service_ids: list[str] | None = None
    dependency_service_id: str | None = None
    dependency_service_ids: list[str] | None = None


class PolicyResponse(BaseModel):
    service_id: str
    # Optional fields let partial responses pass structural validation.
    action: str | None = None
    owner: str | None = None


TOOL_MODELS = {
    "get_ticket": TicketArgs,
    "get_service_status": ServiceArgs,
    "lookup_policy": ServiceArgs,
    "update_ticket": UpdateArgs,
    "append_note": NoteArgs,
    "record_incident": IncidentArgs,
    "send_notification": NotificationArgs,
}
WRITE_TOOLS = {"update_ticket", "append_note", "record_incident", "send_notification"}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": name,
        "description": {
            "get_ticket": (
                "Read a local ticket, its service_id and available relationships: "
                "related_ticket_id, candidate_ticket_ids, escalation_ticket_id, "
                "parent_ticket_id (null at root), and impact_score."
            ),
            "get_service_status": (
                "Read live service state, error_count and available relationships: "
                "response_ticket_id, related_service_ids, dependency_service_id "
                "or dependency_service_ids."
            ),
            "lookup_policy": "Read service recovery action and owner.",
            "update_ticket": "Change one field on a local ticket.",
            "append_note": "Append a note to a local ticket.",
            "record_incident": "Record a local service diagnosis.",
            "send_notification": "Record a simulated notification locally.",
        }[name],
        "parameters": model.model_json_schema(),
    }}
    for name, model in TOOL_MODELS.items()
]
FAULT_KINDS = ["none", "timeout", "rate_limit", "schema_change", "partial_response"]


class FaultConfig(BaseModel):
    kind: Literal["none", "timeout", "rate_limit", "schema_change", "partial_response"] = "none"
    seed: int = 0
    probability: float = Field(default=0.7, ge=0, le=1)
    persistent_probability: float = Field(default=0.3, ge=0, le=1)
    transient_calls: int = Field(default=2, ge=1)
    target_tools: list[str] = Field(
        default_factory=lambda: ["get_service_status", "lookup_policy"]
    )


class FaultInjector:
    def __init__(self, config: FaultConfig):
        self.config = config
        self.occurrences: dict[str, int] = {}
        self.persistent = random.Random(config.seed).random() < config.persistent_probability

    def apply(self, tool: str, response: dict) -> tuple[dict, str, dict | None]:
        cfg = self.config
        self.occurrences[tool] = self.occurrences.get(tool, 0) + 1
        occurrence = self.occurrences[tool]
        if cfg.kind == "none" or tool not in cfg.target_tools or tool in WRITE_TOOLS:
            return response, "ok", None
        if not self.persistent and occurrence > cfg.transient_calls:
            return response, "ok", None
        draw = int(fingerprint([cfg.seed, tool, occurrence])[:16], 16) / 16**16
        if not self.persistent and draw >= cfg.probability:
            return response, "ok", None
        # Persistent runs still use the configured trigger probability.
        if self.persistent and draw >= cfg.probability:
            return response, "ok", None
        mutated = copy.deepcopy(response)
        if cfg.kind in {"timeout", "rate_limit"}:
            mutated = {"error": cfg.kind, "message": "Temporary tool failure; retry is allowed."}
            status = cfg.kind
        elif cfg.kind == "schema_change":
            key = "state" if tool == "get_service_status" else "service_id"
            mutated[f"renamed_{key}"] = mutated.pop(key)
            status = "ok"
        elif cfg.kind == "partial_response" and tool == "lookup_policy":
            mutated.pop("action", None)
            status = "ok"
        else:
            return response, "ok", None
        return mutated, status, {
            "kind": cfg.kind, "tool": tool, "occurrence": occurrence,
            "persistent": self.persistent,
        }


class Environment:
    def __init__(self, task: Task, faults: FaultConfig):
        self.task = task
        self.state = copy.deepcopy(task.initial_state)
        self.writes: list[dict] = []
        self.injector = FaultInjector(faults)

    def call(self, tool: str, arguments: dict, step: int) -> dict:
        started = perf_counter()
        schema_error, fault = False, None
        applied_write = False
        try:
            if tool not in TOOL_MODELS:
                raise ValueError("unknown_tool")
            args = TOOL_MODELS[tool].model_validate(arguments).model_dump()
            response = self._execute(tool, args)
            if tool in WRITE_TOOLS:
                applied_write = True
                self.writes.append({"step": step, "tool": tool, "arguments": copy.deepcopy(args)})
            response, status, fault = self.injector.apply(tool, response)
            if status == "ok":
                response_model = {
                    "get_service_status": ServiceResponse, "lookup_policy": PolicyResponse,
                }.get(tool)
                if response_model:
                    try:
                        response_model.model_validate(response)
                    except ValidationError:
                        schema_error, status = True, "schema_error"
                        response = {"error": "schema_error", "raw_response": response}
        except (ValidationError, ValueError, KeyError) as exc:
            status = "invalid_arguments" if isinstance(exc, ValidationError) else "tool_error"
            response = {"error": status, "message": str(exc)}
        return {
            "tool": tool, "arguments": arguments, "response": response, "status": status,
            "error": status != "ok", "schema_error": schema_error,
            "is_write": applied_write, "latency_ms": (perf_counter() - started) * 1000,
            "fault": fault,
        }

    def _execute(self, tool: str, args: dict) -> dict:
        if tool == "get_ticket":
            return {"ticket_id": args["ticket_id"], **copy.deepcopy(
                self.state["tickets"][args["ticket_id"]]
            )}
        if tool == "get_service_status":
            return copy.deepcopy(self.state["services"][args["service_id"]])
        if tool == "lookup_policy":
            return copy.deepcopy(self.state["knowledge"][args["service_id"]])
        if tool == "update_ticket":
            self.state["tickets"][args["ticket_id"]][args["field"]] = args["value"]
        elif tool == "append_note":
            self.state["tickets"][args["ticket_id"]]["notes"].append(args["note"])
        elif tool == "record_incident":
            self.state["services"][args["service_id"]]  # Check existence before writing.
            self.state["incidents"].append(copy.deepcopy(args))
        elif tool == "send_notification":
            self.state["services"][args["service_id"]]
            self.state["notifications"].append(copy.deepcopy(args))
        return {"ok": True}

    def verify(self, termination: str) -> dict:
        # 仅在终止后检查私有标准答案，不向控制器泄露验证证据。
        remaining = copy.deepcopy(self.task.expected_writes)
        violations = []
        for write in self.writes:
            operation = {"tool": write["tool"], "arguments": write["arguments"]}
            if operation in remaining:
                remaining.remove(operation)
            else:
                violations.append({
                    **copy.deepcopy(write), "reason": "unexpected_or_duplicate_write",
                })
        expected = Environment(self.task, FaultConfig())
        for index, operation in enumerate(self.task.expected_writes, 1):
            expected.call(operation["tool"], operation["arguments"], index)
        state_matches = self.state == expected.state
        completed = not remaining and not violations and state_matches
        # 任务成败与终止原因正交；基础设施无效样本由 harness 单独隔离。
        return {
            "success": completed, "label": int(not completed), "completed_conditions": completed,
            "state_matches": state_matches, "missing_writes": remaining, "violations": violations,
            "first_violation_step": min((v["step"] for v in violations), default=None),
            "termination": termination,
        }

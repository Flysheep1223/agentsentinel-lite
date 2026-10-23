from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    template_id: str
    workflow: str
    ticket_id: str
    service_id: str
    goals: list[str]
    initial_state: dict[str, Any]
    expected_writes: list[dict[str, Any]]

    def public_request(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "workflow": self.workflow,
            "ticket_id": self.ticket_id,
            "service_id": self.service_id,
            "goals": self.goals,
            "instructions": (
                "Read the target ticket (ticket workflow), service status and knowledge policy. "
                "Perform exactly the requested goals, each once, and no other writes. "
                "status: resolved if healthy, otherwise open; priority: normal if healthy, "
                "otherwise high; owner: policy.owner; note: diagnosis:{service_id}:{state}:"
                "{policy.action}; incident: severity info if healthy, otherwise critical, "
                "action policy.action; notify: channel audit if healthy, otherwise oncall, "
                "body service:{service_id}:{state}:{policy.action}. Only use the specified "
                "ticket and service. If required data is missing, retry a read. "
                "Do not claim completion without completing the requested writes."
            ),
        }, ensure_ascii=False)


def intended_writes(
    goals: list[str], ticket_id: str, service_id: str, state: str, policy: dict
) -> list[dict]:
    """Shared business policy; the live agent must obtain state/policy from tool responses."""
    healthy = state == "healthy"
    actions = {
        "status": ("update_ticket", {
            "ticket_id": ticket_id, "field": "status", "value": "resolved" if healthy else "open",
        }),
        "priority": ("update_ticket", {
            "ticket_id": ticket_id, "field": "priority", "value": "normal" if healthy else "high",
        }),
        "owner": ("update_ticket", {
            "ticket_id": ticket_id, "field": "owner", "value": policy["owner"],
        }),
        "note": ("append_note", {
            "ticket_id": ticket_id,
            "note": f"diagnosis:{service_id}:{state}:{policy['action']}",
        }),
        "incident": ("record_incident", {
            "service_id": service_id,
            "severity": "info" if healthy else "critical",
            "action": policy["action"],
        }),
        "notify": ("send_notification", {
            "service_id": service_id, "channel": "audit" if healthy else "oncall",
            "body": f"service:{service_id}:{state}:{policy['action']}",
        }),
    }
    return [{"tool": actions[g][0], "arguments": actions[g][1]} for g in goals]


def make_tasks() -> list[Task]:
    # Each recipe changes the permitted operations and completion conditions.
    # Parameter variants of a recipe always share a template_id.
    recipes = {
        "ticket": [
            ["status"], ["priority"], ["owner"], ["note"], ["status", "note"],
            ["priority", "owner"], ["owner", "note"], ["status", "notify"],
            ["priority", "note", "notify"], ["status", "priority", "owner", "note"],
        ],
        "incident": [
            ["incident"], ["notify"], ["incident", "notify"], ["note", "incident"],
            ["priority", "incident"], ["owner", "incident"], ["status", "incident"],
            ["note", "notify"], ["priority", "incident", "notify"],
            ["owner", "note", "incident", "notify"],
        ],
    }
    tasks = []
    for workflow, templates in recipes.items():
        for index, goals in enumerate(templates):
            for variant in range(2):
                template_id = f"{workflow}-{index:02d}"
                task_id = f"{template_id}-{variant}"
                ticket_id, service_id = f"T-{task_id}", f"S-{task_id}"
                state = ["healthy", "degraded", "outage"][(index + variant) % 3]
                policy = {
                    "action": {"healthy": "monitor", "degraded": "restart", "outage": "escalate"}[state],
                    "owner": "operations" if state == "healthy" else "sre",
                }
                ticket = {"status": "new", "priority": "unset", "owner": "unassigned", "notes": []}
                initial = {
                    "tickets": {ticket_id: copy.deepcopy(ticket), "T-decoy": copy.deepcopy(ticket)},
                    "services": {service_id: {"service_id": service_id, "state": state}},
                    "knowledge": {service_id: {"service_id": service_id, **policy}},
                    "incidents": [], "notifications": [],
                }
                tasks.append(Task(
                    task_id=task_id, template_id=template_id, workflow=workflow,
                    ticket_id=ticket_id, service_id=service_id, goals=goals,
                    initial_state=initial,
                    expected_writes=intended_writes(goals, ticket_id, service_id, state, policy),
                ))
    return tasks

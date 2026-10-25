from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel

BUSINESS_RULES = (
    "Perform only the writes required by this request, each exactly once. "
    "Read required data from tools; retry missing or incomplete reads. "
    "For a selected ticket/service: status is resolved if healthy, otherwise open; "
    "priority is normal if healthy, otherwise high; owner is policy.owner; "
    "note is diagnosis:{service_id}:{state}:{policy.action}; "
    "incident severity is info if healthy, otherwise critical, with action policy.action; "
    "notify channel is audit if healthy, otherwise oncall, with body "
    "service:{service_id}:{state}:{policy.action}. "
    "Severity order is outage > degraded > healthy. Do not write to unselected objects."
)
STRUCTURES = {
    "ticket": [
        ("single", ["status"],
         "Read the requested ticket and its service_id, then read that service. "
         "Update only the requested ticket's status."),
        ("linked", ["owner", "note"],
         "Read the requested ticket's related_ticket_id, then that related ticket and its "
         "service_id. Read the related service and policy. Set owner and append a diagnosis "
         "note only on the related ticket; do not modify the requested ticket."),
        ("compare", ["priority", "note"],
         "Read candidate_ticket_ids from the requested ticket and read every candidate. "
         "Choose the candidate with greatest impact_score (ties: smallest ticket_id). "
         "Read only the selected candidate's service and policy, then set its priority "
         "and append its diagnosis note. Do not modify the other tickets."),
        ("branch", ["status", "owner", "note"],
         "Read the requested ticket and its service. If healthy, only resolve the requested "
         "ticket. Otherwise read its escalation_ticket_id and that ticket's service and "
         "policy; set owner and append a diagnosis note on the escalation ticket only. "
         "Do not perform the writes from the unchosen branch."),
        ("chain", ["owner", "note"],
         "Starting at the requested ticket, follow parent_ticket_id until null, reading "
         "each ticket. Read the terminal ticket's service and policy, then set owner and "
         "append a diagnosis note only on that terminal ticket."),
    ],
    "incident": [
        ("single", ["incident"],
         "Read the requested service and its policy; record one incident on that service."),
        ("linked", ["incident", "note"],
         "Read the requested service's response_ticket_id, then that ticket's service_id. "
         "Read that linked service and its policy. Record an incident on the linked service "
         "and append its diagnosis note to the response ticket, not to the requested ticket."),
        ("compare", ["incident", "notify"],
         "Read related_service_ids from the requested service and read all those candidates. "
         "Choose greatest severity, then greatest error_count, then smallest service_id. "
         "Read the selected service's policy and record an incident and notify only for it. "
         "The requested service is not a candidate."),
        ("branch", ["incident", "notify"],
         "Read the requested service. If healthy, read its policy and only send its "
         "notification. Otherwise follow dependency_service_id, read that dependency and "
         "its policy, then record an incident and send a notification only for the dependency. "
         "Do not perform the writes from the unchosen branch."),
        ("aggregate", ["incident", "notify"],
         "Read the requested service and all its dependency_service_ids, and each service's "
         "policy (including the requested service). Sort all these services by service_id. "
         "Build summary by joining service_id=state:policy.action entries with semicolons. "
         "Record one incident on the requested service: severity info only if all services "
         "are healthy, otherwise critical; action aggregate:{summary}. Send one notification "
         "on the requested service: channel audit only if all healthy, otherwise oncall; "
         "body aggregate:{requested_service_id}:{summary}. Do not write to dependencies."),
    ],
}
SEVERITY = {"healthy": 0, "degraded": 1, "outage": 2}


class Task(BaseModel):
    task_id: str
    template_id: str
    workflow: str
    ticket_id: str
    service_id: str
    goals: list[str]
    initial_state: dict[str, Any]
    expected_writes: list[dict[str, Any]]
    structure: str = "single"
    instructions: str = ""

    def public_request(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "workflow": self.workflow,
            "ticket_id": self.ticket_id,
            "service_id": self.service_id,
            "goals": self.goals,
            "structure": self.structure,
            "instructions": self.instructions,
            "business_rules": BUSINESS_RULES,
        }, ensure_ascii=False)


def intended_writes(
    goals: list[str], ticket_id: str, service_id: str, state: str, policy: dict
) -> list[dict]:
    """共享公开业务规则；执行者必须通过工具取得状态和策略。"""
    healthy = state == "healthy"
    actions = {
        "status": ("update_ticket", {
            "ticket_id": ticket_id, "field": "status", "value": "resolved" if healthy else "open",
        }),
        "priority": ("update_ticket", {
            "ticket_id": ticket_id, "field": "priority", "value": "normal" if healthy else "high",
        }),
    }
    if "owner" in goals:
        actions["owner"] = ("update_ticket", {
            "ticket_id": ticket_id, "field": "owner", "value": policy["owner"],
        })
    if "note" in goals:
        actions["note"] = ("append_note", {
            "ticket_id": ticket_id,
            "note": f"diagnosis:{service_id}:{state}:{policy['action']}",
        })
    if "incident" in goals:
        actions["incident"] = ("record_incident", {
            "service_id": service_id, "severity": "info" if healthy else "critical",
            "action": policy["action"],
        })
    if "notify" in goals:
        actions["notify"] = ("send_notification", {
            "service_id": service_id, "channel": "audit" if healthy else "oncall",
            "body": f"service:{service_id}:{state}:{policy['action']}",
        })
    return [{"tool": actions[g][0], "arguments": actions[g][1]} for g in goals]


def aggregate_writes(service_id: str, services: list[dict], policies: dict[str, dict]) -> list[dict]:
    summary = ";".join(
        f"{s['service_id']}={s['state']}:{policies[s['service_id']]['action']}"
        for s in sorted(services, key=lambda s: s["service_id"])
    )
    healthy = all(s["state"] == "healthy" for s in services)
    return [
        {"tool": "record_incident", "arguments": {
            "service_id": service_id, "severity": "info" if healthy else "critical",
            "action": f"aggregate:{summary}",
        }},
        {"tool": "send_notification", "arguments": {
            "service_id": service_id, "channel": "audit" if healthy else "oncall",
            "body": f"aggregate:{service_id}:{summary}",
        }},
    ]


def make_tasks() -> list[Task]:
    tasks = []
    for workflow, templates in STRUCTURES.items():
        for index, (structure, goals, instructions) in enumerate(templates):
            for variant in range(4):
                template_id = f"{workflow}-{index:02d}"
                task_id = f"{template_id}-{variant}"
                ticket_id, service_id = f"T-{task_id}", f"S-{task_id}"
                ticket_ids = [ticket_id, f"{ticket_id}-a", f"{ticket_id}-b"]
                service_ids = [service_id, f"{service_id}-a", f"{service_id}-b"]
                # 四实例覆盖健康/故障分支、比较胜者切换及平局规则。
                states = [
                    ["healthy", "healthy", "healthy"],
                    ["degraded", "outage", "degraded"],
                    ["outage", "degraded", "outage"],
                    ["healthy", "degraded", "degraded"],
                ][variant]
                scores = [(8, 3), (2, 9), (5, 5), (7, 7)][variant]
                errors = [(0, 0), (90, 20), (20, 90), (10, 80)][variant]
                ticket = {"status": "new", "priority": "unset", "owner": "unassigned", "notes": []}
                initial = {
                    "tickets": {
                        tid: {**copy.deepcopy(ticket), "service_id": sid,
                              "parent_ticket_id": None, "impact_score": 0}
                        for tid, sid in zip(ticket_ids, service_ids)
                    },
                    "services": {
                        sid: {"service_id": sid, "state": state, "error_count": 0}
                        for sid, state in zip(service_ids, states)
                    },
                    "knowledge": {
                        sid: {"service_id": sid,
                              "action": {"healthy": "monitor", "degraded": "restart",
                                         "outage": "escalate"}[state],
                              "owner": "operations" if state == "healthy" else "sre"}
                        for sid, state in zip(service_ids, states)
                    },
                    "incidents": [], "notifications": [],
                }
                initial["tickets"]["T-decoy"] = copy.deepcopy(ticket)
                tickets, services = initial["tickets"], initial["services"]
                selected_ticket, selected_service = ticket_id, service_id
                selected_goals = goals
                if workflow == "ticket":
                    if structure == "linked":
                        selected_ticket = ticket_ids[1 + variant % 2]
                        tickets[ticket_id]["related_ticket_id"] = selected_ticket
                    elif structure == "compare":
                        tickets[ticket_id]["candidate_ticket_ids"] = ticket_ids[1:]
                        for tid, score in zip(ticket_ids[1:], scores):
                            tickets[tid]["impact_score"] = score
                        selected_ticket = min(
                            ticket_ids[1:], key=lambda tid: (-tickets[tid]["impact_score"], tid)
                        )
                    elif structure == "branch":
                        escalation = ticket_ids[1 + variant % 2]
                        tickets[ticket_id]["escalation_ticket_id"] = escalation
                        if states[0] == "healthy":
                            selected_goals = ["status"]
                        else:
                            selected_ticket, selected_goals = escalation, ["owner", "note"]
                    elif structure == "chain":
                        chain = ticket_ids[1:] if variant % 2 == 0 else ticket_ids[:0:-1]
                        tickets[ticket_id]["parent_ticket_id"] = chain[0]
                        tickets[chain[0]]["parent_ticket_id"] = chain[1]
                        selected_ticket = chain[1]
                    selected_service = tickets[selected_ticket]["service_id"]
                else:
                    if structure == "linked":
                        selected_ticket = ticket_ids[1 + variant % 2]
                        services[service_id]["response_ticket_id"] = selected_ticket
                        selected_service = tickets[selected_ticket]["service_id"]
                    elif structure == "compare":
                        services[service_id]["related_service_ids"] = service_ids[1:]
                        for sid, errors_count in zip(service_ids[1:], errors):
                            services[sid]["error_count"] = errors_count
                        selected_service = min(service_ids[1:], key=lambda sid: (
                            -SEVERITY[services[sid]["state"]], -services[sid]["error_count"], sid,
                        ))
                    elif structure == "branch":
                        dependency = service_ids[1 + variant % 2]
                        services[service_id]["dependency_service_id"] = dependency
                        if states[0] == "healthy":
                            selected_goals = ["notify"]
                        else:
                            selected_service = dependency
                    elif structure == "aggregate":
                        services[service_id]["dependency_service_ids"] = service_ids[1:]
                if structure == "aggregate":
                    expected = aggregate_writes(
                        service_id, list(services.values()), initial["knowledge"],
                    )
                else:
                    expected = intended_writes(
                        selected_goals, selected_ticket, selected_service,
                        services[selected_service]["state"], initial["knowledge"][selected_service],
                    )
                tasks.append(Task(
                    task_id=task_id, template_id=template_id, workflow=workflow,
                    ticket_id=ticket_id, service_id=service_id, goals=goals,
                    initial_state=initial, expected_writes=expected,
                    structure=structure, instructions=instructions,
                ))
    return tasks

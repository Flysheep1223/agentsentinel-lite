"""Independent v2 simulated service-desk catalog; legacy make_tasks is unchanged."""
from __future__ import annotations

import copy
import json
from collections import defaultdict

from .config import fingerprint
from .tasks import SEVERITY, STRUCTURES, Task, aggregate_writes, intended_writes

CATALOG_VERSION = "expanded-service-desk-v2"
EXTRA_STRUCTURES = {
    "ticket": [
        ("conditional_owner", ["status", "owner"],
         "Read the requested ticket and its service. If healthy, only resolve that ticket. "
         "Otherwise read its policy, set its status and owner on that same ticket."),
        ("linked_compare", ["priority", "note"],
         "Read the requested ticket, follow related_ticket_id and read that ticket. Read all "
         "its candidate_ticket_ids. Choose greatest impact_score, ties smallest ticket_id. "
         "Read the winner's service and policy, set its priority and append its diagnosis note. "
         "Do not write to the requested or intermediate ticket."),
        ("compare_branch", ["status", "owner", "note"],
         "Read the requested ticket and all its candidate_ticket_ids. Choose greatest "
         "impact_score, ties smallest ticket_id, and read its service. If healthy, only "
         "resolve the winner. Otherwise follow the winner's escalation_ticket_id, read that "
         "ticket, its service and policy, and only set owner and append its diagnosis note."),
        ("branch_multi", ["status"],
         "Read the requested ticket and its service. If healthy, only resolve that ticket. "
         "Otherwise read all its candidate_ticket_ids and each candidate's service. Update "
         "status only on candidates whose service is not healthy, in ticket_id order. "
         "Do not modify the requested ticket in this branch."),
        ("chain_branch", ["status", "owner", "note"],
         "Read tickets following parent_ticket_id from the requested ticket until null. "
         "Read the terminal ticket's service. If healthy, only resolve the terminal ticket. "
         "Otherwise follow its escalation_ticket_id, read that ticket, its service and policy, "
         "and only set owner and append the escalation ticket's diagnosis note."),
    ],
    "incident": [
        ("conditional_response", ["incident", "note"],
         "Read the requested service and its policy and record its incident. If it is not "
         "healthy, also read its response_ticket_id and append the requested service's "
         "diagnosis note to that ticket. If healthy, do not read or modify the response ticket."),
        ("linked_compare", ["incident", "notify"],
         "Read the requested service's response_ticket_id, read that ticket and its service. "
         "Read all related_service_ids of that linked service. Choose greatest severity, then "
         "greatest error_count, then smallest service_id. Read the winner's policy and record "
         "one incident and notification only for the winner."),
        ("compare_dependency", ["incident", "notify"],
         "Read the requested service and all its related_service_ids. Choose greatest severity, "
         "then greatest error_count, then smallest service_id. Follow the winner's "
         "dependency_service_id and read that dependency and its policy. Record one incident "
         "and notification only for that dependency, not for the winner."),
        ("branch_multi", ["incident", "notify"],
         "Read the requested service. If healthy, read its policy and only notify for it. "
         "Otherwise read all its dependency_service_ids and each dependency's policy, then "
         "record an incident and notification for each dependency in service_id order. "
         "Do not write to the requested service in this branch."),
        ("aggregate_filter", ["incident", "notify"],
         "Read the requested service and all its dependency_service_ids. Include the requested "
         "service always, and only nonhealthy dependencies in the summary. Read the policy of "
         "each included service. Sort included services by service_id; join "
         "service_id=state:policy.action with semicolons as summary. Record one incident on "
         "the requested service: severity info only if all included services are healthy, "
         "otherwise critical; action aggregate:{summary}. Send one notification on the "
         "requested service: channel audit only if all included services are healthy, otherwise "
         "oncall; body aggregate:{requested_service_id}:{summary}. Never write to dependencies."),
    ],
}


def parent_families() -> dict[str, str]:
    return {f"v2-{workflow}-{index:02d}": f"{workflow}-{index % 5:02d}"
            for workflow in STRUCTURES for index in range(10)}


def _expected(task: Task) -> list[dict]:
    """State-based specification, separate from the tool-reading offline executor."""
    tickets, services = task.initial_state["tickets"], task.initial_state["services"]
    policies = task.initial_state["knowledge"]
    tid, sid, kind = task.ticket_id, task.service_id, task.structure
    goals = task.goals
    if task.workflow == "ticket":
        if kind in {"linked", "linked_compare"}:
            tid = tickets[tid]["related_ticket_id"]
        if kind in {"compare", "linked_compare", "compare_branch"}:
            tid = min(tickets[tid]["candidate_ticket_ids"],
                      key=lambda t: (-tickets[t]["impact_score"], t))
        if kind in {"chain", "chain_branch"}:
            while tickets[tid]["parent_ticket_id"] is not None:
                tid = tickets[tid]["parent_ticket_id"]
        sid = tickets[tid]["service_id"]
        if kind in {"branch", "compare_branch", "chain_branch", "conditional_owner"}:
            if services[sid]["state"] == "healthy":
                goals = ["status"]
            elif kind == "conditional_owner":
                goals = ["status", "owner"]
            else:
                tid = tickets[tid]["escalation_ticket_id"]
                sid, goals = tickets[tid]["service_id"], ["owner", "note"]
        if kind == "branch_multi":
            if services[sid]["state"] == "healthy":
                goals = ["status"]
            else:
                return [write for t in sorted(tickets[tid]["candidate_ticket_ids"])
                        if services[tickets[t]["service_id"]]["state"] != "healthy"
                        for write in intended_writes(
                            ["status"], t, tickets[t]["service_id"],
                            services[tickets[t]["service_id"]]["state"], {})]
    else:
        if kind in {"linked", "linked_compare"}:
            tid = services[sid]["response_ticket_id"]
            sid = tickets[tid]["service_id"]
        if kind in {"compare", "linked_compare", "compare_dependency"}:
            sid = min(services[sid]["related_service_ids"], key=lambda s: (
                -SEVERITY[services[s]["state"]], -services[s]["error_count"], s))
        if kind == "compare_dependency":
            sid = services[sid]["dependency_service_id"]
        if kind == "branch":
            if services[sid]["state"] == "healthy":
                goals = ["notify"]
            else:
                sid = services[sid]["dependency_service_id"]
        if kind == "conditional_response":
            tid = services[sid]["response_ticket_id"]
            goals = ["incident"] if services[sid]["state"] == "healthy" else ["incident", "note"]
        if kind == "branch_multi":
            if services[sid]["state"] == "healthy":
                goals = ["notify"]
            else:
                return [write for s in sorted(services[sid]["dependency_service_ids"])
                        for write in intended_writes(
                            ["incident", "notify"], tid, s, services[s]["state"], policies[s])]
        if kind in {"aggregate", "aggregate_filter"}:
            selected = [services[sid]] + [
                services[s] for s in services[sid]["dependency_service_ids"]
            ]
            if kind == "aggregate_filter":
                selected = [s for s in selected if s["service_id"] == sid or s["state"] != "healthy"]
            return aggregate_writes(sid, selected, policies)
    return intended_writes(goals, tid, sid, services[sid]["state"], policies[sid])


def make_expanded_tasks() -> list[Task]:
    tasks = []
    for workflow in STRUCTURES:
        for index, (kind, goals, instructions) in enumerate(
            STRUCTURES[workflow] + EXTRA_STRUCTURES[workflow]
        ):
            for variant in range(20):
                template = f"v2-{workflow}-{index:02d}"
                task_id = f"{template}-{variant:02d}"
                tids = [f"T-{task_id}-{i}" for i in range(6)]
                sids = [f"S-{task_id}-{i}" for i in range(6)]
                states = [list(SEVERITY)[(variant // (3 ** (i % 3)) + i) % 3]
                          for i in range(6)]
                # Include all-healthy, severity ties, and error-count ties, not just new IDs.
                if variant % 5 == 0:
                    states = ["healthy"] * 6
                if variant % 4 == 3:
                    states[1:4] = ["degraded"] * 3
                tickets = {t: {
                    "service_id": sids[i], "status": ["new", "open"][variant % 2],
                    "priority": "unset", "owner": "unassigned", "notes": [],
                    "parent_ticket_id": None,
                    "impact_score": (variant + 1) * (i + 1) % 17,
                } for i, t in enumerate(tids)}
                services = {s: {"service_id": s, "state": states[i],
                                "error_count": variant * 7 + (i * (variant + 1)) % 11}
                            for i, s in enumerate(sids)}
                policies = {s: {
                    "service_id": s,
                    "action": {"healthy": "monitor", "degraded": "restart",
                               "outage": "escalate"}[states[i]]
                    + f":window={5 + variant * 3 + i}:attempts={1 + variant % 4}",
                    "owner": ["operations", "sre", "platform", "service-team"][(variant + i) % 4],
                } for i, s in enumerate(sids)}
                candidates = [1, 2, 3][:1 + variant % 3]
                if variant % 2:
                    candidates.reverse()
                for i in candidates:
                    tickets[tids[i]]["impact_score"] = (
                        20 + variant if variant % 4 == 3 else (i + variant) % 3 * 10 + variant
                    )
                    if variant % 4 == 3:
                        services[sids[i]]["error_count"] = 10 + variant
                for i in range(6):
                    tickets[tids[i]]["escalation_ticket_id"] = tids[4 + variant % 2]
                    services[sids[i]]["dependency_service_id"] = sids[4 + variant % 2]
                    services[sids[i]]["response_ticket_id"] = tids[1 + variant % 3]
                tickets[tids[0]]["candidate_ticket_ids"] = [tids[i] for i in candidates]
                tickets[tids[0]]["related_ticket_id"] = tids[4]
                tickets[tids[4]]["candidate_ticket_ids"] = [tids[i] for i in candidates]
                services[sids[0]]["related_service_ids"] = [sids[i] for i in candidates]
                services[sids[0]]["dependency_service_ids"] = [sids[i] for i in candidates]
                if kind == "linked_compare" and workflow == "incident":
                    services[sids[0]]["response_ticket_id"] = tids[4]
                    services[sids[4]]["related_service_ids"] = [sids[i] for i in candidates]
                if kind == "branch_multi":
                    candidates = candidates[:2]
                    tickets[tids[0]]["candidate_ticket_ids"] = [tids[i] for i in candidates]
                    services[sids[0]]["dependency_service_ids"] = [sids[i] for i in candidates]
                if kind in {"chain", "chain_branch"}:
                    chain = [0] + candidates
                    for first, second in zip(chain, chain[1:]):
                        tickets[tids[first]]["parent_ticket_id"] = tids[second]
                task = Task(
                    task_id=task_id, template_id=template, workflow=workflow,
                    ticket_id=tids[0], service_id=sids[0], goals=goals,
                    structure=kind, instructions=instructions,
                    initial_state={"tickets": tickets, "services": services, "knowledge": policies,
                                   "incidents": [], "notifications": []}, expected_writes=[],
                )
                task.expected_writes = _expected(task)
                tasks.append(task)
    return tasks


def diversity_audit(tasks: list[Task]) -> dict:
    """Audit identifier-normalized states and, separately, non-policy state/topology."""
    grouped = defaultdict(list)
    for task in tasks:
        state = copy.deepcopy(task.initial_state)
        identifiers = sorted(set(state["tickets"]) | set(state["services"]), key=len, reverse=True)
        raw = json.dumps(state, sort_keys=True)
        for identifier in identifiers:
            prefix = "ticket" if identifier in state["tickets"] else "service"
            raw = raw.replace(identifier, prefix + "_" + identifier.rsplit("-", 1)[1])
        normalized = json.loads(raw)
        # Numerical service state / impact / topology variation is audited independently
        # from the deliberately varied recovery-policy parameters.
        structural = {key: value for key, value in normalized.items() if key != "knowledge"}
        grouped[task.template_id].append((fingerprint(normalized), fingerprint(structural)))
    return {name: {"instances": len(rows), "id_normalized_unique": len({r[0] for r in rows}),
                   "state_topology_unique": len({r[1] for r in rows})}
            for name, rows in sorted(grouped.items())}

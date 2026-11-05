from __future__ import annotations

import copy
import json
import socket
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from agentsentinel import expanded_collection as runner
from agentsentinel.backends import Completion
from agentsentinel.collection_run import events_at
from agentsentinel.config import Settings, read_json, write_json
from agentsentinel.dataset import freeze_design, validate_design
from agentsentinel.environment import Environment, FaultConfig
from agentsentinel.expanded_design import make_design, make_plans, validate_expanded_design
from agentsentinel.expanded_tasks import diversity_audit, make_expanded_tasks, parent_families
from agentsentinel.harness import run_session
from agentsentinel.tasks import SEVERITY, make_tasks


@pytest.fixture(autouse=True)
def no_network_or_env(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real network and .env access forbidden in offline verification")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Settings, "from_env", forbidden)


class ToolOracle:
    """Execute public structure semantics using only tool responses, never expected_writes."""

    def __init__(self, task):
        self.task = task
        self.environment = Environment(task, FaultConfig())
        self.calls = []

    def call(self, tool, **args):
        self.calls.append({"tool": tool, "arguments": args})
        response = self.environment.call(tool, args, len(self.calls))
        assert response["status"] == "ok"
        return response["response"]

    def ticket(self, tid):
        return self.call("get_ticket", ticket_id=tid)

    def service(self, sid):
        return self.call("get_service_status", service_id=sid)

    def write(self, goals, ticket, service, policy=None):
        healthy = service["state"] == "healthy"
        sid = service["service_id"]
        for goal in goals:
            if goal in {"status", "priority", "owner"}:
                value = {"status": "resolved" if healthy else "open",
                         "priority": "normal" if healthy else "high"}
                if goal == "owner":
                    value[goal] = policy["owner"]
                self.call("update_ticket", ticket_id=ticket["ticket_id"], field=goal, value=value[goal])
            elif goal == "note":
                self.call("append_note", ticket_id=ticket["ticket_id"],
                          note=f"diagnosis:{sid}:{service['state']}:{policy['action']}")
            elif goal == "incident":
                self.call("record_incident", service_id=sid,
                          severity="info" if healthy else "critical", action=policy["action"])
            elif goal == "notify":
                self.call("send_notification", service_id=sid, channel="audit" if healthy else "oncall",
                          body=f"service:{sid}:{service['state']}:{policy['action']}")

    def policy(self, service):
        return self.call("lookup_policy", service_id=service["service_id"])

    def execute(self):
        task, kind = self.task, self.task.structure
        goals, ticket = task.goals, None
        if task.workflow == "ticket":
            ticket = self.ticket(task.ticket_id)
            if kind in {"linked", "linked_compare"}:
                ticket = self.ticket(ticket["related_ticket_id"])
            if kind in {"compare", "linked_compare", "compare_branch"}:
                candidates = [self.ticket(t) for t in ticket["candidate_ticket_ids"]]
                ticket = sorted(candidates, key=lambda t: (-t["impact_score"], t["ticket_id"]))[0]
            if kind in {"chain", "chain_branch"}:
                while ticket["parent_ticket_id"] is not None:
                    ticket = self.ticket(ticket["parent_ticket_id"])
            service = self.service(ticket["service_id"])
            if kind in {"branch", "compare_branch", "chain_branch", "conditional_owner"}:
                if service["state"] == "healthy":
                    goals = ["status"]
                elif kind == "conditional_owner":
                    goals = ["status", "owner"]
                else:
                    ticket = self.ticket(ticket["escalation_ticket_id"])
                    service = self.service(ticket["service_id"])
                    goals = ["owner", "note"]
            if kind == "branch_multi":
                if service["state"] == "healthy":
                    goals = ["status"]
                else:
                    for tid in sorted(ticket["candidate_ticket_ids"]):
                        candidate = self.ticket(tid)
                        state = self.service(candidate["service_id"])
                        if state["state"] != "healthy":
                            self.write(["status"], candidate, state)
                    return self.calls
        else:
            service = self.service(task.service_id)
            if kind in {"linked", "linked_compare"}:
                ticket = self.ticket(service["response_ticket_id"])
                service = self.service(ticket["service_id"])
            if kind in {"compare", "linked_compare", "compare_dependency"}:
                candidates = [self.service(s) for s in service["related_service_ids"]]
                service = sorted(candidates, key=lambda s: (
                    -SEVERITY[s["state"]], -s["error_count"], s["service_id"],
                ))[0]
            if kind == "compare_dependency":
                service = self.service(service["dependency_service_id"])
            if kind == "branch":
                if service["state"] == "healthy":
                    goals = ["notify"]
                else:
                    service = self.service(service["dependency_service_id"])
            if kind == "conditional_response":
                goals = ["incident"]
                if service["state"] != "healthy":
                    ticket = self.ticket(service["response_ticket_id"])
                    goals.append("note")
            if kind == "branch_multi":
                if service["state"] == "healthy":
                    goals = ["notify"]
                else:
                    for sid in sorted(service["dependency_service_ids"]):
                        state = self.service(sid)
                        self.write(["incident", "notify"], None, state, self.policy(state))
                    return self.calls
            if kind in {"aggregate", "aggregate_filter"}:
                selected = [service]
                for sid in service["dependency_service_ids"]:
                    state = self.service(sid)
                    if kind == "aggregate" or state["state"] != "healthy":
                        selected.append(state)
                items = []
                for state in sorted(selected, key=lambda s: s["service_id"]):
                    policy = self.policy(state)
                    items.append(f"{state['service_id']}={state['state']}:{policy['action']}")
                healthy = all(s["state"] == "healthy" for s in selected)
                summary = ";".join(items)
                self.call("record_incident", service_id=service["service_id"],
                          severity="info" if healthy else "critical", action="aggregate:" + summary)
                self.call("send_notification", service_id=service["service_id"],
                          channel="audit" if healthy else "oncall",
                          body=f"aggregate:{service['service_id']}:{summary}")
                return self.calls
        policy = self.policy(service) if set(goals) - {"status", "priority"} else None
        self.write(goals, ticket, service, policy)
        return self.calls


def test_catalog_diversity_and_every_task_feasible():
    tasks = make_expanded_tasks()
    assert len(tasks) == len({t.task_id for t in tasks}) == 400
    assert Counter(t.workflow for t in tasks) == {"ticket": 200, "incident": 200}
    assert set(t.task_id for t in tasks).isdisjoint(t.task_id for t in make_tasks())
    assert all(v == {"instances": 20, "id_normalized_unique": 20, "state_topology_unique": 20}
               for v in diversity_audit(tasks).values())
    max_calls = 0
    semantic_traces = {}
    for task in tasks:
        oracle = ToolOracle(task)
        oracle.execute()
        assert oracle.environment.verify("completed")["success"], task.task_id
        assert len(oracle.calls) <= 10, (task.task_id, len(oracle.calls))
        max_calls = max(max_calls, len(oracle.calls))
        semantic_traces.setdefault(task.template_id, set()).add(tuple(c["tool"] for c in oracle.calls))
    assert max_calls == 10
    # New combinations are not merely renamed parent templates.
    for workflow in ("ticket", "incident"):
        for index in range(5):
            assert semantic_traces[f"v2-{workflow}-{index:02d}"] != semantic_traces[f"v2-{workflow}-{index + 5:02d}"]


def test_schema_splits_plans_and_tamper(tmp_path):
    parent = freeze_design(tmp_path / "parent.json")
    design = make_design(parent)
    assert validate_design(parent) == parent
    assert Counter(design["assignments"].values()) == {"train": 12, "validation": 4, "test": 4}
    for name, parent_name in parent_families().items():
        assert design["assignments"][name] == parent["assignments"][parent_name]
    assert sorted(t for fold in design["cv_folds"] for t in fold) == sorted(
        t for t, split in design["assignments"].items() if split == "train"
    )
    for fold in design["cv_folds"]:
        assert len(fold) == 4
        assert Counter(t.split("-")[1] for t in fold) == {"ticket": 2, "incident": 2}
        for template in fold:
            family = design["parent_families"][template]
            assert sum(design["parent_families"][t] == family for t in fold) == 2
    plans = make_plans(design)
    assert len(plans["probe"]) == 5 and len(plans["collection"]) == 6000
    assert len({s["condition_id"] for plan in plans.values() for s in plan}) == 6005
    tasks = {t["task_id"]: t for t in design["tasks"]}
    assert {tasks[s["task_id"]]["workflow"] for s in plans["probe"]} == {"ticket", "incident"}
    assert len({s["condition"] for s in plans["probe"]}) == 5
    assert all(design["assignments"][tasks[s["task_id"]]["template_id"]] == "train"
               for s in plans["probe"])
    assert set(Counter((s["task_id"], s["condition"]) for s in plans["collection"]).values()) == {3}
    for start in range(0, 6000, 20):
        block = plans["collection"][start:start + 20]
        assert len({tasks[s["task_id"]]["template_id"] for s in block}) == 20
    changed = copy.deepcopy(design)
    changed["assignments"][next(iter(changed["assignments"]))] = "test"
    changed = runner.seal(changed)
    with pytest.raises(ValueError):
        validate_expanded_design(changed, parent)
    with pytest.raises(ValueError):
        validate_design(design)  # No accidental activation of old downstream modeling.


@pytest.fixture
def expanded_root(tmp_path, monkeypatch):
    old = tmp_path / "old"
    parent = freeze_design(old / "design.json")
    settings = Settings(model="offline-model", currency="CNY", input_price=13,
                        cached_input_price=1.3, output_price=68)
    write_json(old / "frozen_collection.json", runner.seal({
        "design_hash": parent["fingerprint"], "settings": settings.model_dump(),
        "fault_design": FaultConfig().model_dump(exclude={"kind", "seed"}),
    }))
    monkeypatch.setattr(runner, "credentials", lambda settings: None)
    root = tmp_path / "expanded"
    runner.initialize(root, old)
    return root


@pytest.fixture
def fake_sessions(monkeypatch):
    tracker = {"count": 0, "active": 0, "max_active": 0, "failure": None}
    lock = threading.Lock()

    class FakeBackend:
        def __init__(self, settings):
            pass

        def complete(self, *args):
            return Completion({"role": "assistant", "content": "offline synthetic completion"},
                              10, 5, 0, 0)

    monkeypatch.setattr("agentsentinel.harness.OpenAIBackend", FakeBackend)

    def fake_run(*args, **kwargs):
        root = args[3].parent
        ledger = runner.checked(root / "ledger.json")
        assert ledger["entries"][kwargs["condition_id"]]["state"] == "started"
        with lock:
            tracker["count"] += 1
            tracker["active"] += 1
            tracker["max_active"] = max(tracker["max_active"], tracker["active"])
        try:
            time.sleep(0.005)
            if tracker["failure"] == "runner":
                raise RuntimeError("isolated synthetic runner failure")
            result = run_session(*args, **kwargs)
            if tracker["failure"]:
                directory = Path(result["path"])
                result["valid_for_analysis"] = False
                result["termination"] = "backend_error"
                result["failure_detail"] = {
                    "auth": "HTTP 401", "forbidden": "HTTP 403", "rate_limit": "HTTP 429",
                    "infra": "LLM transport/protocol error: ReadTimeout",
                }[tracker["failure"]]
                result["totals"]["unknown_usage_requests"] = 1
                events = events_at(directory)
                events[-1]["result"] = {k: v for k, v in result.items() if k != "path"}
                write_json(directory / "result.json", events[-1]["result"])
                (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
            return result
        finally:
            with lock:
                tracker["active"] -= 1
    monkeypatch.setattr(runner, "run_session", fake_run)
    return tracker


CONFIRM = "REVIEWED_PROBE_AND_APPROVE_ONE_BATCH"


def test_probe_batch_caps_and_consumed_approval(expanded_root, fake_sessions):
    root = expanded_root
    with pytest.raises(ValueError):
        runner.run_batch(root, "probe", workers=3)
    with pytest.raises(ValueError):
        runner.approve(root, 50, CONFIRM)
    result = runner.run_batch(root, "probe")
    assert result["completed"] == {"probe": 5, "collection": 0}
    assert fake_sessions["count"] == 5 and fake_sessions["max_active"] <= 2
    runner.run_batch(root, "probe")
    assert fake_sessions["count"] == 5
    for count in (0, 51):
        with pytest.raises(ValueError):
            runner.approve(root, count, CONFIRM)
    with pytest.raises(ValueError):
        runner.approve(root, 50, "not confirmed")
    approval = runner.approve(root, 50, CONFIRM)
    with pytest.raises(ValueError):
        runner.run_batch(root, "collection", "wrong")
    result = runner.run_batch(root, "collection", approval["approval_token"])
    assert result["status"] == "batch_complete"
    assert result["completed"] == {"probe": 5, "collection": 50}
    assert fake_sessions["count"] == 55 and fake_sessions["max_active"] <= 2
    with pytest.raises(ValueError):
        runner.run_batch(root, "collection", approval["approval_token"])
    assert fake_sessions["count"] == 55
    assert result["known_cost_estimate_cny"] > 0
    assert not list(root.rglob("*.md"))
    for directory in (root / "probe").iterdir():
        if directory.is_dir():
            assert {p.name for p in directory.iterdir()} == {
                "session.json", "events.jsonl", "trajectory.json", "result.json",
            }


@pytest.mark.parametrize("target", ["source", "plan", "design", "protocol", "ledger"])
def test_freeze_tamper_blocks_before_request(expanded_root, fake_sessions, monkeypatch, target):
    root = expanded_root
    if target == "source":
        monkeypatch.setattr(runner, "code_hashes", lambda: {"changed.py": "changed"})
    else:
        path = root / {"plan": "probe/plan.json", "design": "design.json",
                       "protocol": "protocol.json", "ledger": "ledger.json"}[target]
        data = read_json(path)
        if target == "plan":
            data.pop()
        else:
            data["tampered"] = True
        write_json(path, data)
    with pytest.raises(ValueError):
        runner.run_batch(root, "probe")
    assert fake_sessions["count"] == 0


def test_ambiguous_never_retried(expanded_root, fake_sessions):
    root = expanded_root
    ledger = runner.checked(root / "ledger.json")
    first = next(iter(ledger["entries"]))
    ledger["entries"][first] = {"state": "started"}
    runner.persist(root, ledger)
    for _ in range(2):
        with pytest.raises(ValueError):
            runner.run_batch(root, "probe")
    assert fake_sessions["count"] == 0
    assert runner.checked(root / "ledger.json")["entries"][first]["state"] == "ambiguous_do_not_retry"
    assert runner.status(root)["ambiguous_attempts"] == [first]


def test_complete_attempt_recovery_no_rerun(expanded_root, fake_sessions):
    root = expanded_root
    runner.run_batch(root, "probe")
    ledger = runner.checked(root / "ledger.json")
    first = next(iter(ledger["entries"]))
    ledger["entries"][first] = {"state": "started"}
    runner.persist(root, ledger)
    runner.run_batch(root, "probe")
    assert fake_sessions["count"] == 5
    assert runner.checked(root / "ledger.json")["entries"][first]["recovered"]


@pytest.mark.parametrize("failure,maximum", [("runner", 2), ("auth", 2), ("infra", 4)])
def test_pause_and_drain_no_auto_refill(expanded_root, fake_sessions, failure, maximum):
    fake_sessions["failure"] = failure
    result = runner.run_batch(expanded_root, "probe")
    assert result["status"] == "paused"
    assert fake_sessions["count"] <= maximum and fake_sessions["active"] == 0
    count = fake_sessions["count"]
    with pytest.raises(ValueError):
        runner.run_batch(expanded_root, "probe")
    assert fake_sessions["count"] == count


def test_invalid_probe_and_completed_artifact_corruption_block(expanded_root, fake_sessions):
    root = expanded_root
    runner.run_batch(root, "probe")
    ledger = runner.checked(root / "ledger.json")
    entry = next(e for e in ledger["entries"].values() if e["state"] == "completed")
    path = Path(entry["path"]) / "trajectory.json"
    trajectory = read_json(path)
    trajectory["final_state"]["incidents"].append({"isolated_test": True})
    write_json(path, trajectory)
    with pytest.raises(ValueError):
        runner.approve(root, 50, CONFIRM)
    assert runner.checked(root / "ledger.json")["status"] == "paused"
    assert fake_sessions["count"] == 5


def test_recent_invalid_window_persists_across_batches(expanded_root, fake_sessions):
    root = expanded_root
    runner.run_batch(root, "probe")
    first = runner.approve(root, 2, CONFIRM)
    fake_sessions["failure"] = "infra"
    result = runner.run_batch(root, "collection", first["approval_token"], workers=1)
    assert result["status"] == "batch_complete"
    second = runner.approve(root, 50, CONFIRM)
    result = runner.run_batch(root, "collection", second["approval_token"], workers=1)
    assert result["status"] == "paused"
    assert result["completed"] == {"probe": 5, "collection": 3}
    assert result["unknown_usage_requests"] == 3
    assert result["known_cost_estimate_cny"] > 0


def test_partial_usage_accounted_without_retry(expanded_root, fake_sessions):
    root = expanded_root
    protocol, design, plans, ledger = runner.load(root)
    spec = plans["probe"][0]
    directory = root / "probe" / "isolated_partial"
    write_json(directory / "session.json", {"condition_id": spec["condition_id"]})
    events = [{"kind": "llm_start"}, {"kind": "llm_end", "usage": {
        "input_tokens": 1000, "output_tokens": 100, "cached_tokens": 0,
    }}, {"kind": "llm_start"}]
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    ledger["entries"][spec["condition_id"]] = {"state": "started"}
    runner.persist(root, ledger)
    with pytest.raises(ValueError):
        runner.run_batch(root, "probe")
    result = runner.status(root)
    assert result["known_cost_estimate_cny"] == pytest.approx(0.0198)
    assert result["unknown_usage_requests"] == 1
    assert fake_sessions["count"] == 0


def test_root_lock_status_no_auth_and_model_mismatch(expanded_root, monkeypatch):
    with runner.root_lock(expanded_root):
        with pytest.raises(ValueError):
            runner.status(expanded_root)
    assert runner.status(expanded_root)["completed"] == {"probe": 0, "collection": 0}
    # Restore the real checker only with a stubbed in-memory environment, never .env.
    monkeypatch.setattr(Settings, "from_env", lambda: Settings(model="other-model"))
    from importlib import reload
    reloaded = reload(runner)
    with pytest.raises(ValueError, match="model/endpoint"):
        reloaded.credentials(Settings(model="frozen-model"))


@pytest.fixture
def auto_root(expanded_root):
    root = expanded_root.parent / "automatic"
    runner.initialize(root, expanded_root.parent / "old", mode="auto")
    return root


def test_auto_freeze_independent_collection_only(auto_root, expanded_root):
    protocol, design, plans, ledger = runner.load(auto_root)
    old_protocol, old_design, old_plans, old_ledger = runner.load(expanded_root)
    assert protocol["mode"] == "auto"
    assert protocol["workers"] == protocol["batch_cap"] == 50
    assert protocol["manual_batch_gate"] is False
    assert protocol["monetary_cap_cny"] is None
    assert protocol["probe_cap"] == protocol["probe_sessions_authorized"] == 0
    assert protocol["collection_sessions_authorized"] == 6000
    assert {k: protocol["settings"][k] for k in runner.AUTO_SETTINGS} == runner.AUTO_SETTINGS
    assert design == old_design
    assert plans == {"collection": old_plans["collection"]}
    assert len(ledger["entries"]) == 6000
    assert all(e == {"state": "pending"} for e in ledger["entries"].values())
    assert old_protocol["settings"]["max_tokens"] == 30000
    assert old_ledger["status"] == "ready"
    assert not (auto_root / "probe").exists()
    for operation in (lambda: runner.run_batch(auto_root, "probe"),
                      lambda: runner.approve(auto_root, 50, CONFIRM),
                      lambda: runner.collect_all(expanded_root)):
        with pytest.raises(ValueError):
            operation()
    assert runner.load(expanded_root)[3] == old_ledger


@pytest.mark.parametrize("field,value", [
    ("workers", 49), ("batch_cap", 51), ("manual_batch_gate", True), ("mode", "unknown"),
    ("probe_cap", 5), ("automatic_api_retries", 1), ("monetary_cap_cny", 100),
    ("collection_sessions_authorized", 6005), ("authorization", "different"),
    ("max_tokens", 30000), ("session_timeout", 120), ("max_completion_tokens", 1024),
    ("request_timeout", 120), ("max_steps", 13), ("max_model_calls", 25),
])
def test_auto_rejects_resealed_invalid_configuration(auto_root, field, value):
    protocol = runner.checked(auto_root / "protocol.json")
    target = protocol["settings"] if field in runner.AUTO_SETTINGS else protocol
    target[field] = value
    runner.save(auto_root / "protocol.json", runner.seal(protocol))
    with pytest.raises(ValueError):
        runner.load(auto_root)


@pytest.mark.parametrize("failure", ["runner", "auth", "forbidden", "rate_limit", "infra"])
def test_auto_pauses_and_never_starts_next_batch(auto_root, fake_sessions, failure):
    fake_sessions["failure"] = failure
    result = runner.collect_all(auto_root)
    assert result["status"] == "paused"
    assert fake_sessions["count"] == 50
    assert fake_sessions["active"] == 0
    ledger = runner.checked(auto_root / "ledger.json")
    assert len(ledger["batches"]) == 1
    assert sum(e["state"] == "pending" for e in ledger["entries"].values()) == 5950
    assert read_json(auto_root / "summary.json")["status"] == "paused"
    with pytest.raises(ValueError, match="Paused"):
        runner.collect_all(auto_root)
    assert fake_sessions["count"] == 50


def test_auto_init_never_changes_paused_manual_or_parent(expanded_root, fake_sessions):
    ledger = runner.checked(expanded_root / "ledger.json")
    ledger["status"] = "paused"
    runner.persist(expanded_root, ledger)
    parent = expanded_root.parent / "old"
    before = {p: p.read_bytes() for root in (parent, expanded_root)
              for p in root.rglob("*") if p.is_file()}
    root = expanded_root.parent / "fresh_auto"
    runner.initialize(root, parent, mode="auto")
    with pytest.raises(ValueError):
        runner.collect_all(expanded_root)
    assert {p: p.read_bytes() for p in before} == before
    assert runner.checked(expanded_root / "ledger.json")["status"] == "paused"
    assert fake_sessions["count"] == 0
    with pytest.raises(ValueError):
        runner.initialize(root, parent, mode="auto")
    with pytest.raises(ValueError):
        runner.initialize(parent / "nested", parent, mode="auto")


def test_auto_fifty_concurrent_and_strict_batch_boundary(auto_root, monkeypatch):
    barrier = threading.Barrier(50, timeout=30)
    lock = threading.Lock()
    tracker = {"active": 0, "maximum": 0, "calls": 0, "finished": 0}
    protocol, design, plans, _ = runner.load(auto_root)
    positions = {s["condition_id"]: i for i, s in enumerate(plans["collection"])}

    class OfflineBackend:
        def __init__(self, settings):
            assert settings.max_tokens is settings.session_timeout is None

        def complete(self, *args):
            return Completion({"role": "assistant", "content": "isolated synthetic fixture"}, 10, 5)

    monkeypatch.setattr("agentsentinel.harness.OpenAIBackend", OfflineBackend)

    def session(*args, **kwargs):
        index = positions[kwargs["condition_id"]]
        assert index < 100
        ledger = runner.checked(auto_root / "ledger.json")
        assert ledger["entries"][kwargs["condition_id"]]["state"] == "started"
        assert len(ledger["batches"]) == index // 50 + 1
        if index >= 50:
            assert tracker["finished"] >= 50
            assert read_json(auto_root / "summary.json")["completed"]["collection"] == 50
            assert ledger["batches"][0]["status"] == "batch_complete"
        with lock:
            tracker["calls"] += 1
            tracker["active"] += 1
            tracker["maximum"] = max(tracker["maximum"], tracker["active"])
        try:
            with pytest.raises(ValueError, match="lock"):
                with runner.root_lock(auto_root):
                    pass
            barrier.wait()
            if index >= 50:
                raise RuntimeError("isolated second-batch stop")
            return run_session(*args, **kwargs)
        finally:
            with lock:
                tracker["active"] -= 1
                tracker["finished"] += 1

    monkeypatch.setattr(runner, "run_session", session)
    result = runner.collect_all(auto_root)
    assert tracker == {"active": 0, "maximum": 50, "calls": 100, "finished": 100}
    assert result["completed"] == {"collection": 50}
    assert result["status"] == "paused"
    assert len(runner.checked(auto_root / "ledger.json")["batches"]) == 2


def test_auto_all_6000_scheduled_in_120_batches(auto_root, monkeypatch):
    batches = []

    def isolated_batch(root, phase, protocol, design, plans, ledger, chosen, settings, workers):
        assert phase == "collection" and workers == 50 and len(chosen) == 50
        assert all(e["state"] in {"pending", "completed"} for e in ledger["entries"].values())
        batches.append([s["condition_id"] for s in chosen])
        for spec in chosen:
            ledger["entries"][spec["condition_id"]] = {"state": "completed"}
        ledger["status"] = "batch_complete"

    monkeypatch.setattr(runner, "execute_batch_locked", isolated_batch)
    monkeypatch.setattr(runner, "summary_locked", lambda root, protocol, design, plans, ledger:
                        {"status": ledger["status"]})
    assert runner.collect_all(auto_root) == {"status": "complete"}
    assert len(batches) == 120
    assert len({cid for batch in batches for cid in batch}) == 6000
    assert runner.checked(auto_root / "ledger.json")["status"] == "complete"


@pytest.mark.parametrize("state,status", [("started", "running"), ("pending", "paused"),
                                         ("ambiguous_do_not_retry", "ready")])
def test_auto_recovery_never_crosses_pause_or_ambiguous(auto_root, fake_sessions, state, status):
    ledger = runner.checked(auto_root / "ledger.json")
    first = next(iter(ledger["entries"]))
    ledger["entries"][first] = {"state": state}
    ledger["status"] = status
    runner.persist(auto_root, ledger)
    with pytest.raises(ValueError, match="Paused"):
        runner.collect_all(auto_root)
    assert fake_sessions["count"] == 0
    assert runner.checked(auto_root / "ledger.json")["status"] == "paused"


def test_simulated_tool_rate_limit_does_not_pause(tmp_path):
    # This fixture is synthetic, not an observation of a provider response.
    task = make_tasks()[0]
    result = run_session(task, Settings(max_tokens=None, session_timeout=None), FaultConfig(
        kind="rate_limit", probability=1, persistent_probability=1,
    ), tmp_path, agent_seed=2)
    assert any(e["kind"] == "fault_injected" for e in events_at(Path(result["path"])))
    ledger = {"recent_invalid": [], "status": "running"}
    for _ in range(10):
        runner.record_health(ledger, result)
    assert ledger["status"] == "running"


def test_auto_recover_complete_attempt_preserves_original_batch_boundary(
    auto_root, fake_sessions, monkeypatch,
):
    protocol, design, plans, ledger = runner.load(auto_root)
    spec = plans["collection"][0]
    ledger["entries"][spec["condition_id"]] = {"state": "started"}
    ledger["status"] = "running"
    ledger["batches"] = [{"status": "running", "phase": "collection",
                          "ids": [s["condition_id"] for s in plans["collection"][:50]]}]
    runner.persist(auto_root, ledger)
    task = next(t for t in make_expanded_tasks() if t.task_id == spec["task_id"])
    runner.run_session(
        task, Settings.model_validate(protocol["settings"]),
        FaultConfig(kind=spec["condition"], seed=spec["fault_seed"], **protocol["fault_design"]),
        auto_root / "collection", "openai", spec["agent_seed"], "collection",
        experiment_id=auto_root.name + "_collection", condition_id=spec["condition_id"],
        design_hash=design["fingerprint"],
    )
    chosen_ids = []

    def stop_after_recovery(root, phase, protocol, design, plans, ledger, chosen, settings, workers):
        assert ledger["entries"][spec["condition_id"]]["recovered"]
        assert ledger["batches"][0]["status"] == "interrupted_recovered"
        chosen_ids.extend(s["condition_id"] for s in chosen)
        ledger["status"] = "paused"
        return {"status": "paused"}

    monkeypatch.setattr(runner, "execute_batch_locked", stop_after_recovery)
    assert runner.collect_all(auto_root)["status"] == "paused"
    assert chosen_ids == [s["condition_id"] for s in plans["collection"][1:50]]
    assert fake_sessions["count"] == 1


def test_auto_final_batch_persists_complete(auto_root, fake_sessions):
    protocol, design, plans, ledger = runner.load(auto_root)
    # A one-row isolated scheduler fixture exercises the real finalization and artifact audit.
    chosen = plans["collection"][:1]
    plans = {"collection": chosen}
    ledger["entries"] = {chosen[0]["condition_id"]: {"state": "pending"}}
    with runner.root_lock(auto_root):
        result = runner.execute_batch_locked(
            auto_root, "collection", protocol, design, plans, ledger, chosen,
            Settings.model_validate(protocol["settings"]), 50,
        )
    assert result["status"] == "complete"
    assert result["completed"] == {"collection": 1}
    assert runner.checked(auto_root / "ledger.json")["status"] == "complete"
    assert read_json(auto_root / "summary.json")["status"] == "complete"


@pytest.mark.parametrize("failure", ["auth", "forbidden", "rate_limit", "infra", "runner"])
def test_explicit_resume_preserves_history_and_pauses_again(auto_root, fake_sessions, monkeypatch,
                                                          failure):
    fake_sessions["failure"] = "infra"
    assert runner.collect_all(auto_root)["status"] == "paused"
    before = runner.checked(auto_root / "ledger.json")
    protocol_bytes = (auto_root / "protocol.json").read_bytes()
    artifacts = {p: p.read_bytes() for p in (auto_root / "collection").rglob("*") if p.is_file()}
    real_execute = runner.execute_batch_locked

    def checked_execute(root, phase, protocol, design, plans, ledger, chosen, settings, workers):
        with pytest.raises(ValueError, match="lock"):
            with runner.root_lock(root):
                pass
        assert ledger["recent_invalid"] == []
        assert all(before["entries"][s["condition_id"]]["state"] == "pending" for s in chosen)
        saved = runner.checked(root / "ledger.json")
        assert saved["resumptions"][-1]["confirmation"] == runner.RESUME_CONFIRM
        return real_execute(root, phase, protocol, design, plans, ledger, chosen, settings, workers)

    monkeypatch.setattr(runner, "execute_batch_locked", checked_execute)
    fake_sessions["failure"] = failure
    assert runner.collect_all(auto_root, resume=True, confirm=runner.RESUME_CONFIRM)["status"] == "paused"
    after = runner.checked(auto_root / "ledger.json")
    assert fake_sessions["count"] == 100
    assert after["batches"][:1] == before["batches"]
    assert all(after["entries"][cid] == entry for cid, entry in before["entries"].items()
               if entry["state"] == "completed")
    record = after["resumptions"][0]
    assert record["recent_invalid"] == before["recent_invalid"]
    assert record["previous_status"] == "paused" and record["resumed_at"]
    assert record["code_hashes"] == runner.code_hashes()
    assert (auto_root / "protocol.json").read_bytes() == protocol_bytes
    assert all(p.read_bytes() == content for p, content in artifacts.items())


@pytest.mark.parametrize("state", ["started", "ambiguous_do_not_retry", "completed"])
def test_resume_rejects_unresolved_or_corrupt_attempt(auto_root, fake_sessions, state):
    ledger = runner.checked(auto_root / "ledger.json")
    first = next(iter(ledger["entries"]))
    ledger["entries"][first] = {"state": state, "path": str(auto_root / "collection/missing")}
    ledger["status"] = "paused"
    runner.persist(auto_root, ledger)
    with pytest.raises(ValueError, match="unresolved"):
        runner.collect_all(auto_root, resume=True, confirm=runner.RESUME_CONFIRM)
    assert fake_sessions["count"] == 0
    after = runner.checked(auto_root / "ledger.json")
    assert after["entries"][first]["state"] == "ambiguous_do_not_retry"
    assert "resumptions" not in after


@pytest.mark.parametrize("status", ["ready", "running", "batch_complete", "complete"])
def test_resume_requires_preexisting_pause(auto_root, fake_sessions, status):
    ledger = runner.checked(auto_root / "ledger.json")
    ledger["status"] = status
    runner.persist(auto_root, ledger)
    with pytest.raises(ValueError, match="already paused"):
        runner.collect_all(auto_root, resume=True, confirm=runner.RESUME_CONFIRM)
    assert fake_sessions["count"] == 0


@pytest.mark.parametrize("confirm", [None, "yes", CONFIRM])
def test_resume_requires_exact_confirmation(auto_root, fake_sessions, confirm):
    before = (auto_root / "ledger.json").read_bytes()
    with pytest.raises(ValueError, match="confirmation"):
        runner.collect_all(auto_root, resume=True, confirm=confirm)
    assert (auto_root / "ledger.json").read_bytes() == before
    assert fake_sessions["count"] == 0


def test_resume_rejects_manual_root(expanded_root, fake_sessions):
    with pytest.raises(ValueError, match="auto root"):
        runner.collect_all(expanded_root, resume=True, confirm=runner.RESUME_CONFIRM)
    assert fake_sessions["count"] == 0


@pytest.mark.parametrize("runner_hash", [runner.PRE_RESUME_RUNNER_HASH, runner.PRE_TRIAL_RUNNER_HASH])
def test_resume_source_compatibility_is_exact(auto_root, runner_hash):
    protocol = runner.checked(auto_root / "protocol.json")
    protocol["code_hashes"]["expanded_collection.py"] = runner_hash
    runner.save(auto_root / "protocol.json", runner.seal(protocol))
    ledger = runner.checked(auto_root / "ledger.json")
    ledger["protocol_hash"] = runner.checked(auto_root / "protocol.json")["fingerprint"]
    runner.persist(auto_root, ledger)
    runner.load(auto_root)
    protocol["code_hashes"]["harness.py"] = "unknown"
    runner.save(auto_root / "protocol.json", runner.seal(protocol))
    with pytest.raises(ValueError, match="Source snapshot"):
        runner.load(auto_root)


def test_resume_cli_forwards_confirmation(auto_root, monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(runner, "collect_all", lambda root, **kwargs: calls.append((root, kwargs)))
    monkeypatch.setattr(sys, "argv", ["expanded_collection", "collect-all", "--root", str(auto_root),
                                    "--resume", "--confirm", runner.RESUME_CONFIRM])
    runner.main()
    assert calls == [(auto_root, {"resume": True, "confirm": runner.RESUME_CONFIRM})]
    monkeypatch.setattr(sys, "argv", ["expanded_collection", "status", "--root", str(auto_root),
                                    "--resume", "--confirm", runner.RESUME_CONFIRM])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_auto_summary_failure_pauses_before_next_batch(auto_root, fake_sessions, monkeypatch):
    def failed_summary(*args):
        raise OSError("isolated summary write failure")

    monkeypatch.setattr(runner, "summary_locked", failed_summary)
    with pytest.raises(OSError):
        runner.collect_all(auto_root)
    ledger = runner.checked(auto_root / "ledger.json")
    assert ledger["status"] == ledger["batches"][-1]["status"] == "paused"
    assert len(ledger["batches"]) == 1
    assert fake_sessions["count"] == 50


@pytest.mark.parametrize("resume", [False, True])
def test_trial_ten_only_skips_500_and_audits_override(auto_root, fake_sessions, monkeypatch, resume):
    protocol, design, plans, ledger = runner.load(auto_root)
    completed_ids = {s["condition_id"] for s in plans["collection"][:500]}
    for cid in completed_ids:
        ledger["entries"][cid] = {"state": "completed", "path": "isolated_prior_fixture"}
    ledger["status"] = "paused" if resume else "batch_complete"
    ledger["recent_invalid"] = [True] if resume else []
    runner.persist(auto_root, ledger)
    previous_entries = copy.deepcopy(ledger["entries"])
    frozen = {p: p.read_bytes() for p in (auto_root / "protocol.json", auto_root / "design.json",
                                         auto_root / "collection/plan.json")}
    real_audit, real_session = runner.audit, runner.run_session
    synthetic_prior = {
        "valid_for_analysis": True, "elapsed_ms": 0, "success": True, "termination": "completed",
        "totals": dict.fromkeys(("model_calls", "total_tokens", "known_input_tokens",
                                 "known_output_tokens", "known_cost_estimate",
                                 "unknown_usage_requests", "unknown_cache_requests"), 0),
    }

    def audit(directory, root, phase, protocol, design, spec):
        if spec["condition_id"] in completed_ids:
            return synthetic_prior, []
        return real_audit(directory, root, phase, protocol, design, spec)

    barrier = threading.Barrier(10, timeout=30)
    seen = []

    def session(*args, **kwargs):
        seen.append(kwargs["condition_id"])
        barrier.wait()
        return real_session(*args, **kwargs)

    monkeypatch.setattr(runner, "audit", audit)
    monkeypatch.setattr(runner, "run_session", session)
    result = runner.collect_all(auto_root, resume=resume,
                                confirm=runner.RESUME_CONFIRM if resume else None,
                                trial_count=10, trial_workers=10)
    assert result["status"] == "batch_complete"
    assert result["completed"] == {"collection": 510}
    assert fake_sessions["count"] == 10 and fake_sessions["active"] == 0
    expected = [s["condition_id"] for s in plans["collection"][500:510]]
    assert sorted(seen) == sorted(expected)
    after = runner.checked(auto_root / "ledger.json")
    assert len(after["batches"]) == 1
    assert after["batches"][0]["ids"] == expected
    assert all(after["entries"][cid] == previous_entries[cid] for cid in completed_ids)
    assert sum(e["state"] == "pending" for e in after["entries"].values()) == 5490
    records = after["batches"] + after.get("resumptions", [])
    for record in records:
        assert record["workers"] == record["count"] == record["trial_count"] == 10
        assert record["frozen_workers"] == record["frozen_batch_cap"] == 50
        assert "50 -> 10" in record["configuration_override"]
        assert record["code_hashes"] == runner.code_hashes()
    if resume:
        assert len(after["resumptions"]) == 1
        assert after["resumptions"][0]["recent_invalid"] == [True]
    assert all(p.read_bytes() == content for p, content in frozen.items())


@pytest.mark.parametrize("count,workers", [(10, None), (None, 10), (0, 1), (51, 10),
                                           (10, 11), (10, 0), (True, 1), (10, 1.5)])
def test_trial_invalid_arguments_do_not_touch_root(auto_root, fake_sessions, count, workers):
    before = (auto_root / "ledger.json").read_bytes()
    with pytest.raises(ValueError, match="trial-count"):
        runner.collect_all(auto_root, trial_count=count, trial_workers=workers)
    assert (auto_root / "ledger.json").read_bytes() == before
    assert fake_sessions["count"] == 0


@pytest.mark.parametrize("failure", ["runner", "auth", "forbidden", "rate_limit", "infra"])
def test_trial_preserves_pause_and_drain(auto_root, fake_sessions, failure):
    fake_sessions["failure"] = failure
    result = runner.collect_all(auto_root, trial_count=10, trial_workers=1)
    assert result["status"] == "paused"
    assert fake_sessions["count"] == (3 if failure == "infra" else 1)
    assert fake_sessions["active"] == 0
    ledger = runner.checked(auto_root / "ledger.json")
    assert len(ledger["batches"]) == 1 and ledger["batches"][0]["workers"] == 1
    before = fake_sessions["count"]
    with pytest.raises(ValueError, match="Paused"):
        runner.collect_all(auto_root, trial_count=10, trial_workers=10)
    assert fake_sessions["count"] == before


@pytest.mark.parametrize("state", ["unknown", "started", "ambiguous_do_not_retry"])
def test_trial_resume_blocks_unknown_and_ambiguous(auto_root, fake_sessions, state):
    ledger = runner.checked(auto_root / "ledger.json")
    ledger["entries"][next(iter(ledger["entries"]))] = {"state": state}
    ledger["status"] = "paused"
    runner.persist(auto_root, ledger)
    with pytest.raises(ValueError, match="Unknown ledger state|unresolved"):
        runner.collect_all(auto_root, resume=True, confirm=runner.RESUME_CONFIRM,
                           trial_count=10, trial_workers=10)
    assert fake_sessions["count"] == 0
    assert "resumptions" not in runner.checked(auto_root / "ledger.json")


def test_trial_preserves_partial_frozen_batch_boundary(auto_root, fake_sessions, monkeypatch):
    protocol, design, plans, ledger = runner.load(auto_root)
    for spec in plans["collection"][:47]:
        ledger["entries"][spec["condition_id"]] = {"state": "completed"}
    calls = []
    monkeypatch.setattr(runner, "load", lambda root: (protocol, design, plans, ledger))
    monkeypatch.setattr(runner, "recover", lambda *args: None)

    def execute(root, phase, protocol, design, plans, ledger, chosen, settings, workers, trial_count):
        calls.append([s["condition_id"] for s in chosen])
        assert workers == trial_count == 10
        ledger["status"] = "batch_complete"
        return {"status": "batch_complete"}

    monkeypatch.setattr(runner, "execute_batch_locked", execute)
    assert runner.collect_all(auto_root, trial_count=10, trial_workers=10)["status"] == "batch_complete"
    assert calls == [[s["condition_id"] for s in plans["collection"][47:50]]]
    assert fake_sessions["count"] == 0


def test_auto_workers_override_requires_trial(auto_root, fake_sessions):
    protocol, design, plans, ledger = runner.load(auto_root)
    with pytest.raises(ValueError, match="requires bounded trial"):
        runner.execute_batch_locked(auto_root, "collection", protocol, design, plans, ledger,
                                    plans["collection"][:10],
                                    Settings.model_validate(protocol["settings"]), 10)
    assert fake_sessions["count"] == 0
    assert not runner.checked(auto_root / "ledger.json")["batches"]


def test_trial_unknown_runner_hash_rejected(auto_root):
    protocol = runner.checked(auto_root / "protocol.json")
    protocol["code_hashes"]["expanded_collection.py"] = "unknown"
    runner.save(auto_root / "protocol.json", runner.seal(protocol))
    with pytest.raises(ValueError, match="Source snapshot"):
        runner.load(auto_root)


def test_trial_cli_forwarding_and_pair_validation(auto_root, monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(runner, "collect_all", lambda root, **kwargs: calls.append((root, kwargs)))
    base = ["expanded_collection", "collect-all", "--root", str(auto_root)]
    monkeypatch.setattr(sys, "argv", base + ["--trial-count", "10", "--trial-workers", "10",
                                           "--resume", "--confirm", runner.RESUME_CONFIRM])
    runner.main()
    assert calls == [(auto_root, {"resume": True, "confirm": runner.RESUME_CONFIRM,
                                 "trial_count": 10, "trial_workers": 10})]
    for action, options in [("collect-all", ["--trial-count", "10"]),
                            ("collect-all", ["--trial-workers", "10"]),
                            ("collect-all", ["--trial-count", "10", "--trial-workers", "11"]),
                            ("status", ["--trial-count", "10", "--trial-workers", "10"])]:
        monkeypatch.setattr(sys, "argv", [base[0], action, *base[2:], *options])
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 2
    assert len(calls) == 1

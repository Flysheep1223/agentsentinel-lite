import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentsentinel.backends import BackendError, Completion, OpenAIBackend
from agentsentinel.config import Settings, read_json
from agentsentinel.dataset import build_dataset, load_dataset, template_split
from agentsentinel.environment import Environment, FaultConfig
from agentsentinel.evaluation import choose_threshold, replay, summarize_replay
from agentsentinel.features import FEATURE_COLUMNS, FeatureBuilder
from agentsentinel.harness import run_session
from agentsentinel.modeling import capped_training_rows
from agentsentinel.tasks import make_tasks


def events(directory: Path):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def test_task_catalog_and_state_verifier():
    tasks = make_tasks()
    assert len(tasks) == 40
    assert len({t.template_id for t in tasks}) == 20
    for task in tasks:
        environment = Environment(task, FaultConfig())
        assert not environment.verify("completed")["success"]
        for step, operation in enumerate(task.expected_writes, 1):
            environment.call(operation["tool"], operation["arguments"], step)
        assert environment.verify("completed")["success"]
        assert not environment.verify("step_budget")["success"]


def test_duplicate_write_cannot_be_hidden_by_correct_final_state():
    task = make_tasks()[0]
    environment = Environment(task, FaultConfig())
    operation = task.expected_writes[0]
    for step in (1, 2):
        environment.call(operation["tool"], operation["arguments"], step)
    result = environment.verify("completed")
    assert result["state_matches"]
    assert not result["success"]
    assert result["first_violation_step"] == 2


def test_unrelated_ticket_write_is_a_violation():
    task = make_tasks()[0]
    environment = Environment(task, FaultConfig())
    environment.call("update_ticket", {
        "ticket_id": "T-decoy", "field": "status", "value": "resolved",
    }, 1)
    assert environment.verify("completed")["first_violation_step"] == 1


@pytest.mark.parametrize("kind", ["timeout", "rate_limit", "schema_change", "partial_response"])
def test_faults_reproduce_and_transient_fault_recovers(kind):
    task = make_tasks()[0]
    config = FaultConfig(kind=kind, seed=4, probability=1, persistent_probability=0, transient_calls=1)
    left, right = Environment(task, config), Environment(task, config)
    tool = "lookup_policy" if kind == "partial_response" else "get_service_status"
    args = {"service_id": task.service_id}
    a, b = left.call(tool, args, 1), right.call(tool, args, 1)
    assert a["response"] == b["response"]
    assert a["fault"] == b["fault"]
    assert a["fault"]
    assert left.call(tool, args, 2)["fault"] is None
    if kind == "schema_change":
        assert a["schema_error"]
    if kind == "partial_response":
        assert not a["error"]
        assert "action" not in a["response"]


def test_hard_budget_is_not_an_alarm_sample(tmp_path):
    result = run_session(make_tasks()[0], Settings(max_steps=1), FaultConfig(), tmp_path)
    recorded = events(Path(result["path"]))
    tool = next(e for e in recorded if e["kind"] == "tool_end")
    assert not tool["decision_eligible"]
    assert result["termination"] == "step_budget"
    assert result["label"] == 1


def test_feature_whitelist_prefix_causality_and_missing_reasoning(tmp_path):
    result = run_session(make_tasks()[0], Settings(), FaultConfig(), tmp_path, agent_seed=2)
    recorded = [e for e in events(Path(result["path"])) if e["kind"] == "tool_end"]
    builder = FeatureBuilder("ticket")
    first = builder.add(recorded[0])
    frozen = dict(first)
    for event in recorded[1:]:
        builder.add(event)
    assert first == frozen
    assert first["reasoning_tokens"] is None
    assert set(first) == set(FEATURE_COLUMNS)
    assert not {"fault_kind", "seed", "label", "first_violation_step", "remaining_steps"} & set(first)


def test_online_observer_receives_only_observable_features(tmp_path):
    received = []

    def observer(row):
        received.append(row)
        return {"stop": True, "score": 1, "threshold": 0.5}

    result = run_session(
        make_tasks()[0], Settings(), FaultConfig(), tmp_path,
        phase="online", policy_name="fixture", observer=observer,
    )
    assert result["termination"] == "risk_stop"
    assert len(received) == 1
    assert set(received[0]) == set(FEATURE_COLUMNS)


def test_split_keeps_variants_together_and_is_reproducible():
    tasks = make_tasks()
    template_map = {t.template_id: t.workflow for t in tasks}
    assignments = template_split(template_map)
    assert assignments == template_split(dict(reversed(list(template_map.items()))))
    assert {name: list(assignments.values()).count(name)
            for name in ("train", "validation", "test")} == {
        "train": 12, "validation": 4, "test": 4,
    }


def test_dataset_does_not_train_on_intervention_runs(tmp_path):
    runs = tmp_path / "runs"
    for task in make_tasks()[::2]:
        run_session(task, Settings(), FaultConfig(), runs, agent_seed=2)
    interrupted = run_session(
        make_tasks()[0], Settings(), FaultConfig(), runs, phase="online",
        policy_name="stop", observer=lambda row: {"stop": True},
    )
    summary = build_dataset(runs, tmp_path / "dataset")
    assert summary["sessions"] == 20
    assert summary["excluded"] == [{
        "session_id": interrupted["session_id"], "reason": "intervention",
    }]
    prefixes = pd.read_csv(tmp_path / "dataset/prefixes.csv")
    assert prefixes.groupby("template_id").split.nunique().max() == 1
    assert len(read_json(tmp_path / "dataset/split.json")["session_ids"]) == 20


def replay_fixture():
    sessions = pd.DataFrame([
        {"session_id": "good", "template_id": "A", "label": 0, "total_steps": 5,
         "final_tokens": 100, "final_cost_usd": np.nan, "final_elapsed_ms": 50,
         "first_violation_step": np.nan},
        {"session_id": "bad", "template_id": "B", "label": 1, "total_steps": 5,
         "final_tokens": 100, "final_cost_usd": np.nan, "final_elapsed_ms": 50,
         "first_violation_step": 2},
        {"session_id": "no_prefix", "template_id": "C", "label": 1, "total_steps": 0,
         "final_tokens": 20, "final_cost_usd": np.nan, "final_elapsed_ms": 10,
         "first_violation_step": np.nan},
    ])
    prefixes = pd.DataFrame([
        {"session_id": sid, "step": step, "total_tokens": step * 10,
         "cost_usd": np.nan, "elapsed_ms": step * 5}
        for sid, step in [("good", 1), ("good", 2), ("bad", 1), ("bad", 2)]
    ])
    return prefixes, sessions


def test_first_alarm_and_session_denominators_and_strict_violation_timing():
    prefixes, sessions = replay_fixture()
    rows = replay(prefixes, sessions, [0.1, 0.9, 0.1, 0.9], 0.5)
    summary = summarize_replay(rows)
    assert summary["false_abort_rate"] == 1
    assert summary["failure_recall"] == 0.5
    assert summary["before_violation_rate"] == 0  # Same-step write is already too late.
    assert summary["tokens_used"] == 60
    assert summary["execution_cost_usd"] is None


def test_threshold_budget_applies_to_sessions_and_includes_never_stop():
    prefixes, sessions = replay_fixture()
    selected = choose_threshold(prefixes, sessions, [0.1, 0.9, 0.1, 0.8], 0.05)
    assert selected["threshold"] is None
    assert selected["validation"]["false_abort_count"] == 0


def test_training_cap_retains_short_runs():
    frame = pd.DataFrame([
        {"session_id": sid, "step": step}
        for sid, size in [("short", 2), ("long", 100)] for step in range(1, size + 1)
    ])
    capped = capped_training_rows(frame, 8)
    assert capped.groupby("session_id").size().to_dict() == {"long": 8, "short": 2}
    assert capped[capped.session_id == "long"].step.tolist() == [1, 15, 29, 43, 57, 71, 85, 100]


def test_sse_fragmentation_unicode_usage_and_lossless_body():
    events_json = [
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "call_1",
                            "function": {"name": "get_ticket", "arguments": '{"ticket_id":'}}],
        }}]},
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": '"中文"}'}}],
        }}]},
        {"choices": [], "usage": {
            "prompt_tokens": 10, "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        }},
    ]
    body = "".join(
        "data: " + json.dumps(e, ensure_ascii=False) + "\r\n\r\n" for e in events_json
    ).encode() + b"data: [DONE]\n\n"
    recorded = []
    completion = OpenAIBackend._consume(
        (body[i:i + 7] for i in range(0, len(body), 7)), recorded.append,
    )
    assert json.loads(completion.message["tool_calls"][0]["function"]["arguments"]) == {
        "ticket_id": "中文",
    }
    assert completion.output_tokens == 5
    assert completion.reasoning_tokens == 2
    assert b"".join(base64.b64decode(e["body_base64"]) for e in recorded) == body


def test_truncated_sse_is_infrastructure_failure():
    with pytest.raises(BackendError, match="before"):
        OpenAIBackend._consume([b'data: {"choices": []}\n\n'], lambda event: None)


def test_missing_provider_usage_stops_and_is_not_training_failure(tmp_path, monkeypatch):
    class MissingUsageBackend:
        def __init__(self, settings):
            pass

        def complete(self, *args):
            return Completion({"role": "assistant", "content": "done"}, None, None)

    monkeypatch.setattr("agentsentinel.harness.OpenAIBackend", MissingUsageBackend)
    result = run_session(
        make_tasks()[0], Settings(), FaultConfig(), tmp_path, backend_name="openai",
    )
    assert result["termination"] == "missing_usage"
    assert not result["valid_for_analysis"]
    assert result["totals"]["total_tokens"] is None
    assert result["totals"]["cost_usd"] is None


def test_fault_event_type_does_not_get_overwritten(tmp_path):
    result = run_session(
        make_tasks()[0], Settings(), FaultConfig(
            kind="timeout", probability=1, persistent_probability=1,
        ), tmp_path, agent_seed=2,
    )
    injected = [e for e in events(Path(result["path"])) if e["kind"] == "fault_injected"]
    assert injected
    assert injected[0]["fault"]["kind"] == "timeout"


def test_frozen_dataset_rejects_changes(tmp_path):
    for task in make_tasks()[::2]:
        run_session(task, Settings(), FaultConfig(), tmp_path / "runs", agent_seed=2)
    dataset = tmp_path / "dataset"
    build_dataset(tmp_path / "runs", dataset)
    load_dataset(dataset)
    with (dataset / "prefixes.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="冻结后改变"):
        load_dataset(dataset)


def test_threshold_uses_earliest_alert_for_cost_objective():
    prefixes, sessions = replay_fixture()
    selected = choose_threshold(prefixes, sessions, [0.1, 0.1, 0.6, 0.9], 0.05)
    assert selected["threshold"] == 0.6
    assert selected["validation"]["tokens_used"] == 130
    assert selected["validation"]["before_violation_rate"] == 1

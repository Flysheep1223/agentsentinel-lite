import base64
import copy
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from agentsentinel import cli
from agentsentinel.backends import BackendError, Completion, OpenAIBackend, ScriptedBackend
from agentsentinel.config import Settings, fingerprint, read_json, write_json
from agentsentinel.dataset import (
    build_dataset,
    freeze_design,
    load_dataset,
    load_design,
    template_split,
    validate_design,
)
from agentsentinel.environment import FAULT_KINDS, Environment, FaultConfig
from agentsentinel.evaluation import choose_threshold, replay, summarize_replay
from agentsentinel.experiments import collect
from agentsentinel.features import FEATURE_COLUMNS, FeatureBuilder
from agentsentinel.harness import run_session
from agentsentinel.modeling import capped_training_rows, train_models
from agentsentinel.tasks import make_tasks


def events(directory: Path):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


@pytest.fixture
def normal_backend(monkeypatch):
    def factory(seed):
        backend = ScriptedBackend(seed)
        backend.behavior = "normal"
        return backend

    monkeypatch.setattr("agentsentinel.harness.ScriptedBackend", factory)


@pytest.fixture
def frozen_runs(tmp_path):
    runs = tmp_path / "runs"
    return runs, freeze_design(runs / "design.json", seed=42)


def test_task_catalog_and_state_verifier():
    tasks = make_tasks()
    assert len(tasks) == 40
    assert len({t.template_id for t in tasks}) == 10
    assert set(Counter(t.template_id for t in tasks).values()) == {4}
    assert {t.structure for t in tasks if t.workflow == "ticket"} == {
        "single", "linked", "compare", "branch", "chain",
    }
    assert {t.structure for t in tasks if t.workflow == "incident"} == {
        "single", "linked", "compare", "branch", "aggregate",
    }
    for task in tasks:
        environment = Environment(task, FaultConfig())
        assert not environment.verify("completed")["success"]
        for step, operation in enumerate(task.expected_writes, 1):
            environment.call(operation["tool"], operation["arguments"], step)
        for termination in ("completed", "step_budget", "model_call_budget", "risk_stop"):
            result = environment.verify(termination)
            assert result["success"] == result["completed_conditions"] is True
            assert result["label"] == 0
            assert result["termination"] == termination


@pytest.mark.parametrize("task", make_tasks(), ids=lambda task: task.task_id)
def test_normal_scripted_tasks_complete_within_twelve_calls(task, tmp_path, normal_backend):
    result = run_session(task, Settings(max_steps=12), FaultConfig(), tmp_path)
    assert result["valid_for_analysis"]
    assert result["success"]
    assert result["label"] == 0
    assert 0 < result["totals"]["tool_calls"] <= 12


def test_public_requests_do_not_reveal_hidden_state_or_expected_writes():
    forbidden = {"initial_state", "expected_writes", "state", "knowledge", "impact_score"}
    for task in make_tasks():
        public = json.loads(task.public_request())
        assert not forbidden & public.keys()
        hidden_changed = task.model_copy(deep=True)
        hidden_changed.initial_state = {"private_marker": "hidden-state-canary"}
        hidden_changed.expected_writes = [{"private_marker": "hidden-answer-canary"}]
        assert hidden_changed.public_request() == task.public_request()
        assert public["instructions"]
        assert public["structure"] == task.structure


@pytest.mark.parametrize("workflow,structure,variant", [
    ("ticket", "linked", 1), ("incident", "linked", 1),
    ("ticket", "compare", 1), ("ticket", "compare", 2),
    ("incident", "compare", 3),
    ("ticket", "branch", 0), ("ticket", "branch", 1),
    ("incident", "branch", 0), ("incident", "branch", 1),
    ("ticket", "chain", 1), ("incident", "aggregate", 1),
])
def test_structures_read_relationships_before_writes(
    workflow, structure, variant, tmp_path, normal_backend,
):
    task = [t for t in make_tasks() if t.workflow == workflow and t.structure == structure][variant]
    result = run_session(task, Settings(max_steps=12), FaultConfig(), tmp_path)
    calls = [e for e in events(Path(result["path"])) if e["kind"] == "tool_end"]
    first_write = next(i for i, event in enumerate(calls) if event["is_write"])
    reads = calls[:first_write]
    tickets = {e["arguments"]["ticket_id"] for e in reads if e["tool"] == "get_ticket"}
    services = {e["arguments"]["service_id"] for e in reads
                if e["tool"] == "get_service_status"}
    policies = {e["arguments"]["service_id"] for e in reads if e["tool"] == "lookup_policy"}
    writes = [e for e in calls if e["is_write"]]
    root_ticket = task.initial_state["tickets"][task.ticket_id]
    root_service = task.initial_state["services"][task.service_id]
    assert result["success"]
    if structure == "linked":
        target = (root_ticket["related_ticket_id"] if workflow == "ticket"
                  else root_service["response_ticket_id"])
        assert target in tickets
        assert task.initial_state["tickets"][target]["service_id"] in services & policies
        assert all(e["arguments"].get("ticket_id", target) == target for e in writes)
    elif structure == "compare":
        if workflow == "ticket":
            assert set(root_ticket["candidate_ticket_ids"]) <= tickets
            # 两个抽样实例分别验证分数胜者切换与平局时按标识排序。
            target = root_ticket["candidate_ticket_ids"][1 if variant == 1 else 0]
            assert {e["arguments"]["ticket_id"] for e in writes} == {target}
        else:
            assert set(root_service["related_service_ids"]) <= services
            target = root_service["related_service_ids"][1]
            assert {e["arguments"]["service_id"] for e in writes} == {target}
    elif structure == "branch":
        if workflow == "ticket":
            target = root_ticket["escalation_ticket_id"]
            assert (target in tickets) == (variant == 1)
            expected_target = task.ticket_id if variant == 0 else target
            assert {e["arguments"]["ticket_id"] for e in writes} == {expected_target}
        else:
            target = root_service["dependency_service_id"]
            assert (target in services) == (variant == 1)
            assert {e["tool"] for e in writes} == (
                {"send_notification"} if variant == 0
                else {"record_incident", "send_notification"}
            )
    elif structure == "chain":
        parent = root_ticket["parent_ticket_id"]
        terminal = task.initial_state["tickets"][parent]["parent_ticket_id"]
        assert tickets == {task.ticket_id, parent, terminal}
        assert {e["arguments"]["ticket_id"] for e in writes} == {terminal}
    elif structure == "aggregate":
        expected_services = {task.service_id, *root_service["dependency_service_ids"]}
        assert services == policies == expected_services
        action = next(e["arguments"]["action"] for e in writes if e["tool"] == "record_incident")
        assert action.startswith("aggregate:")
        entries = action.removeprefix("aggregate:").split(";")
        assert [entry.split("=", 1)[0] for entry in entries] == sorted(expected_services)
        assert {e["arguments"]["service_id"] for e in writes} == {task.service_id}


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


def test_last_allowed_call_can_succeed_without_being_an_alarm_sample(tmp_path, normal_backend):
    observed = []

    def observer(row):
        observed.append(row)
        return {"stop": False}

    task = make_tasks()[0]
    baseline = run_session(task, Settings(), FaultConfig(), tmp_path)
    required_calls = baseline["totals"]["tool_calls"]
    result = run_session(
        task, Settings(max_steps=required_calls), FaultConfig(), tmp_path,
        policy_name="observe", observer=observer,
    )
    tools = [e for e in events(Path(result["path"])) if e["kind"] == "tool_end"]
    assert result["termination"] == "step_budget"
    assert result["success"] and result["completed_conditions"]
    assert result["label"] == 0
    assert tools[-1]["is_write"] and not tools[-1]["decision_eligible"]
    assert len(observed) == required_calls - 1
    assert all(row["step"] < required_calls for row in observed)


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
        "train": 6, "validation": 2, "test": 2,
    }


def test_design_is_immutable_and_detects_tampering(frozen_runs):
    runs, design = frozen_runs
    path = runs / "design.json"
    original = path.read_bytes()
    assert load_design(path) == validate_design(design) == design
    with pytest.raises(ValueError, match="不允许覆盖"):
        freeze_design(path, seed=7)
    assert path.read_bytes() == original
    changed = copy.deepcopy(design)
    changed["seed"] += 1
    write_json(path, changed)
    with pytest.raises(ValueError, match="fingerprint"):
        load_design(path)


def test_design_pairs_workflows_and_limits_pilot_and_online_tasks(frozen_runs):
    _, design = frozen_runs
    templates = {t["template_id"]: t["workflow"] for t in design["tasks"]}
    held_out = [name for fold in design["cv_folds"] for name in fold]
    assert Counter(held_out) == Counter(
        name for name, split in design["assignments"].items() if split == "train"
    )
    for fold in design["cv_folds"]:
        assert len(fold) == 2
        assert {templates[name] for name in fold} == {"ticket", "incident"}
    tasks = {t["task_id"]: t for t in design["tasks"]}
    for key, split, count in (("pilot_task_ids", "train", 6), ("online_task_ids", "test", 4)):
        ids = design[key]
        assert len(ids) == len(set(ids)) == count
        assert all(design["assignments"][tasks[task_id]["template_id"]] == split for task_id in ids)
    changed = copy.deepcopy(design)
    changed["cv_folds"][0][0], changed["cv_folds"][1][1] = (
        changed["cv_folds"][1][1], changed["cv_folds"][0][0],
    )
    changed["fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="两个工作流"):
        validate_design(changed)


@pytest.fixture
def collection_protocol(frozen_runs, monkeypatch):
    _, design = frozen_runs
    settings = Settings(model="offline-test-model", max_steps=12)
    policies = {
        "design_hash": design["fingerprint"],
        "backend": "openai",
        "config_hash": fingerprint(settings.model_dump()),
        "template_assignments": design["assignments"],
        "fault_conditions": list(FAULT_KINDS),
        "fault_design": FaultConfig().model_dump(exclude={"kind", "seed"}),
        "freeze_hash": "offline-policy-fixture",
    }
    monkeypatch.setattr("agentsentinel.experiments.read_policies", Mock(return_value=policies))
    observer = Mock(return_value=lambda row: {"stop": False})
    monkeypatch.setattr("agentsentinel.experiments.load_observer", observer)
    # 即使会话桩意外失效，也不允许测试创建真实 API 后端。
    monkeypatch.setattr("agentsentinel.harness.OpenAIBackend", Mock(
        side_effect=AssertionError("测试禁止真实 API"),
    ))
    return settings, observer


@pytest.mark.parametrize("phase,repeats,total", [
    ("pilot", 1, 30), ("collection", 3, 600), ("online", 2, 160),
])
def test_collect_frozen_protocol_and_manifest_before_scheduling(
    phase, repeats, total, tmp_path, frozen_runs, collection_protocol, monkeypatch,
):
    _, design = frozen_runs
    settings, observer_loader = collection_protocol
    output = tmp_path / "collected"
    tasks = make_tasks()
    if phase != "collection":
        tasks = [task for task in tasks if task.task_id in design[f"{phase}_task_ids"]]
    methods = ["none", "rules", "logistic", "knn"] if phase == "online" else ["none"]
    models = tmp_path / "models" if phase == "online" else None
    backend = Mock()
    monkeypatch.setattr("agentsentinel.experiments.OpenAIBackend", backend)
    calls = []

    def executor_factory(**kwargs):
        # 在线程池创建前检查落盘，不依赖会话运行结果补写设计。
        assert load_design(output / "design.json") == design
        manifests = list(output.glob("experiment_*.json"))
        assert len(manifests) == 1
        manifest = read_json(manifests[0])
        assert manifest["design_hash"] == design["fingerprint"]
        assert manifest["planned_sessions"] == total
        assert manifest["status"] == "running"
        assert not calls
        return ThreadPoolExecutor(**kwargs)

    def session_stub(
        task, settings, faults, output, backend_name, agent_seed, phase,
        policy_name, observer, experiment_id, condition_id, design_hash,
    ):
        assert design_hash == design["fingerprint"]
        assert backend_name == "openai"
        assert (observer is None) == (policy_name == "none")
        calls.append((task.task_id, faults.kind, policy_name, condition_id, agent_seed, faults.seed))
        session_id = f"{condition_id}-{policy_name}"
        # 仅返回采集汇总所需字段，不执行任何真实会话或网络请求。
        return {
            "session_id": session_id, "condition_id": condition_id,
            "policy": policy_name, "path": str(output / session_id),
            "valid_for_analysis": True, "success": True, "termination": "completed",
            "totals": {"cost_usd": 0.01, "total_tokens": 10, "model_calls": 1},
            "monitor_cost_usd": 0, "elapsed_ms": 1,
            "feature_extraction_ms": 0, "model_scoring_ms": 0,
        }

    monkeypatch.setattr("agentsentinel.experiments.ThreadPoolExecutor", executor_factory)
    monkeypatch.setattr("agentsentinel.experiments.run_session", session_stub)
    result = collect(
        tasks, settings, output, "openai", phase, FAULT_KINDS, repeats, 2, 42,
        models=models, design=design,
    )
    backend.assert_called_once_with(settings)
    assert result["status"] == "complete"
    assert result["planned_sessions"] == result["completed_sessions"] == len(calls) == total
    assert result["invalid_sessions"] == 0
    assert result["design_hash"] == design["fingerprint"]
    assert read_json(output / f"experiment_{result['experiment_id']}.json") == result
    assert Counter((task_id, condition, method) for task_id, condition, method, *_ in calls) == {
        (task.task_id, condition, method): repeats
        for task in tasks for condition in FAULT_KINDS for method in methods
    }
    assert len({session["session_id"] for session in result["sessions"]}) == total
    for method in methods:
        summary = result["summaries"][method]
        assert summary["sessions"] == summary["valid_sessions"] == total // len(methods)
        assert summary["total_tokens"] == summary["sessions"] * 10
        assert summary["cost_usd"] == pytest.approx(summary["sessions"] * 0.01)
    if phase == "pilot":
        assert len(tasks) == len({task.template_id for task in tasks}) == 6
        assert {design["assignments"][task.template_id] for task in tasks} == {"train"}
    if phase == "online":
        assert Counter(method for _, _, method, *_ in calls) == dict.fromkeys(methods, 40)
        assert {design["assignments"][task.template_id] for task in tasks} == {"test"}
        assert len({(condition_id, agent_seed, fault_seed)
                    for _, _, _, condition_id, agent_seed, fault_seed in calls}) == 40
        assert observer_loader.call_count == 120
        assert Counter(call.args for call in observer_loader.call_args_list) == {
            (models, method, 0.05): 40 for method in methods if method != "none"
        }
    else:
        observer_loader.assert_not_called()


@pytest.mark.parametrize("phase,case,match", [
    ("pilot", "selection", "预选的 6 个"),
    ("pilot", "repeats", "预选的 6 个"),
    ("pilot", "conditions", "完整五种条件"),
    ("collection", "selection", "全部 40 个"),
    ("collection", "repeats", "全部 40 个"),
    ("collection", "conditions", "完整五种条件"),
    ("online", "selection", "预选的 4 个"),
    ("online", "repeats", "预选的 4 个"),
    ("online", "conditions", "在线故障条件"),
    ("collection", "missing_design", "显式冻结"),
    ("collection", "tampered_design", "fingerprint"),
    ("collection", "different_design", "design.json 与传入设计不一致"),
    ("collection", "nonempty", "禁止混批或覆盖"),
])
def test_collect_rejects_invalid_protocol_before_backend_or_scheduling(
    phase, case, match, tmp_path, frozen_runs, collection_protocol, monkeypatch,
):
    runs, design = frozen_runs
    settings, observer_loader = collection_protocol
    tasks = make_tasks()
    if phase != "collection":
        tasks = [task for task in tasks if task.task_id in design[f"{phase}_task_ids"]]
    repeats = {"pilot": 1, "collection": 3, "online": 2}[phase]
    conditions = list(FAULT_KINDS)
    if case == "selection":
        tasks = tasks[:-1]
    elif case == "repeats":
        repeats += 1
    elif case == "conditions":
        conditions = conditions[:-1]
    elif case == "missing_design":
        design = None
    elif case == "tampered_design":
        design = copy.deepcopy(design)
        design["seed"] += 1
    elif case == "different_design":
        design = freeze_design(tmp_path / "other-design.json", seed=7)
    elif case == "nonempty":
        write_json(runs / "existing-session.json", {"preserve": True})
    original = {path.name: path.read_bytes() for path in runs.iterdir()}
    backend = Mock(side_effect=AssertionError("拒绝前不得初始化后端"))
    executor = Mock(side_effect=AssertionError("拒绝前不得调度"))
    session = Mock(side_effect=AssertionError("拒绝前不得执行会话"))
    monkeypatch.setattr("agentsentinel.experiments.OpenAIBackend", backend)
    monkeypatch.setattr("agentsentinel.experiments.ThreadPoolExecutor", executor)
    monkeypatch.setattr("agentsentinel.experiments.run_session", session)
    with pytest.raises(ValueError, match=match):
        collect(
            tasks, settings, runs, "openai", phase, conditions, repeats, 2, 42,
            models=tmp_path / "models" if phase == "online" else None, design=design,
        )
    backend.assert_not_called()
    executor.assert_not_called()
    session.assert_not_called()
    observer_loader.assert_not_called()
    assert {path.name: path.read_bytes() for path in runs.iterdir()} == original


def test_cli_freeze_and_pilot_defaults(tmp_path, monkeypatch, capsys):
    design_path = tmp_path / "design.json"
    output = tmp_path / "pilot"
    settings = Settings()
    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    monkeypatch.setattr("sys.argv", ["sentinel", "freeze", "--output", str(design_path)])
    assert cli.main() == 0
    design = load_design(design_path)
    assert design["fingerprint"] in capsys.readouterr().out
    collector = Mock(return_value={"invalid_sessions": 0})
    monkeypatch.setattr(cli, "collect", collector)
    monkeypatch.setattr("sys.argv", [
        "sentinel", "collect", "--design", str(design_path),
        "--output", str(output), "--phase", "pilot",
    ])
    assert cli.main() == 0
    collector.assert_called_once()
    args = collector.call_args.args
    assert [task.task_id for task in args[0]] == design["pilot_task_ids"]
    assert args[1:6] == (settings, output, "scripted", "pilot", FAULT_KINDS)
    assert args[6] == 1
    assert collector.call_args.kwargs == {"design": design}
    with pytest.raises(SystemExit) as exc:
        cli.parser().parse_args(["collect", "--output", str(output)])
    assert exc.value.code == 2
    assert "--design" in capsys.readouterr().err


def test_dataset_requires_frozen_design(tmp_path):
    with pytest.raises(ValueError, match="冻结设计缺失"):
        build_dataset(tmp_path / "runs", tmp_path / "dataset")


def test_dataset_does_not_train_on_intervention_runs(tmp_path, frozen_runs):
    runs, design = frozen_runs
    for task in make_tasks()[::2]:
        run_session(task, Settings(), FaultConfig(), runs, agent_seed=2,
                    design_hash=design["fingerprint"])
    interrupted = run_session(
        make_tasks()[0], Settings(), FaultConfig(), runs, phase="online",
        policy_name="stop", observer=lambda row: {"stop": True},
        design_hash=design["fingerprint"],
    )
    summary = build_dataset(runs, tmp_path / "dataset")
    assert summary["sessions"] == 20
    assert summary["excluded"] == [{
        "session_id": interrupted["session_id"], "reason": "intervention",
    }]
    prefixes = pd.read_csv(tmp_path / "dataset/prefixes.csv")
    assert prefixes.groupby("template_id").split.nunique().max() == 1
    manifest = read_json(tmp_path / "dataset/split.json")
    assert len(manifest["session_ids"]) == 20
    assert manifest["design_hash"] == design["fingerprint"]


def test_missing_templates_do_not_change_frozen_assignments(tmp_path, frozen_runs, normal_backend):
    runs, design = frozen_runs
    task = make_tasks()[0]
    run_session(task, Settings(), FaultConfig(), runs, design_hash=design["fingerprint"])
    summary = build_dataset(runs, tmp_path / "dataset", seed=7)
    _, sessions, manifest = load_dataset(tmp_path / "dataset")
    assert manifest["assignments"] == design["assignments"]
    assert manifest["seed"] == design["seed"]
    assert sessions.split.tolist() == [design["assignments"][task.template_id]]
    assert summary["missing_templates"] == sorted(set(design["assignments"]) - {task.template_id})


@pytest.mark.parametrize("corruption", ["design_hash", "task", "template_id"])
def test_dataset_rejects_session_design_or_task_mismatch(
    corruption, tmp_path, frozen_runs, normal_backend,
):
    runs, design = frozen_runs
    result = run_session(make_tasks()[0], Settings(), FaultConfig(), runs,
                         design_hash=design["fingerprint"])
    path = Path(result["path"]) / "session.json"
    metadata = read_json(path)
    assert metadata["design_hash"] == design["fingerprint"]
    assert metadata["task"] == make_tasks()[0].model_dump()
    if corruption == "task":
        metadata["task"]["expected_writes"] = []
    else:
        metadata[corruption] = "wrong-value"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="Session"):
        build_dataset(runs, tmp_path / "dataset")


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_pilot_accepts_only_training_templates(split, tmp_path, frozen_runs, normal_backend):
    runs, design = frozen_runs
    task = next(t for t in make_tasks() if design["assignments"][t.template_id] == split)
    run_session(task, Settings(), FaultConfig(), runs, phase="pilot",
                design_hash=design["fingerprint"])
    if split != "train":
        with pytest.raises(ValueError, match="pilot 仅允许训练模板"):
            build_dataset(runs, tmp_path / "dataset")
    else:
        build_dataset(runs, tmp_path / "dataset")
        with pytest.raises(ValueError, match="试运行数据不能进入正式训练"):
            train_models(tmp_path / "dataset", tmp_path / "models")


@pytest.mark.parametrize("case", ["fixed", "missing_template", "single_class_fold"])
def test_training_respects_fixed_folds_without_expensive_fitting(
    case, tmp_path, frozen_runs, monkeypatch,
):
    _, design = frozen_runs
    rows = []
    for template, split in design["assignments"].items():
        for label in (0, 1):
            rows.append({
                "session_id": f"{template}-{label}", "template_id": template,
                "split": split, "label": label, "step": 1,
            })
    prefixes = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    if case == "missing_template":
        prefixes = prefixes[prefixes.template_id != design["cv_folds"][0][0]]
    elif case == "single_class_fold":
        prefixes.loc[prefixes.template_id.isin(design["cv_folds"][0]), "label"] = 0
    manifest = {
        "phase": "demo", "design": design, "fingerprint": "fixture", "backend": "scripted",
        "config_hash": "fixture", "monitor_price_per_second": 0,
        "assignments": design["assignments"], "fault_design": {}, "fault_conditions": ["none"],
    }
    monkeypatch.setattr("agentsentinel.modeling.load_dataset",
                        lambda path: (prefixes, prefixes.copy(), manifest))

    class FoldsInspected(Exception):
        pass

    def inspect_search(*args, **kwargs):
        training = capped_training_rows(prefixes[prefixes.split == "train"], 12)
        folds = kwargs["cv"]
        assert len(folds) == 3
        for (train_ids, validation_ids), expected in zip(folds, design["cv_folds"]):
            assert set(training.iloc[validation_ids].template_id) == set(expected)
            assert set(training.iloc[train_ids].template_id).isdisjoint(expected)
            assert set(train_ids) | set(validation_ids) == set(range(len(training)))
        # 检查真实训练入口传入的冻结折后即停止，不执行网格搜索。
        raise FoldsInspected

    monkeypatch.setattr("agentsentinel.modeling.GridSearchCV", inspect_search)
    if case == "fixed":
        with pytest.raises(FoldsInspected):
            train_models(tmp_path / "dataset", tmp_path / "models")
    else:
        match = "缺少可用前缀" if case == "missing_template" else "单一类别；不能重新分折"
        with pytest.raises(ValueError, match=match):
            train_models(tmp_path / "dataset", tmp_path / "models")


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
    assert summary["before_violation_rate"] == 0  # 同一步写入发生后再告警已经太晚。
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


def test_frozen_dataset_rejects_changes(tmp_path, frozen_runs):
    runs, design = frozen_runs
    for task in make_tasks()[::2]:
        run_session(task, Settings(), FaultConfig(), runs, agent_seed=2,
                    design_hash=design["fingerprint"])
    dataset = tmp_path / "dataset"
    build_dataset(runs, dataset)
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

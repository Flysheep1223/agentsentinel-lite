from __future__ import annotations

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .backends import OpenAIBackend
from .config import Settings, fingerprint, write_json
from .environment import FaultConfig
from .harness import run_session
from .modeling import load_observer, read_policies
from .tasks import Task


def collect(
    tasks: list[Task], settings: Settings, output: Path,
    backend: str, phase: str, conditions: list[str], repeats: int, workers: int,
    seed: int, fault_probability: float = 0.7, persistent_probability: float = 0.3,
    models: Path | None = None, budget: float = 0.05,
) -> dict:
    if repeats < 1 or workers < 1 or not tasks:
        raise ValueError("任务数、重复数和 worker 数必须大于零。")
    if backend == "scripted" and phase == "collection":
        raise ValueError("正式 collection 只接受真实模型；脚本运行请用 demo 阶段。")
    if backend == "openai":
        OpenAIBackend(settings)  # Validate configuration before scheduling any sessions.
    fault_design = FaultConfig(
        probability=fault_probability, persistent_probability=persistent_probability,
    ).model_dump(exclude={"kind", "seed"})
    if models is not None:
        policy_meta = read_policies(models)
        if policy_meta["backend"] != backend:
            raise ValueError("在线后端必须与训练数据一致。")
        if policy_meta["config_hash"] != fingerprint(settings.model_dump()):
            raise ValueError("在线 Agent 配置必须与冻结配置一致。")
        if any(policy_meta["template_assignments"].get(t.template_id) != "test" for t in tasks):
            raise ValueError("在线对照仅接受冻结清单中的测试模板。")
        fault_design = policy_meta["fault_design"]
        conditions = policy_meta["fault_conditions"]
        phase = "online"
    elif phase == "online":
        raise ValueError("在线对照需要冻结模型目录。")
    experiment_id = uuid.uuid4().hex
    methods = ["none", "rules", "logistic", "knn"] if models else ["none"]
    specifications = []
    for task in tasks:
        for condition in conditions:
            for repeat in range(repeats):
                condition_id = fingerprint([task.task_id, condition, repeat, seed])[:20]
                fault_seed = int(fingerprint([condition_id, "fault"])[:8], 16)
                agent_seed = int(fingerprint([task.task_id, repeat, seed, "agent"])[:8], 16)
                for method in methods:
                    specifications.append((task, condition, condition_id, fault_seed, agent_seed, method))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"experiment_{experiment_id}.json"
    experiment = {
        "experiment_id": experiment_id, "backend": backend, "phase": phase,
        "task_ids": [t.task_id for t in tasks], "conditions": conditions,
        "repeats": repeats, "workers": workers, "seed": seed,
        "fault_design": fault_design,
        "settings": settings.model_dump(), "planned_sessions": len(specifications),
        "methods": methods, "status": "running",
        "freeze_hash": policy_meta["freeze_hash"] if models else None,
    }
    write_json(manifest_path, experiment)
    print(f"{phase}/{backend}: {len(specifications)} sessions, {workers} workers")

    def execute(specification):
        task, condition, condition_id, fault_seed, agent_seed, method = specification
        return run_session(
            task, settings, FaultConfig(
                kind=condition, seed=fault_seed, **fault_design,
            ), output, backend, agent_seed, phase, method,
            load_observer(models, method, budget) if models and method != "none" else None,
            experiment_id, condition_id,
        )

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, spec) for spec in specifications]
        for future in as_completed(futures):
            results.append(future.result())
            if sys.stdout.isatty():
                print(f"\r\033[2KCompleted {len(results)}/{len(specifications)}", end="", flush=True)
    if sys.stdout.isatty():
        print()
    invalid = [r for r in results if not r["valid_for_analysis"]]
    summaries = {}
    for method in methods:
        group = [r for r in results if r["policy"] == method]
        valid = [r for r in group if r["valid_for_analysis"]]
        successes = sum(r["success"] for r in valid)
        priced = all(r["totals"]["cost_usd"] is not None for r in valid) and bool(valid)
        cost = sum(r["totals"]["cost_usd"] + r["monitor_cost_usd"] for r in valid) if priced else None
        tokens = sum(r["totals"]["total_tokens"] for r in valid)
        summaries[method] = {
            "sessions": len(group), "valid_sessions": len(valid),
            "successes": successes,
            "completion_rate": successes / len(valid) if valid else None,
            "stop_rate": sum(r["termination"] == "risk_stop" for r in valid) / len(valid)
            if valid else None,
            "total_tokens": tokens, "cost_usd": cost,
            "cost_per_success_usd": cost / successes if cost is not None and successes else None,
            "tokens_per_success": tokens / successes if successes else None,
            "model_calls": sum(r["totals"]["model_calls"] for r in valid),
            "elapsed_ms": sum(r["elapsed_ms"] for r in valid),
            "feature_extraction_ms": sum(r["feature_extraction_ms"] for r in valid),
            "model_scoring_ms": sum(r["model_scoring_ms"] for r in valid),
            "projected_600_session_tokens": tokens / len(valid) * 600 if valid else None,
            "projected_600_session_api_usd": cost / len(valid) * 600
            if cost is not None and valid else None,
        }
    experiment.update({
        "status": "complete" if not invalid else "completed_with_backend_errors",
        "completed_sessions": len(results), "invalid_sessions": len(invalid),
        "summaries": summaries,
        "sessions": [
            {key: r[key] for key in ("session_id", "condition_id", "policy", "path")}
            for r in results
        ],
    })
    write_json(manifest_path, experiment)
    print(f"Completed {len(results)} sessions; invalid={len(invalid)}. Manifest: {manifest_path}")
    return experiment

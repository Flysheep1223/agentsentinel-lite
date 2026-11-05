"""Resumable, at-most-once pilot and collection for the authorized 30 + 600 protocol."""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import ROOT, Settings, fingerprint, read_json, write_json
from .dataset import freeze_design, load_design
from .environment import FAULT_KINDS, Environment, FaultConfig
from .harness import SYSTEM_PROMPT, run_session
from .tasks import Task


def atomic_json(path, value):
    temporary = path.with_suffix(".pending")
    write_json(temporary, value)
    temporary.replace(path)


def code_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/agentsentinel").glob("*.py"))}


def initialize(root, output_limit=1024):
    if root.exists():
        raise ValueError("Experiment root already exists; do not overwrite it")
    settings = Settings.from_env().model_copy(update={
        "max_steps": 12, "max_tokens": 30000, "session_timeout": 120,
        "max_model_calls": 24, "max_completion_tokens": output_limit,
        "currency": "CNY", "input_price": 13, "cached_input_price": 1.3, "output_price": 68,
    })
    if not settings.model or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing model or API key")
    if "@" in settings.base_url or "?" in settings.base_url:
        raise ValueError("Endpoint must not contain credentials or query parameters")
    design = freeze_design(root / "design.json", seed=42)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings.model_dump(), "design_hash": design["fingerprint"],
        "fault_design": FaultConfig().model_dump(exclude={"kind", "seed"}),
        "seed": 42, "conditions": FAULT_KINDS, "automatic_api_retries": 0,
        "monetary_cap_cny": None, "authorization": "User explicitly authorized no CNY ceiling",
        "online_sessions_authorized": 0, "system_prompt": SYSTEM_PROMPT,
        "reasoning_configuration": "Provider default; no thinking/reasoning override sent",
        "timeout_simulation": "Immediate synthetic error; no artificial sleep",
        "token_budget_boundary": "Reported cumulative input+output; a request can overshoot. "
        "No request or tool write starts after reported exhaustion. Reasoning is included in output.",
        "pause_policy": "Pause after 3 infrastructure-invalid sessions in the latest 10, "
        "or immediately after authentication failure. Deadline usage missing is reported separately.",
        "workers": {"pilot": 2, "collection": 4}, "code_hashes": code_hashes(),
    }
    protocol["fingerprint"] = fingerprint(protocol)
    write_json(root / "protocol.json", protocol)
    for phase in ("pilot", "collection"):
        tasks = design["tasks"] if phase == "collection" else [
            task for task in design["tasks"] if task["task_id"] in design["pilot_task_ids"]
        ]
        plans = []
        for task in tasks:
            for condition in FAULT_KINDS:
                for repeat in range(3 if phase == "collection" else 1):
                    condition_id = fingerprint([task["task_id"], condition, repeat, 42])[:20]
                    plans.append({
                        "task_id": task["task_id"], "condition": condition, "repeat": repeat,
                        "condition_id": condition_id,
                        "fault_seed": int(fingerprint([condition_id, "fault"])[:8], 16),
                        "agent_seed": int(fingerprint([
                            task["task_id"], repeat, 42, "agent",
                        ])[:8], 16),
                    })
        write_json(root / phase / "design.json", design)
        write_json(root / phase / "plan.json", plans)
        write_json(root / phase / "ledger.json", {
            "phase": phase, "status": "not_started", "entries": {
                spec["condition_id"]: {"state": "pending"} for spec in plans
            },
        })
    print(json.dumps({"root": str(root), "model": settings.model,
                      "planned_pilot": 30, "planned_collection": 600,
                      "protocol_hash": protocol["fingerprint"]}, ensure_ascii=False))


def read_protocol(root):
    protocol = read_json(root / "protocol.json")
    if fingerprint({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol["fingerprint"]:
        raise ValueError("Protocol fingerprint mismatch")
    if protocol["code_hashes"] != code_hashes():
        raise ValueError("Source changed since protocol snapshot; use an explicitly separate run")
    design = load_design(root / "design.json")
    if design["fingerprint"] != protocol["design_hash"]:
        raise ValueError("Design mismatch")
    return protocol, design


def events_at(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def audit_session(directory, protocol, design, spec):
    meta, result = read_json(directory / "session.json"), read_json(directory / "result.json")
    trajectory = read_json(directory / "trajectory.json")
    events = events_at(directory)
    assert meta["condition_id"] == spec["condition_id"]
    assert meta["task_id"] == spec["task_id"]
    assert meta["settings"] == protocol["settings"]
    assert meta["design_hash"] == design["fingerprint"]
    assert events[-1]["kind"] == "session_end" and events[-1]["result"] == result
    assert len([e for e in events if e["kind"] == "llm_start"]) == result["totals"]["model_calls"]
    task = Task.model_validate(meta["task"])
    assert task.model_dump() == next(t for t in design["tasks"] if t["task_id"] == task.task_id)
    environment = Environment(task, FaultConfig.model_validate(meta["fault_config"]))
    for event in events:
        if event["kind"] == "tool_end":
            replayed = environment.call(event["tool"], event["arguments"], event["step"])
            assert replayed["response"] == event["response"]
            timeout = protocol["settings"]["session_timeout"]
            if timeout is not None:
                assert event["elapsed_ms"] < timeout * 1000 + 100
    verified = environment.verify(result["termination"])
    assert all(result[key] == value for key, value in verified.items())
    assert trajectory["writes"] == environment.writes
    assert trajectory["final_state"] == environment.state
    secret = os.getenv("OPENAI_API_KEY", "").encode()
    if secret:
        for path in directory.iterdir():
            if path.is_file():
                assert secret not in path.read_bytes(), "Credential found in artifact"
        for event in events:
            if event["kind"] == "llm_sse":
                assert secret not in base64.b64decode(event["body_base64"])
    return meta, result, events


def summarize(root, phase):
    protocol = read_json(root / "protocol.json")
    design = load_design(root / "design.json")
    directory = root / phase
    ledger = read_json(directory / "ledger.json")
    plans = read_json(directory / "plan.json")
    rows, usage_rows = [], []
    integrity_errors = []
    for spec in plans:
        entry = ledger["entries"][spec["condition_id"]]
        if entry["state"] != "completed":
            continue
        session_dir = Path(entry["path"])
        try:
            meta, result, events = audit_session(session_dir, protocol, design, spec)
        except (AssertionError, ValueError, KeyError, OSError) as exc:
            integrity_errors.append({"condition_id": spec["condition_id"],
                                     "error_type": type(exc).__name__})
            continue
        faults = [e for e in events if e["kind"] == "fault_injected"]
        llm_ends = [e for e in events if e["kind"] == "llm_end"]
        row = {
            **spec, "session_id": result["session_id"], "path": str(session_dir),
            "template_id": meta["template_id"], "workflow": meta["workflow"],
            "split": design["assignments"][meta["template_id"]],
            **{key: result[key] for key in (
                "success", "label", "valid_for_analysis", "termination", "elapsed_ms",
            )},
            **result["totals"], "injected_faults": len(faults),
            "fault_triggered": bool(faults), "completed_responses": len(llm_ends),
        }
        rows.append(row)
        for event in llm_ends:
            usage_rows.append(event["usage"])

    def group_summary(group):
        valid = [row for row in group if row["valid_for_analysis"]]
        return {
            "sessions": len(group), "valid": len(valid), "invalid": len(group) - len(valid),
            "success": sum(row["success"] for row in valid),
            "failure": sum(not row["success"] for row in valid),
            "terminations": dict(Counter(row["termination"] for row in group)),
            "requests": sum(row["model_calls"] for row in group),
            "fault_triggered_sessions": sum(row["fault_triggered"] for row in group),
            "fault_events": sum(row["injected_faults"] for row in group),
            "recovered_successes": sum(row["fault_triggered"] and row["success"] for row in valid),
            "known_cost_estimate_cny": sum(row["known_cost_estimate"] for row in group),
            "elapsed_seconds_sum": sum(row["elapsed_ms"] for row in group) / 1000,
        }

    summary = {
        "phase": phase, "status": ledger["status"], "model": protocol["settings"]["model"],
        "planned_sessions": len(plans), "completed_sessions": len(rows),
        "ledger_states": dict(Counter(entry["state"] for entry in ledger["entries"].values())),
        "counts": group_summary(rows), "integrity_errors": integrity_errors,
        "by_fault": {kind: group_summary([r for r in rows if r["condition"] == kind])
                     for kind in FAULT_KINDS},
        "by_split": {split: group_summary([r for r in rows if r["split"] == split])
                     for split in ("train", "validation", "test")},
        "by_template": {template: group_summary([r for r in rows if r["template_id"] == template])
                        for template in sorted({r["template_id"] for r in rows})},
        "known_tokens": {key: sum(u[key] for u in usage_rows if type(u[key]) is int and u[key] >= 0)
                         for key in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")},
        "usage_missing_requests": sum(r["unknown_usage_requests"] for r in rows),
        "cache_missing_requests": sum(r["unknown_cache_requests"] for r in rows),
        "reasoning_missing_responses": sum(u["reasoning_tokens"] is None for u in usage_rows),
        "truncated_responses": sum(r["truncated_responses"] for r in rows),
        "all_spend_includes_invalid": True, "currency": "CNY",
        "pricing": {"uncached_input_per_million": 13, "cached_input_per_million": 1.3,
                    "output_per_million": 68},
        "cost_notice": "Known usage priced, cache missing at full input rate; missing usage is "
        "unknown extra spend, never zero. Provider invoice may differ.",
        "timeout_notice": protocol["timeout_simulation"],
        "protocol_hash": protocol["fingerprint"], "design_hash": design["fingerprint"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "batch_active_seconds": ledger.get("active_seconds", 0),
    }
    if rows:
        summary["projected_600_cost_cny"] = summary["counts"]["known_cost_estimate_cny"] / len(rows) * 600
        with (directory / "sessions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(directory / "summary.json", summary)
    counts = summary["counts"]
    report = (
        f"# {phase} 采集报告\n\n"
        f"状态：{summary['status']}；计划 {len(plans)} Session，完成并通过完整性核验 {len(rows)}。\n\n"
        f"有效 {counts['valid']}，成功 {counts['success']}，失败 {counts['failure']}，"
        f"基础设施/usage 无效 {counts['invalid']}。实际模型请求 {counts['requests']}。\n\n"
        f"已知用量计费估算（含无效样本）：人民币 {counts['known_cost_estimate_cny']:.6f} 元。"
        f"未知 usage 请求 {summary['usage_missing_requests']}，缓存未知请求 "
        f"{summary['cache_missing_requests']}；未知支出不能视作零。\n\n"
        f"截断响应 {summary['truncated_responses']}；终止原因："
        f"`{json.dumps(counts['terminations'], ensure_ascii=False)}`。\n\n"
        "标签根据真实完成条件、全部写操作和最终状态确定，不由终止原因代替。"
        "推理 token 包含在输出中，不重复计费；缺失 reasoning/cache 不补造零。\n\n"
        "单 Session 最多12工具调用、累计30000输入+输出token、端到端120秒；"
        "HTTP请求使用剩余墙钟期限，流式读取可取消，无自动重试。token按已报告usage检查，"
        "单请求可能越过累计阈值，超限后不再执行工具。\n\n"
        "超时/限流注入立即返回模拟错误，不代表真实等待。故障概率0.7、持久概率0.3、"
        "瞬态窗口2次、种子42及派生规则保持不变。\n\n"
        "pilot仅6个预选训练模板各1实例×5条件，不进入正式600；正式40实例×5条件×3次。"
        "不使用验证/测试结果调参；本次不执行160次在线对照。\n\n"
        "完整原始数据在本目录各session子目录，分组、token、耗时与费用见summary.json；"
        "逐会话汇总见sessions.csv。断点恢复只运行pending；started但无完整结果不会自动重跑。\n\n"
        "## 故障分组\n\n| 条件 | 总数 | 成功 | 失败 | 无效 | 触发故障 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for kind, group in summary["by_fault"].items():
        report += (f"| {kind} | {group['sessions']} | {group['success']} | {group['failure']} | "
                   f"{group['invalid']} | {group['fault_triggered_sessions']} |\n")
    (directory / "report.md").write_text(report, encoding="utf-8")
    return summary


def freeze_after_pilot(root):
    protocol, design = read_protocol(root)
    summary = summarize(root, "pilot")
    if summary["completed_sessions"] != 30 or summary["integrity_errors"]:
        raise ValueError("Pilot incomplete or integrity check failed")
    if summary["counts"]["invalid"]:
        raise ValueError("Review infrastructure-invalid pilot sessions before freezing")
    freeze = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["fingerprint"], "design_hash": design["fingerprint"],
        "pilot_summary_hash": fingerprint(summary), "settings": protocol["settings"],
        "fault_design": protocol["fault_design"], "seed": protocol["seed"],
        "code_hashes": protocol["code_hashes"], "plan_hash": fingerprint(
            read_json(root / "collection/plan.json")),
        "pilot_counts": summary["counts"], "pilot_truncated": summary["truncated_responses"],
        "online_sessions": 0,
    }
    freeze["fingerprint"] = fingerprint(freeze)
    with (root / "frozen_collection.json").open("x", encoding="utf-8") as handle:
        json.dump(freeze, handle, ensure_ascii=False, indent=2)
    print("Frozen collection configuration after pilot integrity review")


def run_phase(root, phase):
    Settings.from_env()  # Load credentials privately; immutable settings come from the snapshot.
    protocol, design = read_protocol(root)
    directory = root / phase
    plans = read_json(directory / "plan.json")
    if len(plans) != (30 if phase == "pilot" else 600):
        raise ValueError("Plan cardinality mismatch")
    if phase == "collection":
        frozen = read_json(root / "frozen_collection.json")
        if (frozen["protocol_hash"] != protocol["fingerprint"]
                or frozen["plan_hash"] != fingerprint(plans)
                or frozen["fingerprint"] != fingerprint(
                    {k: v for k, v in frozen.items() if k != "fingerprint"})):
            raise ValueError("Collection freeze mismatch")
    settings = Settings.model_validate(protocol["settings"])
    by_id = {task["task_id"]: Task.model_validate(task) for task in design["tasks"]}
    with (directory / "runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A runner already holds this phase lock") from None
        ledger = read_json(directory / "ledger.json")
        # Recover only already-written results. Never rerun an ambiguous charged attempt.
        for spec in plans:
            entry = ledger["entries"][spec["condition_id"]]
            if entry["state"] == "started":
                matches = [path.parent for path in directory.glob("*/session.json")
                           if read_json(path)["condition_id"] == spec["condition_id"]]
                if len(matches) == 1 and (matches[0] / "result.json").exists():
                    audit_session(matches[0], protocol, design, spec)
                    entry.update(state="completed", path=str(matches[0]))
                else:
                    entry["state"] = "ambiguous_do_not_retry"
        ambiguous = any(entry["state"] == "ambiguous_do_not_retry"
                        for entry in ledger["entries"].values())
        if ambiguous or ledger["status"] == "paused_infrastructure":
            ledger["status"] = "paused_infrastructure"
            atomic_json(directory / "ledger.json", ledger)
            raise ValueError("Paused: inspect ambiguous/infrastructure attempts; no automatic refill")
        pending = iter(spec for spec in plans
                       if ledger["entries"][spec["condition_id"]]["state"] == "pending")
        started = perf_counter()
        ledger["status"] = "running"
        atomic_json(directory / "ledger.json", ledger)
        recent_invalid = []
        paused = False

        def execute(spec):
            return run_session(
                by_id[spec["task_id"]], settings,
                FaultConfig(kind=spec["condition"], seed=spec["fault_seed"],
                            **protocol["fault_design"]),
                directory, "openai", spec["agent_seed"], phase,
                experiment_id=root.name + "_" + phase, condition_id=spec["condition_id"],
                design_hash=design["fingerprint"],
            )

        with ThreadPoolExecutor(max_workers=protocol["workers"][phase]) as executor:
            futures = {}

            def submit_one():
                spec = next(pending, None)
                if spec is None:
                    return
                ledger["entries"][spec["condition_id"]] = {
                    "state": "started", "started_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(directory / "ledger.json", ledger)
                futures[executor.submit(execute, spec)] = spec

            for _ in range(protocol["workers"][phase]):
                submit_one()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    spec = futures.pop(future)
                    entry = ledger["entries"][spec["condition_id"]]
                    try:
                        result = future.result()
                        audit_session(Path(result["path"]), protocol, design, spec)
                        entry.update(state="completed", path=result["path"])
                        infra_invalid = (not result["valid_for_analysis"]
                                         and result["termination"] != "time_budget")
                        recent_invalid.append(infra_invalid)
                        auth_error = any(code in (result.get("failure_detail") or "")
                                         for code in ("HTTP 401", "HTTP 403"))
                        paused |= sum(recent_invalid[-10:]) >= 3 or auth_error
                        count = sum(e["state"] == "completed" for e in ledger["entries"].values())
                        print(f"{phase} {count}/{len(plans)} {spec['condition_id']} "
                              f"valid={result['valid_for_analysis']} "
                              f"termination={result['termination']}", flush=True)
                    except Exception as exc:
                        entry.update(state="ambiguous_do_not_retry", error_type=type(exc).__name__)
                        paused = True
                        print(f"Paused on runner error: {type(exc).__name__}", flush=True)
                    ledger["status"] = "paused_infrastructure" if paused else "running"
                    atomic_json(directory / "ledger.json", ledger)
                if not paused:
                    for _ in done:
                        submit_one()
        ledger["active_seconds"] = ledger.get("active_seconds", 0) + perf_counter() - started
        ledger["status"] = "paused_infrastructure" if paused else "complete"
        atomic_json(directory / "ledger.json", ledger)
        summary = summarize(root, phase)
        print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "pilot", "freeze", "collection", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-limit", type=int, default=1024)
    args = parser.parse_args()
    if args.action == "init":
        initialize(args.root, args.output_limit)
    elif args.action == "freeze":
        freeze_after_pilot(args.root)
    elif args.action == "status":
        Settings.from_env()
        for phase in ("pilot", "collection"):
            summary = summarize(args.root, phase)
            print(json.dumps({"phase": phase, "status": summary["status"],
                              **summary["counts"]}, ensure_ascii=False))
    else:
        run_phase(args.root, args.action)


if __name__ == "__main__":
    main()

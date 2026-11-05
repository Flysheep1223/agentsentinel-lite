"""At-most-once schema3 collection: manual batches or an independent automatic 6000 run."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .collection_run import audit_session, code_hashes
from .config import ROOT, Settings, fingerprint, read_json
from .dataset import load_design
from .environment import FaultConfig
from .expanded_design import make_design, make_plans, validate_expanded_design
from .harness import SYSTEM_PROMPT, run_session
from .tasks import Task

OLD_ROOT = ROOT / "data/authorized_20261025_30plus600_v1"
RESUME_CONFIRM = "REVIEWED_PAUSE_AND_RESUME_PENDING"
PRE_RESUME_RUNNER_HASH = "4a450abd295bd0b213efea83702f8cf90e9483ee4fa91b19645e44c1fc921463"
PRE_TRIAL_RUNNER_HASH = "cb42e0dae2b46ded4bd81737253e68ba8083994344046b3965cd664bb5f3cf9b"
AUTO_SETTINGS = {
    "max_tokens": None, "session_timeout": None, "max_completion_tokens": None,
    "max_steps": 12, "max_model_calls": 24, "request_timeout": 60,
}
AUTO_AUTHORIZATION = (
    "User authorized a fresh independent collection of 6000 sessions, in automatic batches "
    "of 50 concurrent sessions; no cumulative session token, session time or monetary ceiling. "
    "Provider-default output limit; 12 tool steps, 24 model calls, 60s network timeout. "
    "No probe or online sessions authorized; previous data are not reused."
)
AUTO_PAUSE_POLICY = (
    "Pause on runner errors, HTTP 401/403/429, or 3 infrastructure-invalid in latest 10; "
    "drain inflight. Ambiguous charged attempts never retried. Never auto-resume paused roots."
)


def collection_mode(protocol):
    mode = protocol.get("mode", "manual")
    if mode not in {"manual", "auto"}:
        raise ValueError("Unknown collection mode")
    return mode


def auto_settings(previous):
    return Settings.model_validate({**previous["settings"], **AUTO_SETTINGS}).model_dump()


class ConfigurationMismatch(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(value):
    return {**value, "fingerprint": fingerprint({k: v for k, v in value.items() if k != "fingerprint"})}


def checked(path):
    value = read_json(path)
    if not isinstance(value, dict) or value != seal(value):
        raise ValueError(f"Fingerprint mismatch: {path.name}")
    return value


@contextmanager
def root_lock(root):
    if (not root.is_dir() or root.resolve() == OLD_ROOT.resolve()
            or OLD_ROOT.resolve() in root.resolve().parents):
        raise ValueError("Expected an existing independent expanded root")
    with (root / "runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another operation holds this root lock") from None
        yield


def credentials(settings):
    current = Settings.from_env()
    if current.model != settings.model or current.base_url != settings.base_url:
        raise ConfigurationMismatch("Environment model/endpoint differs from the frozen previous batch")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing API credential")


def initialize(root: Path, old_root: Path = OLD_ROOT, mode="manual"):
    if mode not in {"manual", "auto"}:
        raise ValueError("Unknown collection mode")
    root = root.resolve()
    if root == old_root.resolve() or old_root.resolve() in root.parents or root.exists():
        raise ValueError("New independent root must not exist or be inside the previous batch")
    previous = checked(old_root / "frozen_collection.json")
    parent = load_design(old_root / "design.json")
    if previous["design_hash"] != parent["fingerprint"]:
        raise ValueError("Previous freeze/design mismatch")
    settings = Settings.model_validate(
        auto_settings(previous) if mode == "auto" else previous["settings"]
    )
    if "@" in settings.base_url or "?" in settings.base_url or "#" in settings.base_url:
        raise ValueError("Endpoint must not contain credentials, query or fragment")
    credentials(settings)
    design = make_design(parent)
    plans = make_plans(design)
    if mode == "auto":
        plans = {"collection": plans["collection"]}
    protocol = seal({
        "schema_version": 3, "created_at": now(), "settings": settings.model_dump(),
        "fault_design": previous["fault_design"], "seed": design["seed"],
        "design_hash": design["fingerprint"], "parent_freeze_hash": previous["fingerprint"],
        "parent_design_hash": parent["fingerprint"], "code_hashes": code_hashes(),
        "plan_hashes": {phase: fingerprint(plan) for phase, plan in plans.items()},
        "workers": 2, "batch_cap": 50, "probe_cap": 5, "automatic_api_retries": 0,
        "monetary_cap_cny": None, "system_prompt": SYSTEM_PROMPT,
        "authorization": "User authorized independent 6000 sessions without monetary ceiling; "
        "first only five probe sessions, then explicit human approval per batch of at most 50.",
        "manual_batch_gate": True, "online_sessions_authorized": 0,
        "pause_policy": "Any auth/runner error or 3 infrastructure-invalid in latest 10; "
        "drain inflight; ambiguous charged attempts never retried. No automatic resume.",
        "downstream_support": design["downstream_support"],
    })
    if mode == "auto":
        protocol = seal({
            **protocol, "mode": "auto", "workers": 50, "probe_cap": 0,
            "manual_batch_gate": False, "collection_sessions_authorized": 6000,
            "probe_sessions_authorized": 0, "authorization": AUTO_AUTHORIZATION,
            "pause_policy": AUTO_PAUSE_POLICY,
        })
    root.mkdir(parents=True, exist_ok=False)
    with root_lock(root):
        save(root / "parent_design.json", parent)
        save(root / "parent_freeze.json", previous)
        save(root / "design.json", design)
        save(root / "protocol.json", protocol)
        for phase, plan in plans.items():
            save(root / phase / "plan.json", plan)
        save(root / "ledger.json", seal({
            "protocol_hash": protocol["fingerprint"], "status": "ready", "recent_invalid": [],
            "batches": [], "approval": None,
            "entries": {s["condition_id"]: {"state": "pending"}
                        for plan in plans.values() for s in plan},
        }))
    return {"root": str(root), "mode": mode, "probe": 0 if mode == "auto" else 5,
            "collection": 6000, "protocol_hash": protocol["fingerprint"], "model": settings.model}


def load(root):
    if not __debug__:
        raise ValueError("Do not run with python -O: artifact audit requires assertions")
    protocol = checked(root / "protocol.json")
    current_hashes = code_hashes()
    compatible_hashes = [
        {**current_hashes, "expanded_collection.py": runner_hash}
        for runner_hash in (PRE_RESUME_RUNNER_HASH, PRE_TRIAL_RUNNER_HASH)
    ]
    if (protocol["code_hashes"] != current_hashes
            and not (collection_mode(protocol) == "auto"
                     and protocol["code_hashes"] in compatible_hashes)):
        raise ValueError("Source snapshot changed; do not run this root")
    parent = load_design(root / "parent_design.json")
    previous = checked(root / "parent_freeze.json")
    if previous["fingerprint"] != protocol["parent_freeze_hash"]:
        raise ValueError("Parent freeze mismatch")
    design = validate_expanded_design(read_json(root / "design.json"), parent)
    mode = collection_mode(protocol)
    automatic = mode == "auto"
    settings = Settings.model_validate(protocol["settings"], strict=True)
    expected_settings = auto_settings(previous) if automatic else previous["settings"]
    if (design["fingerprint"] != protocol["design_hash"]
            or parent["fingerprint"] != protocol["parent_design_hash"]
            or previous["design_hash"] != parent["fingerprint"]
            or protocol["settings"] != expected_settings
            or protocol["fault_design"] != previous["fault_design"]
            or type(protocol["workers"]) is not int
            or protocol["workers"] != (50 if automatic else 2)
            or type(protocol["batch_cap"]) is not int or protocol["batch_cap"] != 50
            or protocol["probe_cap"] != (0 if automatic else 5)
            or protocol["manual_batch_gate"] is not (not automatic)):
        raise ValueError("Protocol differs from frozen design/settings or hard limits")
    if automatic and (
        protocol.get("collection_sessions_authorized") != 6000
        or protocol.get("probe_sessions_authorized") != 0
        or protocol.get("online_sessions_authorized") != 0
        or protocol.get("authorization") != AUTO_AUTHORIZATION
        or protocol.get("pause_policy") != AUTO_PAUSE_POLICY
        or protocol.get("automatic_api_retries") != 0
        or protocol.get("monetary_cap_cny") is not None
        or protocol.get("system_prompt") != SYSTEM_PROMPT
        or protocol.get("schema_version") != 3
        or any(c in settings.base_url for c in "@?#")
    ):
        raise ValueError("Automatic protocol authorization or safety limits differ")
    expected_plans = make_plans(design)
    if automatic:
        expected_plans = {"collection": expected_plans["collection"]}
    plans = {phase: read_json(root / phase / "plan.json") for phase in expected_plans}
    if set(protocol["plan_hashes"]) != set(plans) or plans != expected_plans or any(
        fingerprint(plan) != protocol["plan_hashes"][phase] for phase, plan in plans.items()
    ):
        raise ValueError("Frozen plan mismatch")
    ledger = checked(root / "ledger.json")
    expected_ids = {s["condition_id"] for plan in plans.values() for s in plan}
    if ledger["protocol_hash"] != protocol["fingerprint"] or set(ledger["entries"]) != expected_ids:
        raise ValueError("Ledger does not match protocol/plan")
    if any(e["state"] not in {"pending", "started", "completed", "ambiguous_do_not_retry"}
           for e in ledger["entries"].values()):
        raise ValueError("Unknown ledger state")
    if ledger["status"] not in {"ready", "running", "batch_complete", "paused", "complete"}:
        raise ValueError("Unknown runner status")
    if automatic and ledger["approval"] is not None:
        raise ValueError("Automatic mode cannot consume manual approvals")
    return protocol, design, plans, ledger


def persist(root, ledger):
    save(root / "ledger.json", seal(ledger))


def audit(directory, root, phase, protocol, design, spec):
    directory = Path(directory).resolve()
    if directory.parent != (root / phase).resolve():
        raise ValueError("Session path outside frozen phase directory")
    meta, result, events = audit_session(directory, protocol, design, spec)
    expected_faults = FaultConfig(kind=spec["condition"], seed=spec["fault_seed"],
                                  **protocol["fault_design"]).model_dump()
    if (meta["fault_config"] != expected_faults or meta["agent_seed"] != spec["agent_seed"]
            or meta["phase"] != ("pilot" if phase == "probe" else "collection")
            or meta["backend"] != "openai" or meta["policy"] != "none"
            or meta["session_id"] != directory.name or result["session_id"] != directory.name
            or meta["config_hash"] != fingerprint(protocol["settings"])
            or meta["experiment_id"] != root.name + "_" + phase):
        raise ValueError("Session provenance mismatch")
    return result, events


def recover(root, protocol, design, plans, ledger):
    for phase, plan in plans.items():
        for spec in plan:
            entry = ledger["entries"][spec["condition_id"]]
            if entry["state"] == "completed":
                try:
                    audit(entry["path"], root, phase, protocol, design, spec)
                except Exception as exc:
                    entry.update(state="ambiguous_do_not_retry", error_type=type(exc).__name__)
                    ledger["status"] = "paused"
                continue
            if entry["state"] != "started":
                continue
            try:
                matches = [p.parent for p in (root / phase).glob("*/session.json")
                           if read_json(p).get("condition_id") == spec["condition_id"]]
                if len(matches) != 1:
                    raise ValueError("Incomplete or duplicated charged attempt")
                result, _ = audit(matches[0], root, phase, protocol, design, spec)
                entry.update(state="completed", path=str(matches[0]), recovered=True)
                record_health(ledger, result)
            except Exception as exc:
                entry.update(state="ambiguous_do_not_retry", error_type=type(exc).__name__)
                ledger["status"] = "paused"
    if any(e["state"] == "ambiguous_do_not_retry" for e in ledger["entries"].values()):
        ledger["status"] = "paused"
    for batch in ledger["batches"]:
        if batch["status"] == "running":
            batch["status"] = "paused" if ledger["status"] == "paused" else "interrupted_recovered"
            batch["finished_at"] = now()
            if ledger["status"] != "paused":
                ledger["status"] = "batch_complete"
    persist(root, ledger)


def record_health(ledger, result):
    ledger["recent_invalid"] = (ledger["recent_invalid"] + [not result["valid_for_analysis"]])[-10:]
    provider_blocked = any(code in (result.get("failure_detail") or "")
                           for code in ("HTTP 401", "HTTP 403", "HTTP 429"))
    if provider_blocked or sum(ledger["recent_invalid"]) >= 3:
        ledger["status"] = "paused"


def probe_gate(root, protocol, design, plans, ledger):
    hashes = []
    for spec in plans["probe"]:
        entry = ledger["entries"][spec["condition_id"]]
        if entry["state"] != "completed":
            raise ValueError("All five probe sessions must complete before approval")
        result, events = audit(entry["path"], root, "probe", protocol, design, spec)
        if not result["valid_for_analysis"]:
            raise ValueError("Probe has infrastructure-invalid results; do not collect")
        hashes.append(fingerprint([result, events]))
    return fingerprint(hashes)


def approve(root, count, confirm):
    if confirm != "REVIEWED_PROBE_AND_APPROVE_ONE_BATCH" or not 1 <= count <= 50:
        raise ValueError("Explicit human review confirmation and batch size 1..50 required")
    with root_lock(root):
        protocol, design, plans, ledger = load(root)
        if collection_mode(protocol) != "manual":
            raise ValueError("Automatic root requires collect-all, not manual approval")
        recover(root, protocol, design, plans, ledger)
        if ledger["status"] == "paused":
            raise ValueError("Paused; no automatic override/resume")
        probe_hash = probe_gate(root, protocol, design, plans, ledger)
        if ledger["approval"] is not None:
            raise ValueError("An unconsumed batch approval already exists")
        ids = [s["condition_id"] for s in plans["collection"]
               if ledger["entries"][s["condition_id"]]["state"] == "pending"][:count]
        if not ids:
            raise ValueError("No pending formal sessions")
        approval = seal({"ids": ids, "probe_hash": probe_hash,
                         "protocol_hash": protocol["fingerprint"], "approved_at": now(),
                         "batch_number": len(ledger["batches"]) + 1})
        ledger["approval"] = approval
        persist(root, ledger)
        return {"approved_sessions": len(ids), "approval_token": approval["fingerprint"]}


def run_batch(root, phase, approval_token=None, workers=2):
    if phase not in {"probe", "collection"} or workers not in (1, 2):
        raise ValueError("Only probe/collection and one or two workers allowed")
    with root_lock(root):
        protocol, design, plans, ledger = load(root)
        if collection_mode(protocol) != "manual":
            raise ValueError("Automatic root requires collect-all")
        recover(root, protocol, design, plans, ledger)
        if ledger["status"] == "paused":
            raise ValueError("Paused: inspect infrastructure/ambiguous attempts; never auto retry")
        if phase == "probe":
            chosen = [s for s in plans[phase] if ledger["entries"][s["condition_id"]]["state"] == "pending"]
            if len(chosen) > 5:
                raise ValueError("Probe hard cap exceeded")
        else:
            probe_hash = probe_gate(root, protocol, design, plans, ledger)
            approval = ledger["approval"]
            if (not approval or approval != seal(approval)
                    or approval["fingerprint"] != approval_token
                    or approval["probe_hash"] != probe_hash
                    or approval["protocol_hash"] != protocol["fingerprint"]):
                raise ValueError("Missing/stale explicit one-batch approval")
            pending = [s for s in plans[phase] if ledger["entries"][s["condition_id"]]["state"] == "pending"]
            chosen = pending[:len(approval["ids"])]
            if not 1 <= len(chosen) <= 50 or [s["condition_id"] for s in chosen] != approval["ids"]:
                raise ValueError("Approval differs from pending batch")
        if not chosen:
            return summary_locked(root, protocol, design, plans, ledger)
        settings = Settings.model_validate(protocol["settings"])
        credentials(settings)
        return execute_batch_locked(root, phase, protocol, design, plans, ledger,
                                    chosen, settings, workers)


def execution_config(protocol, workers, count, trial_count=None):
    return {
        "workers": workers, "count": count, "frozen_workers": protocol["workers"],
        "frozen_batch_cap": protocol["batch_cap"], "trial_count": trial_count,
        "configuration_override": (
            f"Explicit bounded single-batch trial: workers {protocol['workers']} -> {workers}; "
            f"count capped at {trial_count}; frozen protocol/design/plan unchanged."
            if trial_count is not None else "No bounded-trial override."
        ),
        "code_hashes": code_hashes(),
    }


def validate_trial(trial_count, trial_workers):
    if trial_count is None and trial_workers is None:
        return
    if (type(trial_count) is not int or type(trial_workers) is not int
            or not 1 <= trial_workers <= trial_count <= 50):
        raise ValueError("Both trial-count and trial-workers required: 1 <= workers <= count <= 50")


def execute_batch_locked(root, phase, protocol, design, plans, ledger, chosen, settings, workers,
                         trial_count=None):
    validate_trial(trial_count, workers if trial_count is not None else None)
    if collection_mode(protocol) == "auto":
        if trial_count is None and workers != protocol["workers"]:
            raise ValueError("Automatic workers override requires bounded trial mode")
        if trial_count is not None and not 1 <= len(chosen) <= trial_count:
            raise ValueError("Bounded trial count exceeded")
    elif trial_count is not None:
        raise ValueError("Bounded trial requires an automatic root")
    ledger["approval"] = None  # Consume BEFORE any paid attempt, including partial batches.
    batch = {"phase": phase, "ids": [s["condition_id"] for s in chosen],
             "started_at": now(), "status": "running",
             **execution_config(protocol, workers, len(chosen), trial_count)}
    ledger["batches"].append(batch)
    ledger["status"] = "running"
    persist(root, ledger)
    tasks = {t["task_id"]: Task.model_validate(t) for t in design["tasks"]}
    pending = iter(chosen)

    def execute(spec):
        return run_session(
            tasks[spec["task_id"]], settings,
            FaultConfig(kind=spec["condition"], seed=spec["fault_seed"], **protocol["fault_design"]),
            root / phase, "openai", spec["agent_seed"], "pilot" if phase == "probe" else "collection",
            experiment_id=root.name + "_" + phase, condition_id=spec["condition_id"],
            design_hash=design["fingerprint"],
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            def submit():
                spec = next(pending, None)
                if spec is None:
                    return
                entry = ledger["entries"][spec["condition_id"]]
                entry.update(state="started", started_at=now())
                persist(root, ledger)
                try:
                    futures[executor.submit(execute, spec)] = spec
                except Exception as exc:
                    entry.update(state="ambiguous_do_not_retry", error_type=type(exc).__name__)
                    ledger["status"] = "paused"
                    persist(root, ledger)

            for _ in range(workers):
                if ledger["status"] != "paused":
                    submit()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: futures[f]["condition_id"]):
                    spec = futures.pop(future)
                    entry = ledger["entries"][spec["condition_id"]]
                    try:
                        returned = future.result()
                        result, _ = audit(returned["path"], root, phase, protocol, design, spec)
                        entry.update(state="completed", path=returned["path"])
                        record_health(ledger, result)
                        print(json.dumps({"event": "session_complete", "condition_id": spec["condition_id"],
                                          **result_row(result)}, ensure_ascii=False), flush=True)
                    except Exception as exc:
                        entry.update(state="ambiguous_do_not_retry", error_type=type(exc).__name__)
                        ledger["status"] = "paused"
                    persist(root, ledger)
                if ledger["status"] != "paused":
                    for _ in done:
                        submit()
                        if ledger["status"] == "paused":
                            break
    except BaseException:
        ledger["status"] = "paused"
        raise
    finally:
        batch["status"] = "paused" if ledger["status"] == "paused" else "batch_complete"
        batch["finished_at"] = now()
        ledger["status"] = batch["status"]
        if collection_mode(protocol) == "auto" and ledger["status"] != "paused" and all(
            e["state"] == "completed" for e in ledger["entries"].values()
        ):
            ledger["status"] = "complete"
        persist(root, ledger)
        try:
            result = summary_locked(root, protocol, design, plans, ledger)
        except Exception:
            ledger["status"] = batch["status"] = "paused"
            persist(root, ledger)
            raise
    return result


def collect_all(root, resume=False, confirm=None, trial_count=None, trial_workers=None):
    validate_trial(trial_count, trial_workers)
    if resume and confirm != RESUME_CONFIRM:
        raise ValueError("Explicit human pause review confirmation required")
    # Keep one exclusive lock across recovery, every batch, and all summary writes.
    with root_lock(root):
        protocol, design, plans, ledger = load(root)
        if collection_mode(protocol) != "auto":
            raise ValueError("collect-all requires a new independent auto root; manual roots unchanged")
        was_paused = ledger["status"] == "paused"
        recover(root, protocol, design, plans, ledger)
        if resume:
            if not was_paused or ledger["status"] != "paused":
                raise ValueError("Explicit resume requires an already paused automatic root")
            if any(e["state"] in {"started", "ambiguous_do_not_retry"}
                   for e in ledger["entries"].values()):
                raise ValueError("Paused: unresolved charged attempts prohibit resume")
        elif ledger["status"] == "paused":
            summary_locked(root, protocol, design, plans, ledger)
            raise ValueError("Paused: no automatic resume or retry of ambiguous attempts")
        settings = Settings.model_validate(protocol["settings"])
        credentials(settings)
        workers = protocol["workers"] if trial_workers is None else trial_workers
        while True:
            chosen = []
            for start in range(0, len(plans["collection"]), protocol["batch_cap"]):
                block = plans["collection"][start:start + protocol["batch_cap"]]
                chosen = [s for s in block
                          if ledger["entries"][s["condition_id"]]["state"] == "pending"]
                if chosen:
                    break
            if trial_count is not None:
                chosen = chosen[:trial_count]
            if resume:
                ledger.setdefault("resumptions", []).append({
                    "resumed_at": now(), "confirmation": confirm,
                    "previous_status": ledger["status"],
                    "recent_invalid": list(ledger["recent_invalid"]),
                    "protocol_hash": protocol["fingerprint"],
                    **execution_config(protocol, workers, len(chosen), trial_count),
                })
                ledger["recent_invalid"] = []
                ledger["status"] = "ready"
                persist(root, ledger)
                resume = False
            if not chosen:
                ledger["status"] = "complete"
                persist(root, ledger)
                return summary_locked(root, protocol, design, plans, ledger)
            trial_options = {} if trial_count is None else {"trial_count": trial_count}
            result = execute_batch_locked(root, "collection", protocol, design, plans, ledger,
                                          chosen, settings, workers, **trial_options)
            if trial_count is not None or ledger["status"] in {"paused", "complete"}:
                return result


def result_row(result):
    totals = result["totals"]
    return {key: result[key] for key in (
        "elapsed_ms", "valid_for_analysis", "success", "termination",
    )} | {key: totals[key] for key in (
        "model_calls", "total_tokens", "known_input_tokens", "known_output_tokens",
        "known_cost_estimate", "unknown_usage_requests", "unknown_cache_requests",
    )}


def summary_locked(root, protocol, design, plans, ledger):
    rows, errors, partial = [], [], []
    for phase, plan in plans.items():
        for spec in plan:
            entry = ledger["entries"][spec["condition_id"]]
            if entry["state"] == "completed":
                try:
                    result, _ = audit(entry["path"], root, phase, protocol, design, spec)
                    rows.append({"phase": phase, **spec, **result_row(result)})
                except Exception as exc:
                    errors.append({"condition_id": spec["condition_id"], "error_type": type(exc).__name__})
            elif entry["state"] in {"started", "ambiguous_do_not_retry"}:
                partial.append(spec["condition_id"])
    # Include known partial or audit-invalid usage, but never pretend missing usage is zero.
    partial_known, partial_unknown, unresolved = 0.0, 0, []
    accounted = {r["condition_id"] for r in rows}
    for phase in plans:
        for metadata in (root / phase).glob("*/session.json"):
            try:
                meta = read_json(metadata)
                if meta["condition_id"] in accounted:
                    continue
                events = []
                for line in (metadata.parent / "events.jsonl").read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
                ends = [e for e in events if e["kind"] == "llm_end"]
                starts = sum(e["kind"] == "llm_start" for e in events)
                partial_unknown += max(0, starts - len(ends))
                for event in ends:
                    usage = event["usage"]
                    inp, out, cache = (usage.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens"))
                    if (type(inp) is int and inp >= 0 and type(out) is int and out >= 0
                            and (cache is None or type(cache) is int and 0 <= cache <= inp)):
                        partial_known += Settings.model_validate(protocol["settings"]).cost(inp, out, cache) or 0
                    else:
                        partial_unknown += 1
            except Exception:
                unresolved.append(metadata.parent.name)
    if errors:
        ledger["status"] = "paused"
        persist(root, ledger)
    summary = {
        "status": ledger["status"], "mode": collection_mode(protocol),
        "protocol_hash": protocol["fingerprint"],
        "planned": {phase: len(plan) for phase, plan in plans.items()},
        "ledger_states": dict(Counter(e["state"] for e in ledger["entries"].values())),
        "completed": {phase: sum(r["phase"] == phase for r in rows) for phase in plans},
        "valid": sum(r["valid_for_analysis"] for r in rows), "integrity_errors": errors,
        "known_cost_estimate_cny": sum(r["known_cost_estimate"] for r in rows) + partial_known,
        "unknown_usage_requests": sum(r["unknown_usage_requests"] for r in rows) + partial_unknown,
        "ambiguous_attempts": partial, "unreadable_usage_artifacts": unresolved,
        "cost_notice": "Includes invalid and readable partial attempts. Ambiguous attempts may "
        "have additional unknown spend, including requests with no artifact; never interpret as zero.",
        "downstream_support": design["downstream_support"], "sessions": rows,
    }
    save(root / "summary.json", summary)
    return summary


def status(root):
    with root_lock(root):
        protocol, design, plans, ledger = load(root)
        return summary_locked(root, protocol, design, plans, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[
        "init", "probe", "approve", "collect-batch", "collect-all", "status",
    ])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual",
                        help="Initialization only; existing roots retain their frozen mode")
    parser.add_argument("--parent-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="collect-all only: resume audited paused auto root with --confirm "
                        + RESUME_CONFIRM)
    parser.add_argument("--confirm")
    parser.add_argument("--approval-token")
    parser.add_argument("--trial-count", type=int,
                        help="collect-all only: run at most this many pending sessions in one batch")
    parser.add_argument("--trial-workers", type=int,
                        help="Explicit bounded-trial concurrency; requires --trial-count")
    args = parser.parse_args()
    if args.resume and (args.action != "collect-all" or args.confirm != RESUME_CONFIRM):
        parser.error("--resume requires collect-all --confirm " + RESUME_CONFIRM)
    trial_options = {}
    if args.trial_count is not None or args.trial_workers is not None:
        if args.action != "collect-all":
            parser.error("--trial-count and --trial-workers require collect-all")
        try:
            validate_trial(args.trial_count, args.trial_workers)
        except ValueError as exc:
            parser.error(str(exc))
        trial_options = {"trial_count": args.trial_count, "trial_workers": args.trial_workers}
    root = args.root.resolve()
    try:
        if args.action == "init":
            result = initialize(root, args.parent_root, args.mode)
        elif args.action == "collect-all":
            result = collect_all(root, resume=args.resume, confirm=args.confirm, **trial_options)
        elif args.action == "approve":
            result = approve(root, args.count, args.confirm)
        elif args.action == "status":
            result = status(root)
        else:
            result = run_batch(root, "probe" if args.action == "probe" else "collection",
                               args.approval_token, args.workers)
        print(json.dumps(result, ensure_ascii=False))
    except ConfigurationMismatch:
        print(json.dumps({"status": "blocked", "reason": "environment_model_or_endpoint_mismatch",
                          "notice": "Model/endpoint must match the old frozen settings; no requests sent."}))
        raise SystemExit(1) from None
    except Exception as exc:
        # Never print provider exception contents or authentication values.
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__,
                          "action": args.action, "notice": "No automatic retry; inspect locally."}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

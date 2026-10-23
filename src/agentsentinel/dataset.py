from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .config import fingerprint, read_json, write_json
from .features import FEATURE_COLUMNS, FeatureBuilder


def template_split(templates: dict[str, str], seed: int = 42) -> dict[str, str]:
    assignments = {}
    rng = random.Random(seed)
    for workflow in sorted(set(templates.values())):
        names = sorted(t for t, w in templates.items() if w == workflow)
        if len(names) < 5:
            raise ValueError(f"{workflow}: 至少需要 5 个任务模板才能分组切分。")
        rng.shuffle(names)
        validation_count = max(1, round(len(names) * 0.2))
        train_count = len(names) - 2 * validation_count
        for i, name in enumerate(names):
            assignments[name] = (
                "train" if i < train_count
                else "validation" if i < train_count + validation_count
                else "test"
            )
    return assignments


def build_dataset(runs: Path, output: Path, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，避免覆盖已有切分：{output}")
    records, skipped = [], []
    session_ids = set()
    for metadata_path in sorted(runs.glob("*/session.json")):
        directory = metadata_path.parent
        try:
            metadata = read_json(metadata_path)
            result = read_json(directory / "result.json")
            read_json(directory / "trajectory.json")
            events = [
                json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"不完整或损坏的 Session: {directory.name}: {type(exc).__name__}") from exc
        if metadata["policy"] != "none" or metadata["phase"] == "online":
            skipped.append({"session_id": directory.name, "reason": "intervention"})
            continue
        if not result["valid_for_analysis"]:
            skipped.append({"session_id": directory.name, "reason": result["termination"]})
            continue
        session_id = metadata["session_id"]
        if session_id in session_ids:
            raise ValueError(f"重复 Session: {session_id}")
        session_ids.add(session_id)
        tools = [event for event in events if event["kind"] == "tool_end"]
        if not events or events[-1]["kind"] != "session_end":
            raise ValueError(f"缺少结束事件: {session_id}")
        if [e["step"] for e in tools] != list(range(1, len(tools) + 1)):
            raise ValueError(f"调用序号不连续: {session_id}")
        if result["totals"]["tool_calls"] != len(tools):
            raise ValueError(f"调用数与结果不一致: {session_id}")
        if any(e["latency_ms"] < 0 or e["elapsed_ms"] < 0 for e in tools):
            raise ValueError(f"负耗时: {session_id}")
        records.append((metadata, result, tools))
    if not records:
        raise ValueError("没有可用于分析的完整基线 Session。")
    provenance = {(m["phase"], m["backend"], m["config_hash"]) for m, _, _ in records}
    if len(provenance) != 1:
        raise ValueError("不能混合不同阶段、后端或 Agent 配置；请使用独立采集目录。")
    phase, backend, config_hash = next(iter(provenance))
    fault_design = {
        key: value for key, value in records[0][0]["fault_config"].items()
        if key not in {"kind", "seed"}
    }
    if any(
        {k: v for k, v in m["fault_config"].items() if k not in {"kind", "seed"}} != fault_design
        for m, _, _ in records
    ):
        raise ValueError("不能混合不同故障强度或触发规则；请使用独立采集目录。")
    assignments = template_split({m["template_id"]: m["workflow"] for m, _, _ in records}, seed)
    prefixes, sessions = [], []
    for metadata, result, tools in records:
        common = {
            key: metadata[key] for key in ("session_id", "task_id", "template_id", "workflow")
        }
        common.update({
            "split": assignments[metadata["template_id"]], "label": result["label"],
            "fault_kind": metadata["fault_config"]["kind"],
        })
        session = {
            **common, "total_steps": result["totals"]["tool_calls"],
            "final_tokens": result["totals"]["total_tokens"],
            "final_cost_usd": result["totals"]["cost_usd"],
            "final_elapsed_ms": result["elapsed_ms"],
            "first_violation_step": result["first_violation_step"],
            "termination": result["termination"],
        }
        sessions.append(session)
        builder = FeatureBuilder(metadata["workflow"])
        for event in tools:
            row = builder.add(event)
            if event["decision_eligible"]:
                prefixes.append({**common, **row})
    if not prefixes:
        raise ValueError("没有预算耗尽前的可用决策前缀。")
    prefix_df = pd.DataFrame(prefixes)
    session_df = pd.DataFrame(sessions)
    for column in FEATURE_COLUMNS:
        if column not in prefix_df:
            raise ValueError(f"缺少特征: {column}")
    manifest = {
        "schema_version": 1, "seed": seed, "assignments": assignments,
        "phase": phase, "backend": backend, "config_hash": config_hash,
        "session_ids": sorted(session_ids),
        "feature_columns": FEATURE_COLUMNS,
        "fault_design": fault_design,
        "fault_conditions": sorted({m["fault_config"]["kind"] for m, _, _ in records}),
        "monitor_price_per_second": records[0][0]["settings"]["monitor_price_per_second"],
    }
    output.mkdir(parents=True, exist_ok=True)
    prefix_df.to_csv(output / "prefixes.csv", index=False)
    session_df.to_csv(output / "sessions.csv", index=False)
    manifest["file_hashes"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("prefixes.csv", "sessions.csv")
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(output / "split.json", manifest)
    quality = {
        "sessions": len(sessions), "prefixes": len(prefixes), "excluded": skipped,
        "provenance": {"phase": phase, "backend": backend},
        "partitions": {
            name: {
                "templates": int(group.template_id.nunique()), "sessions": len(group),
                "failures": int(group.label.sum()), "failure_rate": float(group.label.mean()),
            }
            for name, group in session_df.groupby("split")
        },
        "missing_feature_values": {
            name: int(prefix_df[name].isna().sum()) for name in FEATURE_COLUMNS
        },
    }
    write_json(output / "quality.json", quality)
    return quality


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = read_json(path / "split.json")
    if fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("数据切分清单已在冻结后改变。")
    for name, expected in manifest["file_hashes"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"数据文件已在冻结后改变：{name}")
    return (
        pd.read_csv(path / "prefixes.csv"), pd.read_csv(path / "sessions.csv"),
        manifest,
    )


def training_eda(dataset: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefixes, sessions, _ = load_dataset(dataset)
    sessions = sessions[sessions.split == "train"]
    prefixes = prefixes[prefixes.split == "train"]
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "training only",
        "by_fault": sessions.groupby("fault_kind").label.agg(["count", "mean"]).to_dict("index"),
        "missing_values": prefixes[FEATURE_COLUMNS].isna().sum().to_dict(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sessions.groupby("fault_kind").label.mean().plot.bar(ax=axes[0], ylim=(0, 1))
    axes[0].set_ylabel("Failure rate (train)")
    sessions.boxplot(column="final_tokens", by="label", ax=axes[1])
    axes[1].set_title("Token totals by outcome (train)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output / "training_eda.png", dpi=160)
    plt.close(fig)
    write_json(output / "training_summary.json", summary)
    return summary

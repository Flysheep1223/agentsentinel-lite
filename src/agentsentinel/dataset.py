from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .config import fingerprint, read_json, write_json
from .features import FEATURE_COLUMNS, FeatureBuilder
from .tasks import make_tasks


def template_split(templates: dict[str, str], seed: int = 42) -> dict[str, str]:
    workflows = sorted(set(templates.values()))
    if len(workflows) != 2 or len(templates) != 10:
        raise ValueError("设计必须包含两个工作流，每个工作流恰好 5 个任务模板。")
    assignments = {}
    rng = random.Random(seed)
    for workflow in workflows:
        names = sorted(t for t, w in templates.items() if w == workflow)
        if len(names) != 5:
            raise ValueError(f"{workflow}: 必须恰好包含 5 个任务模板。")
        rng.shuffle(names)
        for i, name in enumerate(names):
            assignments[name] = "train" if i < 3 else "validation" if i == 3 else "test"
    return assignments


def _selected_task_ids(tasks: list[dict], assignments: dict, split: str, count: int) -> list[str]:
    return [
        task_id
        for template_id in sorted(assignments)
        if assignments[template_id] == split
        for task_id in sorted(
            task["task_id"] for task in tasks if task["template_id"] == template_id
        )[:count]
    ]


def validate_design(design: dict) -> dict:
    required = {
        "schema_version", "seed", "tasks", "assignments", "cv_folds",
        "pilot_task_ids", "online_task_ids", "fingerprint",
    }
    if not isinstance(design, dict) or set(design) != required:
        raise ValueError("冻结设计字段不完整或包含未知字段。")
    if type(design["schema_version"]) is not int or design["schema_version"] != 2:
        raise ValueError("冻结设计必须使用 schema_version=2。")
    if type(design["seed"]) is not int:
        raise ValueError("冻结设计 seed 必须是整数。")
    if fingerprint({k: v for k, v in design.items() if k != "fingerprint"}) != design["fingerprint"]:
        raise ValueError("设计已在冻结后改变，fingerprint 不匹配。")
    tasks = design["tasks"]
    if not isinstance(tasks, list) or any(
        not isinstance(task, dict) or not isinstance(task.get("task_id"), str) for task in tasks
    ):
        raise ValueError("冻结设计 tasks 必须包含完整 Task 数据。")
    current_tasks = {task.task_id: task.model_dump() for task in make_tasks()}
    frozen_tasks = {task["task_id"]: task for task in tasks}
    if (
        len(tasks) != 40 or len(frozen_tasks) != len(tasks)
        or fingerprint(frozen_tasks) != fingerprint(current_tasks)
    ):
        raise ValueError("冻结任务目录与当前 make_tasks 不一致；不能混用旧目录或修改任务。")
    templates = {task["template_id"]: task["workflow"] for task in tasks}
    template_split(templates, design["seed"])
    if any(sum(task["template_id"] == name for task in tasks) != 4 for name in templates):
        raise ValueError("每个模板必须恰好包含 4 个任务实例。")
    assignments = design["assignments"]
    if not isinstance(assignments, dict) or set(assignments) != set(templates):
        raise ValueError("设计分区必须覆盖且仅覆盖全部 10 个模板。")
    if any(split not in ("train", "validation", "test") for split in assignments.values()):
        raise ValueError("设计包含未知分区。")
    workflows = sorted(set(templates.values()))
    for workflow in workflows:
        counts = {
            split: sum(templates[name] == workflow and value == split
                       for name, value in assignments.items())
            for split in ("train", "validation", "test")
        }
        if counts != {"train": 3, "validation": 1, "test": 1}:
            raise ValueError(f"{workflow}: 分区必须为 3/1/1 个模板。")
    folds = design["cv_folds"]
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("设计必须包含 3 个显式交叉验证折。")
    held_out = []
    for index, fold in enumerate(folds, 1):
        if (
            not isinstance(fold, list) or len(fold) != 2
            or any(not isinstance(name, str) for name in fold)
            or len(set(fold)) != 2
            or any(assignments.get(name) != "train" for name in fold)
        ):
            raise ValueError(f"交叉验证第 {index} 折必须留出两个不同训练模板。")
        if sorted(templates[name] for name in fold) != workflows:
            raise ValueError(f"交叉验证第 {index} 折必须从两个工作流各留出一个模板。")
        held_out.extend(fold)
    training_templates = sorted(name for name, split in assignments.items() if split == "train")
    if sorted(held_out) != training_templates:
        raise ValueError("三个交叉验证折必须恰好覆盖全部训练模板一次。")
    if design["pilot_task_ids"] != _selected_task_ids(tasks, assignments, "train", 1):
        raise ValueError("pilot_task_ids 必须为每个训练模板按 task_id 排序的首个实例，共 6 个。")
    if design["online_task_ids"] != _selected_task_ids(tasks, assignments, "test", 2):
        raise ValueError("online_task_ids 必须为每个测试模板按 task_id 排序的前两个实例，共 4 个。")
    return design


def freeze_design(path: Path, seed: int = 42) -> dict:
    if path.exists():
        raise ValueError(f"冻结设计已存在，不允许覆盖：{path}")
    tasks = [task.model_dump() for task in make_tasks()]
    templates = {task["template_id"]: task["workflow"] for task in tasks}
    assignments = template_split(templates, seed)
    training_by_workflow = [
        sorted(name for name, split in assignments.items()
               if split == "train" and templates[name] == workflow)
        for workflow in sorted(set(templates.values()))
    ]
    design = {
        "schema_version": 2, "seed": seed, "tasks": tasks, "assignments": assignments,
        "cv_folds": [list(fold) for fold in zip(*training_by_workflow)],
        "pilot_task_ids": _selected_task_ids(tasks, assignments, "train", 1),
        "online_task_ids": _selected_task_ids(tasks, assignments, "test", 2),
    }
    design["fingerprint"] = fingerprint(design)
    validate_design(design)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建，避免并发采集覆盖既有设计；此阶段不冻结设置或故障强度。
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return design


def load_design(path: Path) -> dict:
    try:
        design = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"冻结设计缺失或损坏：{path}") from exc
    return validate_design(design)


def build_dataset(runs: Path, output: Path, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，避免覆盖已有切分：{output}")
    design = load_design(runs / "design.json")
    assignments = design["assignments"]
    frozen_tasks = {task["task_id"]: task for task in design["tasks"]}
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
        if metadata.get("design_hash") != design["fingerprint"]:
            raise ValueError(f"Session 冻结设计 hash 不匹配：{directory.name}")
        task = frozen_tasks.get(metadata.get("task_id"))
        if task is None or fingerprint(metadata.get("task")) != fingerprint(task):
            raise ValueError(f"Session task 与冻结任务不一致：{directory.name}")
        if any(metadata.get(key) != task[key] for key in ("task_id", "template_id", "workflow")):
            raise ValueError(f"Session 任务标识与冻结任务不一致：{directory.name}")
        if metadata["phase"] == "pilot" and assignments[task["template_id"]] != "train":
            raise ValueError(f"pilot 仅允许训练模板：{directory.name}")
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
        "schema_version": 2, "seed": design["seed"], "assignments": assignments,
        "design": design, "design_hash": design["fingerprint"],
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
        "missing_templates": sorted(set(assignments) - set(session_df.template_id)),
        "missing_prefix_templates": sorted(set(assignments) - set(prefix_df.template_id)),
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
    design = validate_design(manifest.get("design"))
    if manifest.get("design_hash") != design["fingerprint"]:
        raise ValueError("数据集 design_hash 与冻结设计不一致。")
    if manifest.get("assignments") != design["assignments"]:
        raise ValueError("数据集分区与冻结设计不一致。")
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

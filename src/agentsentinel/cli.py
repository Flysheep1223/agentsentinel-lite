from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import ROOT, Settings, write_json
from .dataset import build_dataset, freeze_design, load_design, training_eda
from .environment import FAULT_KINDS
from .experiments import collect
from .modeling import evaluate_models, read_policies, train_models
from .tasks import Task, make_tasks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="AgentSentinel-Lite experiment CLI")
    root.add_argument("--debug", action="store_true", help="Display traceback on errors")
    sub = root.add_subparsers(dest="command", required=True)
    task_cmd = sub.add_parser("tasks", help="Export the 40-task / 10-template catalog only")
    task_cmd.add_argument("--output", type=Path, default=ROOT / "data/tasks.json")
    freeze_cmd = sub.add_parser("freeze", help="Freeze tasks and partitions before any collection")
    freeze_cmd.add_argument("--output", type=Path, required=True)
    freeze_cmd.add_argument("--seed", type=int, default=42)
    collect_cmd = sub.add_parser("collect", help="Collect baseline sessions using a frozen design")
    collect_cmd.add_argument("--design", type=Path, required=True)
    collect_cmd.add_argument("--output", type=Path, required=True)
    collect_cmd.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    collect_cmd.add_argument("--phase", choices=["demo", "pilot", "collection"], default="demo")
    collect_cmd.add_argument("--conditions", nargs="+", choices=FAULT_KINDS, default=FAULT_KINDS)
    collect_cmd.add_argument("--repeats", type=int, help="Default: pilot=1, demo/collection=3")
    collect_cmd.add_argument("--workers", type=int, default=4)
    collect_cmd.add_argument("--seed", type=int, default=42)
    collect_cmd.add_argument("--limit-tasks", type=int, help="Limit frozen tasks for demo only")
    collect_cmd.add_argument("--fault-probability", type=float, default=0.7)
    collect_cmd.add_argument("--persistent-probability", type=float, default=0.3)
    dataset_cmd = sub.add_parser(
        "dataset", help="Validate logs and derive prefixes using the pre-collection frozen design",
    )
    dataset_cmd.add_argument("--runs", type=Path, required=True)
    dataset_cmd.add_argument("--output", type=Path, required=True)
    dataset_cmd.add_argument("--seed", type=int, default=42, help="Compatibility only; design seed is used")
    eda_cmd = sub.add_parser("eda", help="Training-only data exploration")
    eda_cmd.add_argument("--dataset", type=Path, required=True)
    eda_cmd.add_argument("--output", type=Path, required=True)
    train_cmd = sub.add_parser("train", help="Train LR/KNN and freeze validation-selected policies")
    train_cmd.add_argument("--dataset", type=Path, required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--max-prefixes", type=int, default=12)
    train_cmd.add_argument("--ablation", choices=["none", "no_tokens", "no_errors"], default="none")
    eval_cmd = sub.add_parser("evaluate", help="Evaluate frozen policies on held-out templates")
    eval_cmd.add_argument("--dataset", type=Path, required=True)
    eval_cmd.add_argument("--models", type=Path, required=True)
    eval_cmd.add_argument("--output", type=Path, required=True)
    eval_cmd.add_argument("--bootstrap", type=int, default=200)
    online_cmd = sub.add_parser("online", help="Run 160 sessions on four preselected test instances")
    online_cmd.add_argument("--models", type=Path, required=True)
    online_cmd.add_argument("--output", type=Path, required=True)
    online_cmd.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    online_cmd.add_argument("--repeats", type=int, choices=[2], default=2)
    online_cmd.add_argument("--workers", type=int, default=4)
    online_cmd.add_argument("--seed", type=int, default=1729)
    online_cmd.add_argument("--budget", choices=[0.05, 0.1], type=float, default=0.05)
    demo_cmd = sub.add_parser("demo", help="Freeze a design and run the offline synthetic pipeline")
    demo_cmd.add_argument("--output", type=Path, default=ROOT / "data" / (
        "demo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    demo_cmd.add_argument("--repeats", type=int, default=3)
    demo_cmd.add_argument("--workers", type=int, default=4)
    demo_cmd.add_argument("--seed", type=int, default=42)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        settings = Settings.from_env()
        if args.command == "tasks":
            tasks = make_tasks()
            write_json(args.output, [task.model_dump() for task in tasks])
            print(f"Exported {len(tasks)} tasks to {args.output}")
        elif args.command == "freeze":
            design = freeze_design(args.output, args.seed)
            print(f"Frozen design: {args.output}; hash={design['fingerprint']}")
        elif args.command == "collect":
            if args.limit_tasks is not None:
                if args.phase != "demo":
                    raise ValueError("--limit-tasks 仅允许用于 demo，pilot/collection 必须遵守冻结协议。")
                if args.limit_tasks < 1:
                    raise ValueError("--limit-tasks 必须大于零。")
            design = load_design(args.design)
            tasks = [Task.model_validate(task) for task in design["tasks"]]
            if args.phase == "pilot":
                by_id = {task.task_id: task for task in tasks}
                tasks = [by_id[task_id] for task_id in design["pilot_task_ids"]]
            repeats = args.repeats
            if repeats is None:
                repeats = 1 if args.phase == "pilot" else 3
            result = collect(
                tasks[:args.limit_tasks], settings, args.output, args.backend, args.phase,
                args.conditions, repeats, args.workers, args.seed,
                args.fault_probability, args.persistent_probability, design=design,
            )
            if result["invalid_sessions"]:
                return 1
        elif args.command == "dataset":
            result = build_dataset(args.runs, args.output, args.seed)
            print(json.dumps(result["partitions"], ensure_ascii=False, indent=2))
        elif args.command == "eda":
            training_eda(args.dataset, args.output)
            print(f"Training EDA: {args.output}")
        elif args.command == "train":
            result = train_models(
                args.dataset, args.output, args.seed, args.max_prefixes, args.ablation,
            )
            print(f"Frozen policies: {args.output}; hash={result['freeze_hash'][:12]}")
        elif args.command == "evaluate":
            evaluate_models(args.dataset, args.models, args.output, args.bootstrap)
            print(f"Test report: {args.output / 'evaluation.json'}")
        elif args.command == "online":
            policies = read_policies(args.models)
            design = policies["design"]
            by_id = {task["task_id"]: task for task in design["tasks"]}
            tasks = [Task.model_validate(by_id[task_id]) for task_id in design["online_task_ids"]]
            result = collect(
                tasks, settings, args.output, args.backend, "online", policies["fault_conditions"],
                args.repeats, args.workers, args.seed, models=args.models, budget=args.budget,
                design=design,
            )
            if result["invalid_sessions"]:
                return 1
        elif args.command == "demo":
            if args.output.exists() and (
                not args.output.is_dir() or any(args.output.iterdir())
            ):
                raise ValueError(f"Demo 输出目录必须为空：{args.output}")
            design = freeze_design(args.output / "design.json", args.seed)
            tasks = [Task.model_validate(task) for task in design["tasks"]]
            print("Synthetic fixture only: results do not establish LLM warning performance.")
            collect(
                tasks, settings, args.output / "runs", "scripted", "demo", FAULT_KINDS,
                args.repeats, args.workers, args.seed, design=design,
            )
            build_dataset(args.output / "runs", args.output / "dataset", args.seed)
            training_eda(args.output / "dataset", args.output / "reports")
            train_models(args.output / "dataset", args.output / "models", args.seed)
            evaluate_models(
                args.output / "dataset", args.output / "models", args.output / "reports",
            )
            print(f"Demo finished: {args.output}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        if args.debug:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; incomplete sessions remain on disk for inspection.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

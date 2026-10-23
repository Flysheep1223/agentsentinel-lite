from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import ROOT, Settings, read_json, write_json
from .dataset import build_dataset, training_eda
from .environment import FAULT_KINDS
from .experiments import collect
from .modeling import evaluate_models, train_models
from .tasks import make_tasks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="AgentSentinel-Lite experiment CLI")
    root.add_argument("--debug", action="store_true", help="Display traceback on errors")
    sub = root.add_subparsers(dest="command", required=True)
    task_cmd = sub.add_parser("tasks", help="Export the 40-task / 20-template catalog")
    task_cmd.add_argument("--output", type=Path, default=ROOT / "data/tasks.json")
    collect_cmd = sub.add_parser("collect", help="Collect complete baseline sessions")
    collect_cmd.add_argument("--output", type=Path, required=True)
    collect_cmd.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    collect_cmd.add_argument("--phase", choices=["demo", "pilot", "collection"], default="demo")
    collect_cmd.add_argument("--conditions", nargs="+", choices=FAULT_KINDS, default=FAULT_KINDS)
    collect_cmd.add_argument("--repeats", type=int, default=3)
    collect_cmd.add_argument("--workers", type=int, default=4)
    collect_cmd.add_argument("--seed", type=int, default=42)
    collect_cmd.add_argument("--limit-tasks", type=int)
    collect_cmd.add_argument("--fault-probability", type=float, default=0.7)
    collect_cmd.add_argument("--persistent-probability", type=float, default=0.3)
    dataset_cmd = sub.add_parser("dataset", help="Validate logs, derive prefixes and freeze split")
    dataset_cmd.add_argument("--runs", type=Path, required=True)
    dataset_cmd.add_argument("--output", type=Path, required=True)
    dataset_cmd.add_argument("--seed", type=int, default=42)
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
    online_cmd = sub.add_parser("online", help="Run independent four-policy test comparisons")
    online_cmd.add_argument("--models", type=Path, required=True)
    online_cmd.add_argument("--output", type=Path, required=True)
    online_cmd.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    online_cmd.add_argument("--repeats", type=int, default=3)
    online_cmd.add_argument("--workers", type=int, default=4)
    online_cmd.add_argument("--seed", type=int, default=1729)
    online_cmd.add_argument("--budget", choices=[0.05, 0.1], type=float, default=0.05)
    demo_cmd = sub.add_parser("demo", help="Run the entire offline pipeline with synthetic fixtures")
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
        tasks = make_tasks()
        if args.command == "tasks":
            write_json(args.output, [task.model_dump() for task in tasks])
            print(f"Exported {len(tasks)} tasks to {args.output}")
        elif args.command == "collect":
            if args.limit_tasks is not None and args.limit_tasks < 1:
                raise ValueError("--limit-tasks 必须大于零。")
            result = collect(
                tasks[:args.limit_tasks], settings, args.output, args.backend, args.phase,
                args.conditions, args.repeats, args.workers, args.seed,
                args.fault_probability, args.persistent_probability,
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
            policies = read_json(args.models / "policies.json")
            tasks = [t for t in tasks if policies["template_assignments"][t.template_id] == "test"]
            result = collect(
                tasks, settings, args.output, args.backend, "online", FAULT_KINDS,
                args.repeats, args.workers, args.seed, models=args.models, budget=args.budget,
            )
            if result["invalid_sessions"]:
                return 1
        elif args.command == "demo":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError(f"Demo 输出目录必须为空：{args.output}")
            print("Synthetic fixture only: results do not establish LLM warning performance.")
            collect(
                tasks, settings, args.output / "runs", "scripted", "demo", FAULT_KINDS,
                args.repeats, args.workers, args.seed,
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

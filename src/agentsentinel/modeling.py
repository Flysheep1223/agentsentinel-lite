from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import fingerprint, read_json, write_json
from .dataset import load_dataset
from .evaluation import (
    bootstrap_intervals,
    choose_threshold,
    prefix_metrics,
    replay,
    summarize_replay,
)
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

RULE_FEATURES = ["step", "retry_count", "consecutive_errors", "total_tokens"]


def rule_scores(frame: pd.DataFrame, limits: dict) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    for feature, threshold in limits.items():
        if threshold is not None:
            result |= frame[feature].to_numpy() >= threshold
    return result.astype(float)


def preprocessing(columns: list[str]) -> ColumnTransformer:
    numeric = [name for name in NUMERIC_FEATURES if name in columns]
    categorical = [name for name in CATEGORICAL_FEATURES if name in columns]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def capped_training_rows(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap < 1:
        raise ValueError("max_prefixes must be positive")
    groups = []
    for _, group in frame.sort_values("step").groupby("session_id"):
        positions = np.unique(np.linspace(0, len(group) - 1, min(cap, len(group))).astype(int))
        groups.append(group.iloc[positions])
    return pd.concat(groups, ignore_index=True)


def score_latency(model: Pipeline, sample: pd.DataFrame) -> float:
    times = []
    for index in range(min(24, len(sample))):
        row = sample.iloc[[index]]
        started = perf_counter()
        model.predict_proba(row)
        times.append((perf_counter() - started) * 1000)
    return float(np.median(times))


def tune_rules(training: pd.DataFrame, validation: pd.DataFrame, sessions: pd.DataFrame,
               budget: float, monitor_price: float) -> dict:
    choices = []
    for feature in RULE_FEATURES:
        quantiles = training[feature].quantile([0.5, 0.8, 0.95]).to_numpy()
        choices.append([None] + sorted({float(q) for q in quantiles if q > 0}))
    best = None
    probe = validation.iloc[[0]]
    for values in itertools.product(*choices):
        limits = dict(zip(RULE_FEATURES, values))
        started = perf_counter()
        rule_scores(probe, limits)
        elapsed = (perf_counter() - started) * 1000
        scores = rule_scores(validation, limits)
        stats = summarize_replay(replay(validation, sessions, scores, 0.5, elapsed, monitor_price))
        if stats["false_abort_rate"] is None or stats["false_abort_rate"] > budget:
            continue
        priced = stats["execution_cost_usd"] is not None
        execution = stats["execution_cost_usd"] if priced else stats["tokens_used"]
        key = (execution, stats["false_abort_count"], -stats["failure_alarm_count"])
        if best is None or key < best[0]:
            best = (key, {
                "limits": limits, "threshold": 0.5, "budget": budget,
                "score_ms": elapsed, "validation": stats,
                "selection_unit": "USD" if priced else "tokens",
            })
    if best is None:
        raise ValueError("无法在验证集上选择规则阈值。")
    return best[1]


def train_models(
    dataset: Path, output: Path, seed: int = 42, max_prefixes: int = 12,
    ablation: str = "none",
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"模型目录非空，避免覆盖已冻结策略：{output}")
    prefixes, sessions, manifest = load_dataset(dataset)
    if manifest["phase"] not in {"demo", "collection"}:
        raise ValueError("试运行数据不能进入正式训练；仅接受 demo 或 collection 数据。")
    training = capped_training_rows(prefixes[prefixes.split == "train"], max_prefixes)
    validation = prefixes[prefixes.split == "validation"]
    validation_sessions = sessions[sessions.split == "validation"]
    if training.label.nunique() != 2 or not (validation_sessions.label == 0).any():
        raise ValueError("训练集需要成功和失败样本，验证集至少需要成功 Session。")
    columns = list(FEATURE_COLUMNS)
    if ablation == "no_tokens":
        columns = [c for c in columns if "token" not in c and c != "cost_usd"]
    elif ablation == "no_errors":
        columns = [c for c in columns if "error" not in c and c not in {"status", "retry_count"}]
    elif ablation != "none":
        raise ValueError("Unknown ablation")
    groups = training.template_id
    n_splits = min(3, groups.nunique())
    if n_splits < 2:
        raise ValueError("训练集需要至少两个独立模板。")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(training[columns], training.label, groups))
    if any(training.iloc[tr].label.nunique() != 2 or training.iloc[va].label.nunique() != 2
           for tr, va in folds):
        raise ValueError("分组交叉验证存在单一类别折；需要更多独立模板或重新设计试运行。")
    smallest_fold = min(len(tr) for tr, _ in folds)
    definitions = {
        "logistic": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [k for k in [3, 5, 9] if k <= smallest_fold],
             "model__weights": ["uniform", "distance"]},
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "dataset_fingerprint": manifest["fingerprint"],
        "phase": manifest["phase"], "backend": manifest["backend"],
        "config_hash": manifest["config_hash"], "seed": seed, "ablation": ablation,
        "columns": columns, "max_training_prefixes_per_session": max_prefixes,
        "training_rows": len(training), "training_sessions": training.session_id.nunique(),
        "training_templates": training.template_id.nunique(), "models": {},
        "monitor_price_per_second": manifest["monitor_price_per_second"],
        "template_assignments": manifest["assignments"],
        "fault_design": manifest["fault_design"],
        "fault_conditions": manifest["fault_conditions"],
    }
    for name, (estimator, grid) in definitions.items():
        pipeline = Pipeline([("preprocess", preprocessing(columns)), ("model", estimator)])
        search = GridSearchCV(
            pipeline, grid, cv=folds, scoring="average_precision",
            n_jobs=1, error_score="raise",
        )
        search.fit(training[columns], training.label)
        model = search.best_estimator_
        scores = model.predict_proba(validation[columns])[:, 1]
        latency = score_latency(model, validation[columns])
        policies = {
            str(budget): choose_threshold(
                validation, validation_sessions, scores, budget,
                latency, manifest["monitor_price_per_second"],
            )
            for budget in [0.05, 0.10]
        }
        joblib.dump(model, output / f"{name}.joblib")
        metadata["models"][name] = {
            "best_params": search.best_params_, "cv_average_precision": float(search.best_score_),
            "score_ms": latency, "policies": policies,
            "file_sha256": hashlib.sha256((output / f"{name}.joblib").read_bytes()).hexdigest(),
        }
        if name == "logistic":
            coefficients = dict(zip(
                model.named_steps["preprocess"].get_feature_names_out(),
                model.named_steps["model"].coef_[0].tolist(),
            ))
            write_json(output / "logistic_coefficients.json", coefficients)
    metadata["models"]["rules"] = {
        "policies": {
            str(budget): tune_rules(
                training, validation, validation_sessions, budget,
                manifest["monitor_price_per_second"],
            )
            for budget in [0.05, 0.10]
        },
    }
    metadata["freeze_hash"] = fingerprint(metadata)
    write_json(output / "policies.json", metadata)
    return metadata


def read_policies(models: Path) -> dict:
    policies = read_json(models / "policies.json")
    if fingerprint({k: v for k, v in policies.items() if k != "freeze_hash"}) != policies["freeze_hash"]:
        raise ValueError("停止策略配置已在冻结后改变。")
    for name in ("logistic", "knn"):
        if hashlib.sha256((models / f"{name}.joblib").read_bytes()).hexdigest() != (
            policies["models"][name]["file_sha256"]
        ):
            raise ValueError(f"模型已在冻结后改变：{name}")
    return policies


def evaluate_models(dataset: Path, models: Path, output: Path, bootstrap: int = 200) -> dict:
    prefixes, sessions, manifest = load_dataset(dataset)
    policies = read_policies(models)
    if policies["dataset_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("测试数据与冻结模型的数据清单不匹配。")
    prefixes = prefixes[prefixes.split == "test"]
    sessions = sessions[sessions.split == "test"]
    final_steps = sessions.set_index("session_id").total_steps
    stages = pd.cut(
        prefixes.step / prefixes.session_id.map(final_steps),
        bins=[0, 1 / 3, 2 / 3, 1], labels=["early", "middle", "late"],
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": manifest["phase"], "backend": manifest["backend"],
        "warning": "Synthetic engineering fixture; not research evidence."
        if manifest["backend"] == "scripted" else None,
        "freeze_hash": policies["freeze_hash"], "test_sessions": len(sessions), "methods": {},
    }
    for name, model_meta in policies["models"].items():
        model = joblib.load(models / f"{name}.joblib") if name != "rules" else None
        for budget, policy in model_meta["policies"].items():
            scores = (
                model.predict_proba(prefixes[policies["columns"]])[:, 1]
                if model is not None else rule_scores(prefixes, policy["limits"])
            )
            latency = model_meta.get("score_ms", policy.get("score_ms", 0))
            rows = replay(
                prefixes, sessions, scores, policy["threshold"], latency,
                policies["monitor_price_per_second"],
            )
            summary = summarize_replay(rows)
            key = f"{name}@{budget}"
            report["methods"][key] = {
                "prefix_metrics": prefix_metrics(prefixes.label, scores, policy["threshold"]),
                "session_metrics": summary,
                "bootstrap_95": bootstrap_intervals(rows, bootstrap),
                "by_fault": {
                    k: summarize_replay(g) for k, g in rows.groupby("fault_kind")
                },
                "by_workflow": {
                    k: summarize_replay(g) for k, g in rows.groupby("workflow")
                },
                "by_stage_prefix_metrics": {
                    stage: prefix_metrics(
                        prefixes.loc[stages == stage, "label"], scores[(stages == stage).to_numpy()],
                        policy["threshold"],
                    ) for stage in ("early", "middle", "late")
                },
                "cost_sensitivity_usd": [
                    {"miss_cost": miss, "abort_cost": abort,
                     "objective_usd": miss * int(rows.missed_failure.sum())
                     + abort * int(rows.false_abort.sum()) + summary["execution_cost_usd"]}
                    for miss, abort in [(0.01, 0.1), (0.1, 1), (1, 10)]
                ] if summary["execution_cost_usd"] is not None else None,
            }
            rows.to_csv(output / f"{key.replace('@', '_')}_sessions.csv", index=False)
            predictions = prefixes[["session_id", "template_id", "step", "label"]].copy()
            predictions["score"] = scores
            predictions.to_csv(output / f"{key.replace('@', '_')}_prefixes.csv", index=False)
    baseline = replay(prefixes, sessions, np.zeros(len(prefixes)), None)
    report["baseline"] = summarize_replay(baseline)
    write_json(output / "evaluation.json", report)
    _plot_report(report, output)
    return report


def _plot_report(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field in zip(axes, ["failure_recall", "false_abort_rate"]):
        ax.bar(names, [report["methods"][n]["session_metrics"][field] or 0 for n in names])
        ax.set_title(field)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Synthetic demo" if report["backend"] == "scripted" else "Held-out evaluation")
    fig.tight_layout()
    fig.savefig(output / "stopping_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, result in report["methods"].items():
        if not name.endswith("@0.05") or name.startswith("rules"):
            continue
        metrics = result["prefix_metrics"]
        axes[0].plot(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], label=name)
        axes[1].plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=name)
        axes[2].plot(
            metrics["calibration"]["predicted"], metrics["calibration"]["observed"],
            marker="o", label=name,
        )
    for ax, title, xlabel, ylabel in zip(
        axes, ["Precision-recall", "ROC", "Calibration"],
        ["Recall", "False positive rate", "Predicted risk"],
        ["Precision", "True positive rate", "Observed failure frequency"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "prediction_curves.png", dpi=160)
    plt.close(fig)


def load_observer(models: Path, method: str, budget: float):
    policies = read_policies(models)
    policy = policies["models"][method]["policies"][str(budget)]
    model = joblib.load(models / f"{method}.joblib") if method != "rules" else None

    def observe(features: dict) -> dict:
        frame = pd.DataFrame([{
            key: np.nan if value is None else value for key, value in features.items()
        }])
        score = float(
            model.predict_proba(frame[policies["columns"]])[0, 1] if model is not None
            else rule_scores(frame, policy["limits"])[0]
        )
        threshold = policy["threshold"]
        return {"score": score, "threshold": threshold,
                "stop": threshold is not None and score >= threshold}

    return observe

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def prefix_metrics(labels, scores, threshold: float | None = 0.5) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels):
        return {"count": 0}
    predictions = scores >= (float("inf") if threshold is None else threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0,
    )
    two_classes = len(np.unique(labels)) == 2
    observed, predicted = calibration_curve(labels, scores, n_bins=8, strategy="quantile")
    curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
    false_positive, true_positive, _ = roc_curve(labels, scores) if two_classes else ([], [], [])
    return {
        "count": len(labels), "failure_prevalence": float(labels.mean()),
        "pr_auc_average_precision": float(average_precision_score(labels, scores))
        if two_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration": {"predicted": predicted.tolist(), "observed": observed.tolist()},
        "pr_curve": {"precision": curve_precision.tolist(), "recall": curve_recall.tolist()},
        "roc_curve": {"fpr": list(false_positive), "tpr": list(true_positive)},
    }


def replay(
    prefixes: pd.DataFrame, sessions: pd.DataFrame, scores,
    threshold: float | None, score_ms: float = 0, monitor_price: float = 0,
) -> pd.DataFrame:
    scored = prefixes.copy()
    scored["_score"] = np.asarray(scores)
    scored = scored.sort_values(["session_id", "step"])
    groups = {sid: group for sid, group in scored.groupby("session_id")}
    rows = []
    for session in sessions.to_dict("records"):
        group = groups.get(session["session_id"])
        alarm = None
        checks = 0
        if group is not None:
            checks = len(group)
            if threshold is not None:
                hits = np.flatnonzero(group["_score"].to_numpy() >= threshold)
                if len(hits):
                    checks = int(hits[0]) + 1
                    alarm = group.iloc[int(hits[0])]
        aborted = alarm is not None
        failed = bool(session["label"])
        used_tokens = float(alarm.total_tokens) if aborted else session["final_tokens"]
        used_cost = float(alarm.cost_usd) if aborted else session["final_cost_usd"]
        used_ms = float(alarm.elapsed_ms) if aborted else session["final_elapsed_ms"]
        scoring_total_ms = checks * score_ms
        rows.append({
            **session, "aborted": int(aborted), "false_abort": int(aborted and not failed),
            "failure_alarm": int(aborted and failed), "missed_failure": int(not aborted and failed),
            "alarm_step": int(alarm.step) if aborted else None,
            "used_tokens": used_tokens,
            "used_cost_usd": used_cost + scoring_total_ms / 1000 * monitor_price,
            "used_elapsed_ms": used_ms + scoring_total_ms,
            "monitor_ms": scoring_total_ms,
            "saved_tokens": session["final_tokens"] - used_tokens,
            "lead_steps": session["total_steps"] - int(alarm.step) if aborted and failed else None,
            "lead_tokens": session["final_tokens"] - used_tokens if aborted and failed else None,
            "lead_ms": session["final_elapsed_ms"] - used_ms if aborted and failed else None,
            "before_violation": int(
                aborted and pd.notna(session["first_violation_step"])
                and alarm.step < session["first_violation_step"]
            ),
        })
    return pd.DataFrame(rows)


def summarize_replay(rows: pd.DataFrame) -> dict:
    success = int((rows.label == 0).sum())
    failures = int(rows.label.sum())
    violations = int(rows.first_violation_step.notna().sum())
    priced = rows.used_cost_usd.notna().all()
    surviving_successes = success - int(rows.false_abort.sum())

    def optional_mean(column: str) -> float | None:
        return float(rows[column].mean()) if rows[column].notna().any() else None

    return {
        "sessions": len(rows), "successes_in_baseline": success, "failures_in_baseline": failures,
        "false_abort_count": int(rows.false_abort.sum()),
        "false_abort_rate": float(rows.false_abort.sum() / success) if success else None,
        "failure_alarm_count": int(rows.failure_alarm.sum()),
        "failure_recall": float(rows.failure_alarm.sum() / failures) if failures else None,
        "stop_rate": float(rows.aborted.mean()),
        "tokens_used": float(rows.used_tokens.sum()), "tokens_saved": float(rows.saved_tokens.sum()),
        "execution_cost_usd": float(rows.used_cost_usd.sum()) if priced else None,
        "cost_per_retained_success_usd": float(rows.used_cost_usd.sum() / surviving_successes)
        if priced and surviving_successes else None,
        "elapsed_ms": float(rows.used_elapsed_ms.sum()),
        "monitor_ms": float(rows.monitor_ms.sum()),
        "mean_lead_steps": optional_mean("lead_steps"),
        "mean_lead_tokens": optional_mean("lead_tokens"),
        "mean_lead_ms": optional_mean("lead_ms"),
        "violation_sessions": violations,
        "before_violation_rate": float(rows.before_violation.sum() / violations)
        if violations else None,
    }


def choose_threshold(prefixes, sessions, scores, budget, score_ms=0, monitor_price=0) -> dict:
    if not (sessions.label == 0).any():
        raise ValueError("验证集没有成功 Session，无法选择错误中断阈值。")
    # Include every distinct score plus the explicit never-stop policy.
    candidates = [None] + sorted(set(float(p) for p in scores), reverse=True)
    thresholds = np.array([np.inf if t is None else t for t in candidates])
    # Compare all thresholds within a session at once; don't rebuild frames per threshold.
    scored = prefixes.assign(_score=np.asarray(scores)).sort_values(["session_id", "step"])
    groups = {sid: group for sid, group in scored.groupby("session_id")}
    false_aborts = np.zeros(len(candidates), dtype=int)
    failure_alarms = np.zeros(len(candidates), dtype=int)
    execution = np.zeros(len(candidates))
    priced = sessions.final_cost_usd.notna().all() and prefixes.cost_usd.notna().all()
    for session in sessions.to_dict("records"):
        group = groups.get(session["session_id"])
        baseline = session["final_cost_usd"] if priced else session["final_tokens"]
        if group is None:
            execution += baseline
            continue
        hits = group["_score"].to_numpy()[:, None] >= thresholds[None, :]
        any_hit = hits.any(axis=0)
        first = hits.argmax(axis=0)
        values = group["cost_usd" if priced else "total_tokens"].to_numpy()
        execution += np.where(any_hit, values[first], baseline)
        if priced:
            checks = np.where(any_hit, first + 1, len(group))
            execution += checks * score_ms / 1000 * monitor_price
        if session["label"]:
            failure_alarms += any_hit
        else:
            false_aborts += any_hit
    successes = int((sessions.label == 0).sum())
    feasible = np.flatnonzero(false_aborts / successes <= budget)
    index = min(feasible, key=lambda i: (execution[i], false_aborts[i], -failure_alarms[i]))
    threshold = candidates[index]
    return {
        "threshold": threshold, "budget": budget,
        "validation": summarize_replay(replay(
            prefixes, sessions, scores, threshold, score_ms, monitor_price,
        )),
        "selection_unit": "USD" if priced else "tokens",
    }


def bootstrap_intervals(rows: pd.DataFrame, repetitions: int = 200, seed: int = 42) -> dict:
    """Resample templates, preserving each selected template's complete sessions."""
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    templates = rows.template_id.unique()
    grouped = {key: rows[rows.template_id == key] for key in templates}
    samples = {"false_abort_rate": [], "failure_recall": [], "tokens_saved": []}
    for _ in range(repetitions):
        sampled = pd.concat([grouped[t] for t in rng.choice(templates, len(templates), replace=True)])
        summary = summarize_replay(sampled)
        for key in samples:
            if summary[key] is not None:
                samples[key].append(summary[key])
    return {
        key: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        } if values else None
        for key, values in samples.items()
    }

"""Report-stage, train-only EDA; never mutate frozen datasets or model artifacts.

Run with .venv/bin/python scripts/supplement_training_eda.py --output <new-directory>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {"incident": "服务事件", "ticket": "工单"}
CONDITIONS = {
    "none": "正常",
    "partial_response": "部分响应",
    "rate_limit": "限流",
    "schema_change": "Schema变化",
    "timeout": "超时",
}
RESOURCE_COLUMNS = [
    "tool_calls", "model_calls", "input_tokens", "output_tokens", "total_tokens",
    "reasoning_tokens", "cached_tokens", "elapsed_seconds", "cost_estimate",
    "explicit_error_count", "immediate_exact_retry_count",
]
CORR_LABELS = {
    "error_count": "累计错误",
    "retry_count": "累计重试",
    "schema_error_count": "Schema错误",
    "input_tokens": "输入Token",
    "output_tokens": "输出Token",
    "reasoning_tokens": "推理Token",
    "cached_tokens": "缓存Token",
    "total_tokens": "总Token",
    "cost_estimate": "累计费用",
    "elapsed_ms": "累计时间",
    "recent_latency_ms": "最近延迟",
    "recent_repeat_ratio": "最近重复比",
}
READ_TOOLS = {"get_ticket", "get_service_status", "lookup_policy"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON type: {type(item).__name__}")

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar) + "\n",
        encoding="utf-8",
    )


def train_csv(path):
    """Filter each chunk immediately; never summarize validation/test observations."""
    return pd.concat(
        [part.loc[part["split"].eq("train")].copy()
         for part in pd.read_csv(path, chunksize=10000)],
        ignore_index=True,
    )


def groups(frame, dimensions):
    if not dimensions:
        yield {}, frame
        return
    for key, group in frame.groupby(dimensions, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        yield dict(zip(dimensions, values)), group


def descriptive_tables(sessions):
    moments, counts, outliers = [], [], []
    dimensions = [
        [], ["label"], ["workflow"], ["condition"],
        ["workflow", "label"], ["condition", "label"],
        ["workflow", "condition"], ["workflow", "condition", "label"],
    ]
    for dims in dimensions:
        for keys, group in groups(sessions, dims):
            identity = {"grouping": "+".join(dims) or "all", **keys}
            counts.append({
                **identity, "sessions": len(group), "failures": int(group.label.sum()),
                "failure_rate": float(group.label.mean()),
            })
            for metric in RESOURCE_COLUMNS:
                series = group[metric].dropna()
                q1, median, q3 = series.quantile([.25, .5, .75])
                moments.append({
                    **identity, "metric": metric, "sessions": len(group),
                    "nonmissing": len(series), "mean": series.mean(),
                    "std_sample": series.std(ddof=1), "min": series.min(),
                    "q1": q1, "median": median, "q3": q3, "max": series.max(),
                })
                if dims == ["workflow", "label"] and metric in (
                    "total_tokens", "elapsed_seconds", "tool_calls",
                ):
                    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                    outliers.append({
                        **identity, "metric": metric, "sessions": len(group),
                        "lower_fence": lower, "upper_fence": upper,
                        "below_fence": int((series < lower).sum()),
                        "above_fence": int((series > upper).sum()),
                        "iqr_zero": bool(q1 == q3),
                    })
    return pd.DataFrame(moments), pd.DataFrame(counts), pd.DataFrame(outliers)


def quality_tables(sessions, prefixes, events, attempts, feature_columns):
    rows = []
    specifications = [
        ("valid_sessions", sessions, RESOURCE_COLUMNS),
        ("eligible_prefixes", prefixes, feature_columns),
        ("all_train_attempts", attempts,
         ["input_tokens", "output_tokens", "reasoning_tokens", "cost_estimate"]),
    ]
    for name, frame, columns in specifications:
        for column in columns:
            series = frame[column]
            numeric = pd.api.types.is_numeric_dtype(series)
            rows.append({
                "table": name, "column": column, "rows": len(series),
                "missing": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "nonfinite_nonmissing": (
                    int((~np.isfinite(series.dropna())).sum()) if numeric else 0
                ),
                "negative": int((series.dropna() < 0).sum()) if numeric else 0,
            })
    table = pd.DataFrame(rows)
    require(not table.nonfinite_nonmissing.any(), "Nonfinite numeric input")
    require(not table.negative.any(), "Negative numeric input")
    duplicates = {
        "valid_session_id": int(sessions.duplicated(["session_id"]).sum()),
        "train_attempt_condition_id": int(attempts.duplicated(["condition_id"]).sum()),
        "eligible_prefix_session_step": int(prefixes.duplicated(["session_id", "step"]).sum()),
        "tool_event_session_step": int(events.duplicated(["session_id", "step"]).sum()),
        "exact_diagnostic_event_rows": int(events.duplicated().sum()),
    }
    require(not any(duplicates.values()), "Duplicate table observation")
    return table, duplicates


def recovery_tables(events, sessions):
    joined = events.merge(
        sessions[["session_id", "workflow", "condition", "label"]],
        on="session_id", validate="many_to_one",
    )
    rows = []
    masks = {
        "explicit_read_error": (
            joined.error & joined.tool.isin(READ_TOOLS), "later_ok_step"
        ),
        "explicit_policy_error": (
            joined.error & joined.tool.eq("lookup_policy"), "later_ok_step"
        ),
        "incomplete_policy_response": (
            joined.incomplete_policy, "later_complete_policy_step"
        ),
    }
    for kind, (mask, recovered_column) in masks.items():
        for dims in ([], ["workflow"], ["condition"], ["workflow", "condition"]):
            for keys, subset in groups(joined.loc[mask], dims):
                recovered = subset[recovered_column].notna()
                delay = subset.loc[recovered, recovered_column] - subset.loc[recovered, "step"]
                require((delay > 0).all(), "Recovery must happen after the event")
                affected = subset.drop_duplicates("session_id")
                rows.append({
                    "kind": kind, "grouping": "+".join(dims) or "all", **keys,
                    "events": len(subset), "recovered_events": int(recovered.sum()),
                    "event_recovery_rate": float(recovered.mean()),
                    "affected_sessions": len(affected),
                    "successful_sessions": int(affected.label.eq(0).sum()),
                    "session_success_rate": float(affected.label.eq(0).mean()),
                    "median_event_step": float(subset.step.median()),
                    "recovered_delay_q1": delay.quantile(.25),
                    "recovered_delay_median": delay.median(),
                    "recovered_delay_q3": delay.quantile(.75),
                })
    return pd.DataFrame(rows)


def inspect_extremes(sessions, source_root, source_hashes, checked):
    selections = {}
    for keys, group in groups(sessions, ["workflow", "label"]):
        for metric in ("total_tokens", "elapsed_seconds"):
            row = group.loc[group[metric].idxmax()]
            selections.setdefault(row.session_id, []).append({**keys, "maximum_of": metric})
    results = []
    for sid, selected_for in selections.items():
        raw = {}
        for filename in ("session.json", "events.jsonl", "trajectory.json", "result.json"):
            rel = f"collection/{sid}/{filename}"
            path = source_root / rel
            digest = sha256(path)
            require(digest == source_hashes[rel], f"Raw hash mismatch: {rel}")
            checked[path] = digest
            raw[filename] = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if filename.endswith("jsonl") else json.loads(path.read_text())
            )
        result = raw["result.json"]
        events = raw["events.jsonl"]
        tools = [event for event in events if event["kind"] == "tool_end"]
        requests = [event for event in events if event["kind"] == "llm_end"]
        elapsed = np.array([event["elapsed_ms"] for event in events])
        require(np.isfinite(elapsed).all() and (elapsed >= 0).all()
                and (np.diff(elapsed) >= 0).all(), f"Invalid event times: {sid}")
        require([event["step"] for event in tools] == list(range(1, len(tools) + 1)),
                f"Tool step sequence mismatch: {sid}")
        require([event["model_call"] for event in requests] == list(range(1, len(requests) + 1)),
                f"Request sequence mismatch: {sid}")
        require(datetime.fromisoformat(raw["session.json"]["created_at"]).tzinfo is not None,
                f"Missing timezone: {sid}")
        row = sessions.loc[sessions.session_id.eq(sid)].iloc[0]
        for metric in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
            usage = sum(event["usage"][metric] for event in requests)
            require(usage == row[metric] == result["totals"][metric],
                    f"Usage mismatch: {sid}/{metric}")
        require(row.total_tokens == row.input_tokens + row.output_tokens,
                f"Total usage mismatch: {sid}")
        require(np.isclose(row.elapsed_ms, result["elapsed_ms"], rtol=0, atol=1e-6),
                f"Duration mismatch: {sid}")
        require(len(raw["trajectory.json"]["tools"]) == row.tool_calls == len(tools),
                f"Trajectory length mismatch: {sid}")
        require(events[-2]["elapsed_ms"] <= result["elapsed_ms"] <= events[-1]["elapsed_ms"],
                f"Final time outside raw event boundary: {sid}")
        results.append({
            "session_id": sid, "selected_for": selected_for,
            "template_id": row.template_id, "workflow": row.workflow, "condition": row.condition,
            "label": int(row.label), "termination": row.termination,
            "total_tokens": int(row.total_tokens), "reasoning_tokens": int(row.reasoning_tokens),
            "elapsed_seconds": float(row.elapsed_seconds), "model_calls": len(requests),
            "tool_calls": len(tools),
            "summed_model_latency_seconds": sum(r["latency_ms"] for r in requests) / 1000,
            "max_model_latency_seconds": max(r["latency_ms"] for r in requests) / 1000,
            "chronology_and_usage_checks": "passed",
            "trace": [{"step": event["step"], "tool": event["tool"], "status": event["status"]}
                      for event in tools],
        })
    return results


def style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC")
                 if name in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font, "font.size": 11, "axes.titlesize": 14,
        "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    return font


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=200, bbox_inches="tight",
                    metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)


def condition_figure(sessions, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (workflow, title) in zip(axes, WORKFLOWS.items()):
        group = sessions.loc[sessions.workflow.eq(workflow)]
        stats = group.groupby("condition").label.agg(["sum", "count", "mean"]).loc[list(CONDITIONS)]
        bars = ax.bar(range(5), stats["mean"], color="#527eae", width=.62)
        for bar, (_, row) in zip(bars, stats.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .015,
                    f'{row["mean"]:.1%}\n{int(row["sum"])}/{int(row["count"])}',
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(5), list(CONDITIONS.values()))
        ax.set_title(f"{title}（n={len(group):,}）")
        ax.set_ylim(0, .65)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("任务失败率")
    fig.suptitle("训练集：分配运行条件下的任务失败率", fontsize=17)
    fig.text(.5, .025, "记录单位：完整任务；柱上为失败率及失败数/有效任务数。"
             "条件名称不表示每次运行都触发故障。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .94))
    save_figure(fig, directory, "05_condition_failure")


def resource_figures(sessions, directory):
    for metric, title, unit, name in (
        ("total_tokens", "完整任务Token消耗", "总Token", "06_token_boxplot"),
        ("elapsed_seconds", "完整任务墙钟时间", "秒", "07_time_boxplot"),
        ("tool_calls", "完整任务工具调用步数", "工具调用次数", "08_steps_boxplot"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        for ax, (workflow, workflow_title) in zip(axes, WORKFLOWS.items()):
            values = [
                sessions.loc[sessions.workflow.eq(workflow) & sessions.label.eq(label), metric]
                for label in (0, 1)
            ]
            box = ax.boxplot(
                values, tick_labels=[f"成功\nn={len(values[0]):,}", f"失败\nn={len(values[1]):,}"],
                patch_artist=True, widths=.5, whis=1.5,
                medianprops={"color": "#182631", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 3, "alpha": .5},
            )
            for patch, color in zip(box["boxes"], ("#89b6ad", "#dfa08a")):
                patch.set_facecolor(color)
            ax.set_title(workflow_title)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            if metric == "tool_calls":
                ax.set_yticks(range(0, 13, 2))
                ax.set_ylim(0, 13)
        axes[0].set_ylabel(unit)
        fig.suptitle(f"训练集：{title}，按工作流及最终结果比较", fontsize=17)
        fig.text(.5, .025, "记录单位：完整任务。箱体为Q1–Q3，中线为中位数，"
                 "须线延伸至1.5×IQR内最远观测；外点保留。", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .08, 1, .94))
        save_figure(fig, directory, name)


def correlation_tables_and_figure(step3, numeric_columns, output):
    constants, pairs = [], []
    for name, group in [("all", step3), *list(step3.groupby("workflow"))]:
        corr = group[numeric_columns].corr(method="spearman")
        corr.to_csv(output / f"step3_spearman_{name}.csv", index_label="feature")
        nonmissing = group[numeric_columns].notna().astype(int)
        (nonmissing.T @ nonmissing).to_csv(output / f"step3_pair_n_{name}.csv", index_label="feature")
        for column in numeric_columns:
            if group[column].nunique(dropna=True) <= 1:
                constants.append({"workflow": name, "feature": column, "reason": "constant_or_empty"})
        for i, first in enumerate(numeric_columns):
            for second in numeric_columns[i + 1:]:
                if pd.notna(corr.loc[first, second]):
                    pairs.append({
                        "workflow": name, "first": first, "second": second,
                        "rho": corr.loc[first, second],
                        "pair_n": int(group[[first, second]].notna().all(axis=1).sum()),
                    })
    pd.DataFrame(constants).to_csv(output / "step3_undefined_correlations.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "step3_correlation_pairs.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 21))
    columns = list(CORR_LABELS)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")
    for ax, (workflow, title) in zip(axes, WORKFLOWS.items()):
        group = step3.loc[step3.workflow.eq(workflow)]
        corr = group[columns].corr(method="spearman")
        image = ax.imshow(np.ma.masked_invalid(corr.to_numpy()), vmin=-1, vmax=1, cmap=cmap)
        labels = [CORR_LABELS[column] for column in columns]
        ax.set_xticks(range(len(columns)), labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(columns)), labels, fontsize=11)
        for y in range(len(columns)):
            for x in range(len(columns)):
                value = corr.iloc[y, x]
                ax.text(x, y, "—" if pd.isna(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if pd.notna(value) and abs(value) > .65 else "#222222")
        ax.set_title(f"{title}：第3步，每个任务一行（n={len(group):,}）", pad=14)
        fig.colorbar(image, ax=ax, fraction=.032, pad=.025, label="Spearman ρ")
    fig.suptitle("训练集固定第3步：错误、重复与资源特征的相关性", fontsize=18, y=.995)
    fig.text(.5, .008, "图示12项数值特征；不纳入标签或事后诊断。固定步数/工具次数为常量，"
             "相关系数未定义；其余常量单元格为灰色。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .98), h_pad=3)
    save_figure(fig, output / "figures", "09_step3_spearman")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/expanded_dataset_v1")
    parser.add_argument("--eda", type=Path, default=ROOT / "artifacts/expanded_eda_v2")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory; frozen artifacts are never overwritten")
    checked = {}
    manifests = {}
    for name, root, files in (
        ("dataset", args.dataset, ["all_sessions.csv", "prefixes.csv", "source_hashes.json"]),
        ("eda", args.eda, ["diagnostic_sessions.csv", "diagnostic_events.csv"]),
    ):
        path = root / "manifest.json"
        checked[path] = sha256(path)
        manifests[name] = json.loads(path.read_text())
        for filename in files:
            path = root / filename
            digest = sha256(path)
            require(digest == manifests[name]["files"][filename], f"Input hash mismatch: {path}")
            checked[path] = digest
    require(manifests["dataset"]["fingerprint"] == manifests["eda"]["dataset_fingerprint"],
            "EDA and dataset fingerprints differ")
    sessions = pd.read_csv(args.eda / "diagnostic_sessions.csv")
    require(sessions.split.eq("train").all() and sessions.valid_for_analysis.all(),
            "Diagnostic sessions must be valid training observations only")
    events = pd.read_csv(args.eda / "diagnostic_events.csv")
    prefixes = train_csv(args.dataset / "prefixes.csv")
    attempts = train_csv(args.dataset / "all_sessions.csv")
    require(len(sessions) == 3542 and sessions.label.sum() == 972, "Unexpected training population")
    require(len(prefixes) == 27367 and len(attempts) == 3600, "Unexpected training row count")
    require(set(events.session_id) == set(prefixes.session_id) == set(sessions.session_id),
            "Training observation IDs differ")
    require(len(events) == sessions.tool_calls.sum(), "Tool event count mismatch")
    require(len(events.loc[events.decision_eligible]) == len(prefixes), "Eligibility count mismatch")
    sessions["elapsed_seconds"] = sessions.elapsed_ms / 1000
    require(set(sessions.condition) == set(CONDITIONS), "Unknown condition")
    feature_columns = manifests["dataset"]["feature_columns"]
    numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(prefixes[column])]
    step3 = prefixes.loc[prefixes.step.eq(3)].copy()
    require(len(step3) == len(sessions) and step3.session_id.is_unique, "Step-3 population mismatch")
    quality, duplicates = quality_tables(sessions, prefixes, events, attempts, feature_columns)
    moments, counts, outliers = descriptive_tables(sessions)
    recovery = recovery_tables(events, sessions)
    source_root = args.source or Path(manifests["dataset"]["source_root"])
    extremes = inspect_extremes(
        sessions, source_root, json.loads((args.dataset / "source_hashes.json").read_text()), checked,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    for filename, table in (
        ("grouped_descriptive.csv", moments), ("grouped_counts.csv", counts),
        ("missing_and_numeric_quality.csv", quality), ("tukey_outliers.csv", outliers),
        ("recovery_comparison.csv", recovery),
    ):
        table.to_csv(output / filename, index=False)
    write_json(output / "extreme_trajectory_checks.json", extremes)
    font = style()
    condition_figure(sessions, output / "figures")
    resource_figures(sessions, output / "figures")
    correlation_tables_and_figure(step3, numeric, output)
    require(all(sha256(path) == value for path, value in checked.items()), "Inputs changed during EDA")
    summary = {
        "scope": "train only; supplementary EDA at report stage, not pre-model discovery",
        "sessions": len(sessions), "failures": int(sessions.label.sum()),
        "prefixes": len(prefixes), "tool_events": len(events), "attempts": len(attempts),
        "duplicates": duplicates, "extreme_sessions_checked": len(extremes),
        "raw_files_checked_before_and_after": len(extremes) * 4,
        "inputs_unchanged": True, "font": font,
        "numeric_feature_columns": numeric,
        "existing_audit": {
            "scope": "All 6000 collected tasks; inherited evidence, not rerun by this supplement",
            "integrity_issues": manifests["dataset"]["quality"]["integrity_issues"],
            "checks": "Created-at/batch intervals; finite nonnegative nondecreasing elapsed time; "
                      "request/tool event sequence; raw usage and state replay.",
        },
        "conventions": [
            "Sample standard deviation (ddof=1); pandas linear quartile interpolation.",
            "Tukey fences are descriptive; no observations dropped, winsorized, or relabeled.",
            "Missing usage and undefined correlations remain missing, never replaced with zero.",
            "Recovery denominators are events; task success denominators are unique affected sessions.",
            "Explicit policy errors are a subset of read errors; categories and tasks may overlap.",
            "Recovery is limited by each original trajectory; later events have less follow-up.",
            "Error recovery=next same tool/arguments with ok; incomplete recovery=both action/owner present.",
            "No model, split, threshold, source file, or existing artifact was modified.",
        ],
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "format": "expanded_eda_report_supplement_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "scope": "train",
        "dataset_fingerprint": manifests["dataset"]["fingerprint"],
        "eda_fingerprint": manifests["eda"]["fingerprint"],
        "script_sha256": sha256(__file__),
        "inputs_verified_before_and_after": {str(path.resolve()): digest for path, digest in checked.items()},
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "files": {str(path.relative_to(output)): sha256(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Offline collection EDA. Run with .venv/bin/python artifacts/eda_v1/reproduce.py."""
from pathlib import Path
import sys
import json
import hashlib
import math
from collections import Counter
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentsentinel.dataset import load_design
from agentsentinel.features import FeatureBuilder, NUMERIC_FEATURES, signature
from agentsentinel.config import fingerprint

SOURCE = ROOT / "data/authorized_20261025_30plus600_v1/collection"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent
if OUT.exists() and any(p.name != "reproduce.py" for p in OUT.iterdir()):
    raise SystemExit(f"Refusing to overwrite existing artifacts: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=False)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def csv(name, frame):
    frame.to_csv(OUT / name, index=False)

def save_fig(name, fig):
    fig.tight_layout()
    fig.savefig(OUT / "figures" / name, dpi=160, bbox_inches="tight")
    plt.close(fig)

def md(frame):
    # No optional tabulate dependency.
    def fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))

design = load_design(SOURCE / "design.json")
tasks = {t["task_id"]: t for t in design["tasks"]}
assignments = design["assignments"]
sessions, prefixes, exclusions, audits, hashes = [], [], [], [], {}
conditions, session_ids = set(), set()

def numeric_issues(obj):
    if isinstance(obj, dict):
        return sum(numeric_issues(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(numeric_issues(v) for v in obj)
    return int(isinstance(obj, (int, float)) and not isinstance(obj, bool) and not math.isfinite(obj))

for directory in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
    paths = [directory / n for n in ("session.json", "result.json", "trajectory.json", "events.jsonl")]
    assert all(p.is_file() for p in paths), directory.name
    raw = [p.read_bytes() for p in paths]
    for p, b in zip(paths, raw):
        hashes[str(p.relative_to(SOURCE))] = hashlib.sha256(b).hexdigest()
    meta, result, trajectory = [json.loads(b) for b in raw[:3]]
    events = [json.loads(line) for line in raw[3].splitlines() if line.strip()]
    sid = meta["session_id"]
    assert sid == directory.name == result["session_id"] and sid not in session_ids
    session_ids.add(sid)
    assert meta["phase"] == "collection" and meta["policy"] == "none"
    assert meta["design_hash"] == design["fingerprint"]
    assert fingerprint(meta["task"]) == fingerprint(tasks[meta["task_id"]])
    assert all(meta[k] == tasks[meta["task_id"]][k] for k in ("template_id", "workflow"))
    assert meta["settings"]["currency"] == result["totals"]["currency"] == "CNY"
    condition = meta["condition_id"]
    assert condition not in conditions
    conditions.add(condition)
    split = assignments[meta["template_id"]]
    tools = [e for e in events if e["kind"] == "tool_end"]
    llms = [e for e in events if e["kind"] == "llm_end"]
    duplicate_events = len(events) - len({json.dumps(e, sort_keys=True) for e in events})
    elapsed = [e.get("elapsed_ms") for e in events]
    missing_time = sum(t is None for t in elapsed)
    bad_time = sum(t is not None and (not math.isfinite(t) or t < 0) for t in elapsed)
    reversals = sum(b < a for a, b in zip(elapsed, elapsed[1:]) if a is not None and b is not None)
    created = datetime.fromisoformat(meta["created_at"])
    assert created.tzinfo is not None
    nonfinite = sum(numeric_issues(o) for o in (meta, result, trajectory, events))
    assert events[0]["kind"] == "session_start" and events[-1]["kind"] == "session_end"
    assert [e["step"] for e in tools] == list(range(1, len(tools) + 1))
    assert len(tools) == result["totals"]["tool_calls"]
    assert all(e["latency_ms"] >= 0 for e in tools)
    assert not (duplicate_events or missing_time or bad_time or reversals or nonfinite), sid
    expected_eligible = [e["step"] < meta["settings"]["max_steps"] and e["totals"]["model_calls"] < meta["settings"]["max_model_calls"] and e["totals"]["total_tokens"] < meta["settings"]["max_tokens"] and e["elapsed_ms"] < meta["settings"]["session_timeout"] * 1000 for e in tools]
    assert expected_eligible == [e["decision_eligible"] for e in tools]
    assert len({e["step"] for e in tools}) == len(tools)
    actual_repeats = len(tools) - len({signature(e) for e in tools})
    audits.append({"session_id": sid, "split": split, "valid": result["valid_for_analysis"], "events": len(events), "duplicate_events": duplicate_events, "missing_elapsed": missing_time, "invalid_elapsed": bad_time, "elapsed_reversals": reversals, "nonfinite_numbers": nonfinite, "actual_repeated_calls_retained": actual_repeats, "unknown_usage_requests": result["totals"]["unknown_usage_requests"], "tool_calls": len(tools), "eligible_prefixes": sum(expected_eligible)})
    if not result["valid_for_analysis"]:
        exclusions.append({"session_id": sid, "split": split, "template_id": meta["template_id"], "termination": result["termination"], "reason": "基础设施异常或无法完整核算 usage；不把未知结果作为任务失败", "unknown_usage_requests": result["totals"]["unknown_usage_requests"]})
        continue
    assert result["label"] in (0, 1) and result["label"] == int(not result["success"])
    common = {k: meta[k] for k in ("session_id", "task_id", "template_id", "workflow")}
    common.update(split=split, label=result["label"], fault_kind=meta["fault_config"]["kind"])
    builder = FeatureBuilder(meta["workflow"])
    last = None
    for e in tools:
        last = builder.add(e)
        if e["decision_eligible"]:
            row = {**common, **last, "cost_cny": e["totals"]["cost_cny"], "currency": "CNY", "reasoning_tokens_missing": int(last["reasoning_tokens"] is None), "decision_eligible": True}
            prefixes.append(row)
    totals = result["totals"]
    assert totals["total_tokens"] == totals["input_tokens"] + totals["output_tokens"]
    assert totals["cost_usd"] is None and totals["cost_cny"] is not None
    triggered = sum(e["kind"] == "fault_injected" for e in events)
    session = {**common, "total_steps": len(tools), "final_tokens": totals["total_tokens"], "final_elapsed_ms": result["elapsed_ms"], "final_elapsed_seconds": result["elapsed_ms"] / 1000, "final_cost_cny": totals["cost_cny"], "currency": "CNY", "termination": result["termination"], "fault_triggered_count": triggered, "fault_triggered": bool(triggered), "recovered_after_fault": bool(triggered and result["label"] == 0), "first_violation_step": result["first_violation_step"], "truncated_responses": totals["truncated_responses"], "actual_repeated_calls": actual_repeats}
    for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
        session[k] = totals[k]
    for k in ("error_count", "retry_count", "schema_error_count", "write_count"):
        session[k] = last[k] if last else 0
    session["reasoning_tokens_missing"] = int(totals["reasoning_tokens"] is None)
    sessions.append(session)

s, p = pd.DataFrame(sessions), pd.DataFrame(prefixes)
assert len(s) == 593 and len(exclusions) == 7 and len(session_ids) == 600
assert not p.duplicated(["session_id", "step"]).any()
assert s.groupby("template_id").split.nunique().eq(1).all()
assert p.session_id.isin(s.session_id).all()
assert s.groupby("split").template_id.nunique().to_dict() == {"test": 2, "train": 6, "validation": 2}
csv("sessions.csv", s)
csv("prefixes.csv", p)
csv("excluded_sessions.csv", pd.DataFrame(exclusions))
csv("quality_by_session.csv", pd.DataFrame(audits))
missing = pd.concat([pd.DataFrame({"table": name, "column": frame.columns, "n": len(frame), "missing": frame.isna().sum().values, "missing_rate": frame.isna().mean().values}) for name, frame in (("sessions", s), ("prefixes", p))], ignore_index=True)
csv("missing_values.csv", missing)
partitions = s.groupby("split").agg(n=("session_id", "size"), templates=("template_id", "nunique"), instances=("task_id", "nunique"), failures=("label", "sum"), failure_rate=("label", "mean")).reset_index()
partitions["successes"] = partitions.n - partitions.failures
partitions["prefixes"] = partitions.split.map(p.groupby("split").size())
csv("partition_summary.csv", partitions)
csv("template_split.csv", pd.DataFrame([{"template_id": k, "split": v} for k, v in assignments.items()]))
save_json("split.json", {"design_hash": design["fingerprint"], "assignments": assignments, "cv_folds": design["cv_folds"], "source": str(SOURCE.relative_to(ROOT)), "seed": design["seed"], "file_hashes": {n: hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ("sessions.csv", "prefixes.csv")}, "note": "冻结分区原样沿用；本EDA表扩充CNY字段，非原load_dataset格式"})
save_json("source_hashes.json", hashes)
quality = {"raw_sessions": 600, "valid_sessions": len(s), "excluded_sessions": len(exclusions), "prefixes": len(p), "events": int(sum(a["events"] for a in audits)), "duplicate_events": int(sum(a["duplicate_events"] for a in audits)), "timestamp_anomalies": int(sum(a["missing_elapsed"] + a["invalid_elapsed"] + a["elapsed_reversals"] for a in audits)), "nonfinite_numbers": int(sum(a["nonfinite_numbers"] for a in audits)), "actual_repeated_calls_retained": int(sum(a["actual_repeated_calls_retained"] for a in audits)), "raw_unknown_usage_requests": int(sum(a["unknown_usage_requests"] for a in audits)), "pilot_included": False, "currency": "CNY", "reasoning_missing_sessions": int(s.reasoning_tokens.isna().sum()), "reasoning_missing_prefixes": int(p.reasoning_tokens.isna().sum()), "checks": ["600套四文件均可解析", "设计hash与任务一致", "Session/condition ID唯一", "工具步数连续且与结果相符", "全部事件elapsed_ms非负有限且单调", "created_at为带时区ISO时间", "decision_eligible与当时硬预算一致", "593标签与success一致", "模板不跨分区", "7无效保留原始记录"]}
save_json("quality.json", quality)
train, pref = s[s.split == "train"].copy(), p[p.split == "train"].copy()
fixed = pref[pref.step == 3].copy()
assert fixed.session_id.is_unique
csv("train_step3_prefixes.csv", fixed)
metrics = ["final_tokens", "final_elapsed_seconds", "total_steps", "final_cost_cny", "input_tokens", "output_tokens", "reasoning_tokens", "error_count", "retry_count", "schema_error_count", "actual_repeated_calls"]
stats = []
for grouping in (["label"], ["workflow"], ["fault_kind"], ["workflow", "label"], ["workflow", "fault_kind"], ["workflow", "fault_kind", "label"]):
    for keys, group in train.groupby(grouping):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            values = group[metric]
            stats.append({"grouping": "+".join(grouping), **dict(zip(grouping, keys)), "n": len(group), "failures": int(group.label.sum()), "failure_rate": group.label.mean(), "metric": metric, "nonmissing_n": values.count(), "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1), "q1": values.quantile(.25), "q3": values.quantile(.75)})
csv("train_grouped_descriptive.csv", pd.DataFrame(stats))
recovery = []
for workflow, frame in [("all", train)] + list(train.groupby("workflow")):
    for fault, group in frame.groupby("fault_kind"):
        hit = group[group.fault_triggered]
        errors = group[group.error_count > 0]
        recovery.append({"workflow": workflow, "fault_kind": fault, "n": len(group), "failures": int(group.label.sum()), "failure_rate": group.label.mean(), "triggered_n": len(hit), "recovered_n": int((hit.label == 0).sum()), "recovery_rate": (hit.label == 0).mean(), "explicit_error_n": len(errors), "success_after_error_n": int((errors.label == 0).sum()), "success_after_error_rate": (errors.label == 0).mean()})
rec = pd.DataFrame(recovery)
csv("train_fault_recovery.csv", rec)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, workflow in zip(axes, ["all"] + sorted(train.workflow.unique())):
    g = rec[rec.workflow == workflow]
    bars = ax.bar(g.fault_kind, g.failure_rate, color="#496a9c")
    for bar, row in zip(bars, g.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+.02, f"{row.failures}/{row.n}", ha="center", fontsize=9)
    ax.set(title=f"Train sessions: {workflow}, n={g.n.sum()}", ylabel="Failure rate", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=35)
save_fig("01_fault_failure.png", fig)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, workflow in enumerate(sorted(train.workflow.unique())):
    g = train[train.workflow == workflow]
    for ax, metric in zip(axes[row], ["final_tokens", "final_elapsed_seconds", "total_steps"]):
        ax.boxplot([g[g.label == label][metric] for label in (0, 1)], tick_labels=[f"Success n={(g.label==0).sum()}", f"Failure n={(g.label==1).sum()}"])
        ax.set(title=f"{workflow}: {metric}", ylabel=metric)
fig.suptitle("Train sessions: resource distributions (outliers retained)")
save_fig("02_resources.png", fig)
fixed["repeat_bin"] = fixed.recent_repeat_ratio.map(lambda v: f"{v:.3f}")
cells = fixed.groupby(["workflow", "error_count", "repeat_bin"]).label.agg(n="size", failures="sum", failure_rate="mean").reset_index()
csv("train_step3_heatmap_cells.csv", cells)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, workflow in zip(axes, sorted(train.workflow.unique())):
    g = cells[cells.workflow == workflow]
    rows, cols = [0, 1, 2, 3], ["0.000", "0.333", "0.667"]
    grid = g.pivot(index="error_count", columns="repeat_bin", values="failure_rate").reindex(index=rows, columns=cols)
    image = ax.imshow(grid, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    for i, err in enumerate(rows):
        for j, rep in enumerate(cols):
            cell = g[(g.error_count == err) & (g.repeat_bin == rep)]
            text = "n=0" if cell.empty else f"{cell.failure_rate.iloc[0]:.0%}\nn={cell.n.iloc[0]}" + (" *" if cell.n.iloc[0] < 10 else "")
            ax.text(j, i, text, ha="center", va="center")
    ax.set(xticks=range(3), xticklabels=cols, yticks=range(4), yticklabels=rows, xlabel="Last-3 repeated-call ratio", ylabel="Cumulative errors", title=f"{workflow}: eligible step 3, n={g.n.sum()}")
    fig.colorbar(image, ax=ax, label="Final failure rate")
fig.suptitle("Train only; one prefix per session; * n<10; gray/blank = no observations")
save_fig("03_step3_failure_heatmap.png", fig)
trend = pref.groupby(["workflow", "step", "label"]).agg(n=("session_id", "size"), error_count=("error_count", "mean"), recent_repeat_ratio=("recent_repeat_ratio", "mean"), total_tokens=("total_tokens", "mean")).reset_index()
csv("train_prefix_trends.csv", trend)
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for row, workflow in enumerate(sorted(train.workflow.unique())):
    g = trend[trend.workflow == workflow]
    for ax, metric in zip(axes[row], ["error_count", "recent_repeat_ratio", "total_tokens"]):
        for label, color in [(0, "#238b45"), (1, "#cb181d")]:
            v = g[g.label == label]
            ax.plot(v.step, v[metric], "o-", label="Success" if label == 0 else "Failure", color=color)
            for item in v.itertuples():
                ax.annotate(str(item.n), (item.step, getattr(item, metric)), xytext=(0, 5 if label else -12), textcoords="offset points", ha="center", fontsize=7, color=color)
        ax.set(title=f"{workflow}: {metric}", xlabel="Tool-call step (numbers = eligible survivor n)", ylabel="Mean " + metric, xticks=sorted(pref.step.unique()))
        ax.legend()
fig.suptitle("Train eligible prefixes: same tool-call count, changing survivor composition")
save_fig("04_prefix_trends.png", fig)
cor_features = [k for k in NUMERIC_FEATURES if k != "cost_usd"] + ["cost_cny"]
cor = fixed[cor_features].corr(method="spearman")
cor.to_csv(OUT / "train_step3_spearman.csv")
pair_n = fixed[cor_features].notna().astype(int).T.dot(fixed[cor_features].notna().astype(int))
pair_n.to_csv(OUT / "train_step3_spearman_pair_n.csv")
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(cor, vmin=-1, vmax=1, cmap="coolwarm")
ax.set(xticks=range(len(cor)), yticks=range(len(cor)), xticklabels=cor.columns, yticklabels=cor.index, title=f"Train step 3 Spearman (eligible sessions n={len(fixed)}; blank = undefined)")
plt.setp(ax.get_xticklabels(), rotation=90)
for i in range(len(cor)):
    for j in range(len(cor)):
        if pd.notna(cor.iloc[i, j]):
            ax.text(j, i, f"{cor.iloc[i,j]:.1f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax)
save_fig("05_step3_spearman.png", fig)
# Extremes are reviewed only within training data, and kept unchanged.
extremes = train.loc[sorted(set(train.nlargest(5, "final_tokens").index) | set(train.nlargest(5, "final_elapsed_seconds").index)), ["session_id", "workflow", "label", "final_tokens", "final_elapsed_seconds", "total_steps", "termination", "truncated_responses"]].copy()
for r in extremes.itertuples():
    result = read(SOURCE/r.session_id/"result.json")
    assert result["totals"]["total_tokens"] == r.final_tokens
    assert result["elapsed_ms"] / 1000 == r.final_elapsed_seconds
csv("train_extreme_sessions.csv", extremes)
term = train.groupby("termination").label.agg(n="size", failures="sum", failure_rate="mean").reset_index()
csv("train_termination.csv", term)
# Check source immutability and generated artifacts after all analysis.
assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
assert len(pd.read_csv(OUT/"sessions.csv")) == 593
assert len(pd.read_csv(OUT/"prefixes.csv")) == len(p)
assert len(list((OUT/"figures").glob("*.png"))) == 5
for image in (OUT/"figures").glob("*.png"):
    assert image.stat().st_size > 1000 and plt.imread(image).ndim == 3
validation = {"passed": True, "source_files_sha256_unchanged": len(hashes), "sessions_csv_rows": len(s), "prefixes_csv_rows": len(p), "train_step3_rows": len(fixed), "figures_readable": 5, "frozen_template_split": "6/2/2", "models_trained": 0, "api_calls": 0}
save_json("validation.json", validation)
summary = {"train_sessions": len(train), "train_failures": int(train.label.sum()), "train_failure_rate": float(train.label.mean()), "train_step3_sessions": len(fixed), "train_step3_failures": int(fixed.label.sum()), "constant_or_all_missing_step3_features": [k for k in cor_features if fixed[k].nunique() <= 1], "train_outcome_resource_medians": train.groupby(["workflow", "label"])[["final_tokens", "final_elapsed_seconds", "total_steps"]].median().reset_index().to_dict("records")}
save_json("training_summary.json", summary)
report = f'''# AgentSentinel-Lite 正式采集 EDA v1

本报告由真实离线数据生成；未调用 API、未读取真实 .env、未训练任何模型。主数据来自 `{SOURCE.relative_to(ROOT)}`，pilot 不混入。proposal 的“尚未采集”为计划时表述，本报告给出采集后的结果。全部探索图、恢复分析、相关性和特征描述仅使用训练分区；验证和测试仅做质量、数量和预先要求的标签比例核对。

## 样本与冻结分区

原始 600 Session，保留有效 {len(s)}，排除 7；有效可继续前缀 {len(p)}。模板分区原样沿用 6/2/2，没有重新随机划分。训练 {len(train)} 条，失败 {int(train.label.sum())}（{train.label.mean():.2%}）；第3步可继续样本 {len(fixed)} 条，其中失败 {int(fixed.label.sum())}（{fixed.label.mean():.2%}）。每个固定步 Session 至多贡献一行。

{md(partitions)}

[Session表](sessions.csv) · [前缀表](prefixes.csv) · [冻结分区](split.json) · [模板分区](template_split.csv) · [分区数量](partition_summary.csv)

## 数据质量与变量口径

检查 {quality['events']} 个事件，完整重复事件 {quality['duplicate_events']}，异常 elapsed 时间 {quality['timestamp_anomalies']}，JSON 数值非有限项 {quality['nonfinite_numbers']}。600套四文件可解析，Session/condition ID 唯一，任务与设计指纹一致，工具步数连续且与 totals 相符。时间检查覆盖全部事件 elapsed_ms 的缺失、负值、非有限和倒退；created_at 验证 ISO 与时区，无法独立验证提供方时钟准确性。真实相同工具与参数再次调用共 {quality['actual_repeated_calls_retained']} 次（含无效Session），全部保留，不当作重复日志删除。本轮没有需要去重的事件。

Reasoning Token 缺失：有效 Session {quality['reasoning_missing_sessions']}/{len(s)}，前缀 {quality['reasoning_missing_prefixes']}/{len(p)}，保留 NA 和 missing 标记；描述统计按非缺失计数，相关系数按成对完整记录，不填零、不插补。Reasoning 已包含在输出Token中，不再相加。USD前缀字段因本轮币种而全空，不代表免费；实际成本使用 `final_cost_cny` / `cost_cny`，单位人民币元，保留 provider usage 的估算口径。无效运行有 {quality['raw_unknown_usage_requests']} 个未知 usage 请求，无法将其总支出当作零。

[质量汇总](quality.json) · [逐Session质量](quality_by_session.csv) · [全部字段缺失率](missing_values.csv) · [7条排除清单](excluded_sessions.csv) · [源文件hash](source_hashes.json)

{md(pd.DataFrame(exclusions)[['session_id','split','termination','unknown_usage_requests']])}

7条无效因基础设施或无法完整核算usage而排除，原始日志仍保留。失败标签来自验证器（完成条件及违规写操作），不以故障条件或终止原因代替。`first_violation_step` 缺失表示没有记录违规，不是需要补零的特征。

## 训练集分组描述与故障恢复

[完整分组描述](train_grouped_descriptive.csv) 按结果、工作流、故障及组合输出 n、失败数、失败率，以及11项数值的非缺失n、均值、中位数、样本标准差、Q1/Q3。标准差使用 ddof=1；单样本组保留NA。

{md(rec[rec.workflow == 'all'].drop(columns='workflow'))}

![训练故障失败率](figures/01_fault_failure.png)

柱上为失败数/有效n；[恢复统计](train_fault_recovery.csv) 同时区分“配置了故障”与“确实触发”。恢复定义为至少一次 `fault_injected` 且最终成功，分母为实际触发Session；另外列出有可见工具错误后成功的比例。故障注入信息只用于事后分组解释，不进入前缀特征。正常环境的恢复率无定义。

## 资源与任务长度

![按工作流的Token、秒数和工具步数箱图](figures/02_resources.png)

{md(train.groupby(['workflow','label'])[['final_tokens','final_elapsed_seconds','total_steps']].median().reset_index())}

label=0为成功，1为失败。箱图保留极端值，须在工作流内比较。已回查训练集Token和耗时各前5名对应原始结果与事件时序，未删除或截尾：[极端轨迹索引](train_extreme_sessions.csv)。完整Session消耗包含最后一轮模型调用，工具后前缀不含未来调用，两者不是同一观察时点。

## 第3步错误与重复调用

![第3步失败率热图](figures/03_step3_failure_heatmap.png)

[每格统计](train_step3_heatmap_cells.csv) · [训练第3步样本](train_step3_prefixes.csv)

仅使用当时 `decision_eligible=true` 的第3次工具调用。重复程度为最近3次调用中 `(3-不同工具及规范化参数签名数)/3`，并非文本重复。星号格 n<10，空格 n=0；不合并工作流。最终标签只作事后比较，故障类别和未来总步数不作早期输入。可继续表示当时尚有预算并可进行下一次模型调用，不要求未来一定再调用工具，不借最终长度筛样本。

## 同步数前缀趋势与相关性

![前缀趋势](figures/04_prefix_trends.png)

[趋势数据及存活n](train_prefix_trends.csv)。每个点比较相同工具调用数，数字是该结果组当时仍可继续的Session数。累计错误、最近重复比例、Token均只从当前及过去工具事件计算。晚期曲线的样本组成不同，不能把跨步均值变化解释成同一批任务的演化，更不能把多行前缀当独立Session。

![训练第3步Spearman](figures/05_step3_spearman.png)

[相关矩阵](train_step3_spearman.csv) · [成对样本数](train_step3_spearman_pair_n.csv)。矩阵不包含最终标签、故障配置或未来资源，使用人民币成本。常量或全缺失列 {', '.join(summary['constant_or_all_missing_step3_features'])} 的相关性未定义，保留空格。固定第3步无法估计长度与其他变量的相关性；不同工作流混合后的相关性也可能反映任务组成。此处为描述，不据此训练或宣布特征选择结论。

## 截断、预算及结论边界

{md(term)}

[训练终止统计](train_termination.csv)。输出截断、12工具步/30000输入输出Token/120秒预算共同影响实际结局；本数据的“失败风险”是当前Agent和预算下的风险，不是任务永远无法完成。max_completion_tokens=1024 可造成模型输出截断。模拟超时和限流立即返回，耗时不代表真实网络故障等待。因早期终止而未到第3步的任务不进入固定步图；达到预算的终末前缀也不作为预警点。长轨迹在前缀总表中行数更多且后期存活者往往更困难，存在生存偏差。

数据仅6个训练模板（每工作流3个），同模板的实例、故障条件和重复运行相关；不把593 Session或数千前缀视为同量独立任务。稀疏格和故障小组比例不稳定，没有进行显著性检验或因果推断。测试仅两个冻结模板，不能据本EDA断言泛化或在线节省。未探索测试特征、未调整分区、未训练模型。

## 复现与验证

在项目根目录直接运行 `.venv/bin/python artifacts/eda_v1/reproduce.py artifacts/eda_v1_recheck` 可在新的空输出目录复现（不经过会读.env的CLI）。默认输出本目录，若已有产物会拒绝覆盖。脚本只导入离线FeatureBuilder、设计验证与JSON函数，不加载Settings.from_env或后端。

[复现脚本](reproduce.py) · [核验结果](validation.json) · [训练摘要](training_summary.json)。全部 {len(hashes)} 个源文件分析前后SHA256一致，表行数和冻结分区核对通过，5张PNG可解码。未修改源数据、分区或执行git提交。
'''
(OUT / "report.md").write_text(report, encoding="utf-8")
print(json.dumps({"quality": quality, "training": summary, "recovery": rec[rec.workflow == "all"].fillna("NA").to_dict("records"), "output": str(OUT)}, ensure_ascii=False, indent=2))

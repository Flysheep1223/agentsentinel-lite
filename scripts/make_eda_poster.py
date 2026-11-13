"""Render bilingual A1 posters emphasizing training-only exploratory analysis.

Uses the reviewed Seedream illustration and the existing poster drawing helpers.
No image-generation request is made. Study inputs are read-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from make_research_poster import (
    AMBER,
    ART,
    BG,
    BOLD,
    CJK,
    COL,
    GRAY,
    INK,
    LINE,
    MUTED,
    PALE,
    REGULAR,
    REPORT,
    RIGHT,
    ROOT,
    TEAL,
    TRAIN,
    WHITE,
    H,
    Poster,
    W,
    check_exports,
    sha256,
    verify_inputs,
)

OUT = ROOT / "docs/poster/agentsentinel_eda_a1"
EARLY = ROOT / "artifacts/expanded_eda_v2/fixed_step_summary.csv"
PARTITIONS = ["violation_and_missing", "violation_only", "missing_only", "state_only"]
PARTITION_COLORS = ["#B86B47", AMBER, "#C8AA71", GRAY]
CONDITIONS = ["none", "partial_response", "rate_limit", "schema_change", "timeout"]

COPY = {
    "en": {
        "kicker": "EXPLORATORY DATA ANALYSIS  /  LLM AGENT RELIABILITY",
        "subtitle": "Failure patterns and recovery\nin tool-call traces",
        "takeaway": "Failure has many forms. Recovery matters.",
        "takeaway_sub": "Training-only EDA: failure modes, task differences, recovery and resource use",
        "intro": "Introduction",
        "question": "How do failure patterns, recovery and resource\nuse vary across tool-using agent tasks?",
        "methods": "Data & methods",
        "dataset": "6,000 runs → 5,894 valid · EDA: train only",
        "cohort_labels": ["EDA TASKS", "SUCCESS", "FAILURE"],
        "scope1": "Kimi · simulated service desk · 12 templates",
        "scope2": "6 parent families · 27,367 prefixes · 5 conditions",
        "verification": "Strict success: required writes, correct final state,\nand no unexpected or duplicate writes.",
        "left_results": "Results / Failure patterns",
        "failure_title": "Four distinct failure outcomes",
        "failure_sub": "Share of 972 failures · mutually exclusive categories",
        "failure_labels": [
            "Violation + missing writes", "Violation only",
            "Missing required writes only", "Final-state mismatch only",
        ],
        "failure_note": "757 / 972 failures exhausted the 12-call budget;\n664 involved violating writes. These groups overlap.",
        "condition_title": "Failure varies by workflow",
        "condition_sub": "Failure rate (%) by assigned condition",
        "condition_names": ["Normal", "Partial response", "Rate limit", "Schema change", "Timeout"],
        "incident": "Incident",
        "ticket": "Ticket",
        "condition_n": "Each workflow × condition: n = 351–356",
        "template_range_note": "failure rates across the\n12 training templates",
        "right_results": "Results / Recovery & resources",
        "error_title": "Tool errors occur in successful runs",
        "error_stat_note": "1,316 / 2,570 successful runs\nexperienced tool errors",
        "success": "Success",
        "failure": "Failure",
        "with_error": "Tool error",
        "without_error": "No tool error",
        "recovery_note": "4,031 / 5,926 read-error events recovered (68.02%);\nmedian recovery gap: 1 step.",
        "recovery_definition": "Recovery: the same read later returns normally.",
        "late_success": "successes first completed at step 12",
        "later_failure": "failures had once met strict completion",
        "resource_title": "Resource differences appear late",
        "resource_sub": "Full-run total tokens · n = 3,542",
        "tokens_unit": "Tokens (×1,000)",
        "box_note": "IQR boxes · 1.5 × IQR whiskers · all outliers retained",
        "step3": "Step 3 · mean tokens",
        "early_note": "Early means remain close: about 4.2k–4.3k.",
        "discussion": "Discussion",
        "discussion_text": "Single-LLM, six-family EDA: associations alone\ndo not establish a universal stopping threshold.",
        "model_note": "Downstream LR: AUC 0.8413; 9.74% cost saved,\nbut 14 fewer successes (offline test, n = 1,176).",
        "conclusion_label": "CONCLUSION",
        "conclusion": "Account for task structure.\nProtect recoverable runs.",
        "footer": "TRAINING-ONLY EDA  /  3,542 TASKS  /  27,367 PREFIXES",
        "footer_sub": "Source: final research report, 6 Nov 2026  ·  EDA uses no validation/test tasks  ·  Illustration: Seedream 5.5",
    },
    "zh": {
        "kicker": "探索性数据分析（EDA） / 大语言模型智能体可靠性",
        "subtitle": "工具调用轨迹中的\n失败模式与恢复线索",
        "takeaway": "失败模式各异，报错不等于失败。",
        "takeaway_sub": "仅训练集 EDA：失败构成、任务差异、错误恢复与资源消耗",
        "intro": "引言",
        "question": "智能体的失败模式、错误恢复与资源消耗，\n如何随任务结构和执行过程变化？",
        "methods": "数据与方法",
        "dataset": "6,000 次运行 → 5,894 条有效任务",
        "cohort_labels": ["EDA 任务", "成功", "失败"],
        "scope1": "Kimi · 模拟服务台 · 仅训练集 · 12 个模板",
        "scope2": "6 个父家族 · 27,367 条前缀 · 5 种条件",
        "verification": "严格成功：必要写入齐全、最终状态正确，\n且无非预期或重复写入。",
        "left_results": "结果 / 失败模式",
        "failure_title": "失败表现为四种不同结果",
        "failure_sub": "占 972 条失败任务的比例 · 各类别互斥",
        "failure_labels": [
            "违规写入且缺必要写入", "仅违规写入",
            "仅缺必要写入", "仅最终状态不匹配",
        ],
        "failure_note": "757 / 972 条失败耗尽 12 次工具调用预算；\n664 条涉及违规写入。两类证据存在重叠。",
        "condition_title": "工作流改变故障条件下的表现",
        "condition_sub": "各运行条件的失败率（%）",
        "condition_names": ["正常", "部分响应", "限流", "Schema 变化", "超时"],
        "incident": "服务事件",
        "ticket": "工单",
        "condition_n": "每个工作流 × 条件：n = 351–356",
        "template_range_note": "12 个训练模板的\n失败率范围",
        "right_results": "结果 / 恢复与资源",
        "error_title": "大量成功任务也经历过工具报错",
        "error_stat_note": "1,316 / 2,570 条成功任务\n曾出现显式工具错误",
        "success": "成功",
        "failure": "失败",
        "with_error": "出现报错",
        "without_error": "未出现报错",
        "recovery_note": "4,031 / 5,926 次读取报错后恢复（68.02%）；\n恢复间隔中位数为 1 步。",
        "recovery_definition": "恢复：后续相同工具、相同参数的读取正常返回。",
        "late_success": "条成功任务直到第 12 步才首次达标",
        "later_failure": "条失败任务曾达标，随后继续执行而失败",
        "resource_title": "完整轨迹差异明显，早期差异弱",
        "resource_sub": "完整轨迹总 Token · n = 3,542",
        "tokens_unit": "Token（千）",
        "box_note": "箱体为 IQR · 须为 1.5 × IQR · 保留全部外点",
        "step3": "第 3 步 Token 均值",
        "early_note": "早期均值仍接近：约 4.2k–4.3k Token。",
        "discussion": "讨论",
        "discussion_text": "证据仅来自一个模型、六个训练家族；\n描述性关联不足以推出统一的停止阈值。",
        "model_note": "后续 LR：AUC 0.8413，费用节省 9.74%，\n但成功净减 14 条（离线测试，n = 1,176）。",
        "conclusion_label": "结  论",
        "conclusion": "结合任务结构分析失败，\n保护仍可恢复的执行。",
        "footer": "仅训练集 EDA  /  3,542 条任务  /  27,367 条前缀",
        "footer_sub": "来源：最终研究报告，2026 年 11 月 6 日 · EDA 未使用验证／测试任务 · 插图：Seedream 5.5",
    },
}


def eda_evidence(data: pd.DataFrame) -> dict:
    failure_counts = data.loc[data.label == 1, "failure_partition"].value_counts()
    assert [int(failure_counts[key]) for key in PARTITIONS] == [411, 253, 277, 31]
    assert data.template_id.nunique() == 12 and data.parent_family.nunique() == 6
    assert int(data.eligible_prefix_count.sum()) == 27367
    assert int(((data.label == 0) & (data.first_clean_completion_step == 12)).sum()) == 77
    assert int(((data.label == 1) & data.once_cleanly_complete).sum()) == 80
    assert int(data.recovered_read_error_events.sum()) == 4031
    assert int(data.explicit_error_count.sum()) == 5926
    assert int(((data.label == 1) & data.has_error).sum()) == 603
    assert int(((data.label == 1) & data.has_violation).sum()) == 664
    assert int(((data.label == 1) & (data.tool_calls == 12) &
                (data.termination == "step_budget")).sum()) == 757
    templates = data.groupby("template_id").label.agg(["count", "sum", "mean"])
    assert round(float(templates["mean"].min() * 100), 2) == 0.67
    assert round(float(templates["mean"].max() * 100), 2) == 90.41
    conditions = []
    for condition in CONDITIONS:
        for workflow in ["incident", "ticket"]:
            cohort = data[(data.condition == condition) & (data.workflow == workflow)]
            n, failed = len(cohort), int(cohort.label.sum())
            assert 351 <= n <= 356
            conditions.append({
                "condition": condition, "workflow": workflow,
                "n": n, "failed": failed, "failure_rate_pct": failed / n * 100,
            })
    return {
        "training_only": True,
        "tasks": 3542, "successes": 2570, "failures": 972,
        "templates": 12, "parent_families": 6, "eligible_prefixes": 27367,
        "failure_composition": {key: int(failure_counts[key]) for key in PARTITIONS},
        "budget_exhausted_failures": 757,
        "failures_with_violation": 664,
        "condition_workflow_rates": conditions,
        "error_by_outcome": {
            "success": {"with_error": 1316, "without_error": 1254},
            "failure": {"with_error": 603, "without_error": 369},
        },
        "read_error_recovery": {"recovered_events": 4031, "error_events": 5926, "median_gap_steps": 1},
        "late_successes_step12": 77, "failures_once_complete": 80,
        "template_failure_rate_pct": {
            "minimum": float(templates["mean"].min() * 100),
            "maximum": float(templates["mean"].max() * 100),
        },
    }


class EDAPoster(Poster):
    def __init__(self, language, data, evidence):
        super().__init__(language, data)
        self.c = COPY[language]
        self.evidence = evidence

    def introduction_methods(self):
        c = self.c
        self.section(20, 190, "01", c["intro"])
        self.bullet(20, 219, c["question"], 27)
        self.section(20, 258, "02", c["methods"])
        self.t(20, 287, c["dataset"], 28 if self.lang == "en" else 26, bold=True, w=270)
        for i, (label, value, color) in enumerate(zip(
            c["cohort_labels"], ["3,542", "2,570", "972"], [INK, TEAL, AMBER],
        )):
            x = 20 + i * 92
            self.rect(x, 306, 86, 39, PALE)
            self.t(x + 7, 312, label, 20.5, MUTED, w=73)
            self.t(x + 7, 324, value, 37, color, bold=True, latin=True, w=73)
        self.t(20, 353, c["scope1"], 22.5, TEAL, bold=True, w=270)
        self.t(20, 366, c["scope2"], 22.5, MUTED, w=270)
        self.bullet(20, 383, c["verification"], 24)

    def failure_patterns(self):
        c = self.c
        self.section(20, 415, "03", c["left_results"], size=32)
        self.t(20, 444, c["failure_title"], 29, bold=True, w=270)
        self.t(20, 460, c["failure_sub"], 21.5, MUTED, w=270)
        for index, (key, label, color) in enumerate(zip(PARTITIONS, c["failure_labels"], PARTITION_COLORS)):
            count = self.evidence["failure_composition"][key]
            pct = count / 972 * 100
            y = 481 + index * 24
            self.t(20, y, label, 24, w=182)
            self.t(290, y, f"{count}  ·  {pct:.2f}%", 23.5, color, bold=True,
                   latin=True, align="right", w=84)
            self.rect(20, y + 13, COL, 5, PALE)
            self.rect(20, y + 13, COL * pct / 100, 5, color)
        self.t(20, 580, c["failure_note"], 22.5, MUTED, w=270, h=23)

    def condition_chart(self):
        c = self.c
        self.t(20, 616, c["condition_title"], 29 if self.lang == "en" else 27,
               bold=True, w=270)
        self.t(20, 633, c["condition_sub"], 21.5, MUTED, w=270)
        self.bg.scatter([184], [649], s=65, c=AMBER, marker="o", zorder=4)
        self.t(190, 645, c["incident"], 20, MUTED, w=48)
        self.bg.scatter([247], [649], s=65, facecolor=BG, edgecolor=AMBER,
                        marker="D", linewidth=1.7, zorder=4)
        self.t(253, 645, c["ticket"], 20, MUTED, w=37)
        axis = self.ax(109, 661, 166, 91)
        for index, condition in enumerate(CONDITIONS):
            y = 4 - index
            rows = {row["workflow"]: row for row in self.evidence["condition_workflow_rates"]
                    if row["condition"] == condition}
            incident, ticket = [rows[key]["failure_rate_pct"] for key in ["incident", "ticket"]]
            axis.plot([ticket, incident], [y, y], color=LINE, lw=2.5, zorder=2)
            axis.scatter([incident], [y], s=95, c=AMBER, marker="o", zorder=4)
            axis.scatter([ticket], [y], s=80, facecolor=BG, edgecolor=AMBER,
                         linewidth=1.8, marker="D", zorder=4)
            axis.text(incident + 1.3, y, f"{incident:.1f}", ha="left", va="center",
                      color=INK, fontproperties=self.font(20, latin=True))
            axis.text(ticket - 1.3, y, f"{ticket:.1f}", ha="right", va="center",
                      color=MUTED, fontproperties=self.font(20, latin=True))
        axis.set(xlim=(0, 55), ylim=(-0.55, 4.55), yticks=range(5), xticks=[0, 20, 40])
        axis.set_yticklabels(list(reversed(c["condition_names"])), fontproperties=self.font(21))
        for tick in axis.get_xticklabels():
            tick.set_fontproperties(self.font(19, latin=True))
        axis.tick_params(axis="both", length=0, pad=7, colors=MUTED)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color=LINE, linewidth=0.9)
        for spine in axis.spines.values():
            spine.set_visible(False)
        self.t(20, 767, c["condition_n"], 19.5, MUTED, w=270)
        self.t(20, 783, "0.67%–90.41%", 33, INK, bold=True, latin=True, w=124)
        self.t(156, 782, c["template_range_note"], 23.5, MUTED, w=134, h=25)

    def recovery_chart(self):
        c = self.c
        self.section(RIGHT, 190, "03", c["right_results"], size=30)
        self.t(RIGHT, 220, c["error_title"], 28, bold=True, w=270)
        self.t(RIGHT, 241, "51.21%", 50, TEAL, bold=True, latin=True, w=86)
        self.t(402, 243, c["error_stat_note"], 23.5, w=172, h=26)
        for x, label, color in [(396, c["with_error"], INK), (494, c["without_error"], PALE)]:
            self.rect(x, 277, 4.3, 4.3, color, edge=LINE, lw=0.6)
            self.t(x + 7, 274, label, 20, MUTED, w=73)
        for y, label, n, count, color in [
            (296, c["success"], 2570, 1316, TEAL),
            (328, c["failure"], 972, 603, AMBER),
        ]:
            self.t(RIGHT, y - 2, label, 23.5, color, bold=True, w=79)
            self.t(RIGHT, y + 10, f"n = {n:,}", 19.5, MUTED, latin=True, w=79)
            share = count / n
            x, width = 396, 178
            self.rect(x, y, width, 17, PALE)
            self.rect(x, y, width * share, 17, INK)
            self.t(x + width * share / 2, y + 3, f"{share * 100:.1f}%", 24, WHITE,
                   bold=True, latin=True, align="center", w=width * share - 2, h=14)
            self.t(x + width * (share + (1 - share) / 2), y + 3, f"{(1 - share) * 100:.1f}%",
                   24, MUTED, latin=True, align="center", w=width * (1 - share) - 2, h=14)
        self.t(RIGHT, 357, c["recovery_note"], 22.5, MUTED, w=270, h=25)
        self.t(RIGHT, 380, c["recovery_definition"], 18.5, MUTED, w=270, h=9)
        for y, value, text, color in [
            (389, "77", c["late_success"], TEAL),
            (409, "80", c["later_failure"], AMBER),
        ]:
            self.t(RIGHT, y - 2, value, 31, color, bold=True, latin=True, w=22)
            self.t(RIGHT + 29, y, text, 22.5, INK, w=241, h=15)
        self.line(RIGHT, 432, 574, 432)

    def resources(self):
        c = self.c
        self.t(RIGHT, 444, c["resource_title"], 29 if self.lang == "en" else 27,
               bold=True, w=270)
        self.t(RIGHT, 461, c["resource_sub"], 21.5, MUTED, w=191)
        for x, label, color in [(494, c["success"], TEAL), (539, c["failure"], AMBER)]:
            self.rect(x, 464, 4, 3.5, color)
            self.t(x + 6, 461, label, 18.5, MUTED, w=32)
        self.t(RIGHT, 477, c["tokens_unit"], 19, MUTED, w=80)
        axis = self.ax(328, 492, 241, 65)
        groups = [self.data.loc[(self.data.workflow == workflow) & (self.data.label == label),
                               "total_tokens"].to_numpy() / 1000
                  for workflow, label in [("incident", 0), ("incident", 1), ("ticket", 0), ("ticket", 1)]]
        plot = axis.boxplot(
            groups, positions=[0.8, 1.4, 2.6, 3.2], widths=0.4,
            patch_artist=True, whis=1.5, showfliers=True, manage_ticks=False,
            medianprops={"color": INK, "linewidth": 2},
            whiskerprops={"color": MUTED, "linewidth": 1.3},
            capprops={"color": MUTED, "linewidth": 1.3},
            flierprops={"marker": "o", "markersize": 3, "markeredgecolor": MUTED,
                        "markerfacecolor": BG, "markeredgewidth": 0.7},
        )
        for box, color in zip(plot["boxes"], [TEAL, AMBER, TEAL, AMBER]):
            box.set(facecolor=color, edgecolor=color, alpha=0.78, linewidth=1.5)
        axis.set(xlim=(0.35, 3.65), ylim=(0, 33), yticks=[0, 10, 20, 30])
        axis.set_xticks([1.1, 2.9])
        axis.set_xticklabels([
            c["incident"] + "\nn = 1,003 / 766", c["ticket"] + "\nn = 1,567 / 206",
        ], fontproperties=self.font(19.5), linespacing=1.10)
        for tick in axis.get_yticklabels():
            tick.set_fontproperties(self.font(19, latin=True))
        axis.tick_params(axis="both", length=0, pad=5, colors=MUTED)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=LINE, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        self.t(RIGHT, 584, c["box_note"], 18.5, MUTED, w=270)
        self.t(RIGHT, 604, c["step3"], 23, bold=True, w=142)
        self.t(485, 604, c["success"], 21, TEAL, align="right", w=64)
        self.t(574, 604, c["failure"], 21, AMBER, align="right", w=64)
        self.line(RIGHT, 617, 574, 617)
        early = pd.read_csv(EARLY)
        for y, workflow in [(623, "incident"), (639, "ticket")]:
            self.t(RIGHT, y, c[workflow], 23.5, w=118)
            for x, label, color in [(485, 0, TEAL), (574, 1, AMBER)]:
                value = float(early.loc[
                    (early.step == 3) & (early.workflow == workflow) & (early.label == label),
                    "total_tokens_mean",
                ].iloc[0])
                self.t(x, y, f"{value:,.0f}", 25, color, bold=True, latin=True,
                       align="right", w=64)
        self.t(RIGHT, 659, c["early_note"], 22.5, TEAL, bold=True, w=270)

    def discussion_conclusion(self):
        c = self.c
        self.section(RIGHT, 685, "04", c["discussion"], size=32)
        self.bullet(RIGHT, 713, c["discussion_text"], 23.5)
        self.t(RIGHT, 746, c["model_note"], 22.5, MUTED, w=270, h=25)
        self.rect(RIGHT, 778, COL, 27, PALE, edge=TEAL, lw=2)
        self.rect(RIGHT, 778, 2.2, 27, TEAL)
        self.t(RIGHT + 10, 785, c["conclusion_label"], 19, TEAL, bold=True, w=79)
        self.t(RIGHT + 84, 783, c["conclusion"], 24.5, bold=True,
               w=178, h=23, linespacing=1.08)

    def build(self):
        self.header()
        self.introduction_methods()
        self.failure_patterns()
        self.condition_chart()
        self.recovery_chart()
        self.resources()
        self.discussion_conclusion()
        self.footer()
        self.fig.canvas.draw()
        return self.check_layout()

    def export(self, preview_only=False):
        stem = OUT / f"AgentSentinel_EDA_A1_{self.lang.upper()}"
        preview = Path(str(stem) + "_preview.png")
        self.fig.savefig(preview, dpi=90, facecolor=BG)
        files = [preview]
        if not preview_only:
            for suffix in [".pdf", ".svg", ".png"]:
                path = stem.with_suffix(suffix)
                kwargs = {"dpi": 300, "facecolor": BG}
                if suffix == ".pdf":
                    kwargs["metadata"] = {
                        "Title": "AgentSentinel-Lite — Training-only exploratory data analysis",
                        "Subject": "Failure composition, task heterogeneity, recovery, and resource distributions",
                        "Creator": "Matplotlib; conceptual illustration by Seedream 5.5",
                    }
                elif suffix == ".svg":
                    kwargs["metadata"] = {
                        "Title": "AgentSentinel-Lite — EDA poster",
                        "Description": "Editable A1 training-only exploratory data analysis poster",
                    }
                self.fig.savefig(path, **kwargs)
                files.append(path)
        (OUT / f"poster_text_{self.lang}.txt").write_text("\n\n".join(self.body_texts) + "\n")
        plt.close(self.fig)
        return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--language", choices=["en", "zh", "both"], default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "savefig.bbox": None,
        "font.family": "Arial", "font.size": 24,
    })
    sources = [REPORT, TRAIN, EARLY, ART, ART.parent / "metadata.json"]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    data, distribution_checks = verify_inputs()
    evidence = eda_evidence(data)
    files, layout = [], {}
    for language in (["en", "zh"] if args.language == "both" else [args.language]):
        poster = EDAPoster(language, data, evidence)
        layout[language] = poster.build()
        files.extend(poster.export(args.preview_only))
        print(f"{language.upper()}: EDA poster rendered; text boxes fit.", flush=True)
    assert before == {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": "EDA-focused revision requested by user",
        "page_mm": [W, H], "print_dpi": 300, "preview_only": args.preview_only,
        "source_inputs_unchanged": True, "source_sha256": before,
        "scripts": {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__), ROOT / "scripts/make_research_poster.py",
        ]},
        "fonts": {"latin_regular": REGULAR, "latin_bold": BOLD, "chinese": CJK},
        "eda_evidence": evidence, "distribution_checks": distribution_checks,
        "scope": {
            "all_eda_charts": "training split only",
            "condition_comparison": "assigned conditions; descriptive, not causal",
            "resource_boxplot_n_labels": "success / failure within each workflow",
            "recovery_rate": "events, not tasks; a later same-tool/same-arguments read returns normally",
            "model_result": "small discussion note, separately labeled offline test n=1176",
            "feature_leakage": "verifier data used to describe outcomes, not as model inputs",
        },
        "outputs": check_exports(files),
    }
    name = "preview_manifest.json" if args.preview_only else "poster_manifest.json"
    (OUT / name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "layout_checks.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"outputs": manifest["outputs"], "source_inputs_unchanged": True},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

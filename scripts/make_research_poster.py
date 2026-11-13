"""Build bilingual A1 research posters from the frozen main study.

Usage:
    .venv/bin/python scripts/make_research_poster.py --preview-only
    .venv/bin/python scripts/make_research_poster.py

Inputs are read-only. The sole generated image is conceptual decoration; all
charts, numbers, labels, and body copy are rendered separately as vector objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/poster/agentsentinel_a1"
REPORT = ROOT / "docs/RESEARCH_REPORT_6000_FINAL.md"
TRAIN = ROOT / "artifacts/expanded_eda_v2/diagnostic_sessions.csv"
EARLY = ROOT / "artifacts/expanded_eda_v2/fixed_step_summary.csv"
ART = OUT / "visual_v1/seedream_visual.png"
W, H = 594.0, 841.0
MARGIN, COL, RIGHT = 20.0, 270.0, 304.0
BG = "#F6F4EF"
INK = "#12313B"
TEAL = "#26867C"
AMBER = "#D89845"
GRAY = "#8D9C9D"
MUTED = "#526A70"
LINE = "#CED7D2"
PALE = "#E7EEEA"
WHITE = "#FFFFFF"
URL = "https://github.com/Flysheep1223/agentsentinel-lite"
REPORT_URL = URL + "/blob/main/docs/RESEARCH_REPORT_6000_FINAL.md"
REGULAR = fm.findfont(FontProperties(family="Arial"))
BOLD = fm.findfont(FontProperties(family="Arial", weight="bold"))
CJK = "/Library/Fonts/Arial Unicode.ttf"

METRICS = {
    "experiment": "frozen_main_22_features",
    "execution_model": "kimi-k3-1-pro-261024",
    "planned_runs": 6000,
    "invalid_runs": 106,
    "valid_tasks": 5894,
    "eligible_prefixes": 47184,
    "train_tasks": 3542,
    "validation_tasks": 1176,
    "test_tasks": 1176,
    "test_success": 855,
    "test_failure": 321,
    "test_prefixes": 9288,
    "lr_full_prefix_roc_auc": 0.8413,
    "knn_full_prefix_roc_auc": 0.7749,
    "lr_full_prefix_average_precision": 0.8030,
    "knn_full_prefix_average_precision": 0.7295,
    "lr_step3_roc_auc": 0.7422,
    "knn_step3_roc_auc": 0.6099,
    "threshold": 0.7380596938,
    "failure_alerts": 267,
    "false_stops": 17,
    "rescued": 3,
    "replay_success": 841,
    "net_success_change": -14,
    "baseline_cost_cny": 206.1985612,
    "saved_cost_cny": 20.087795,
    "remaining_cost_cny": 186.1107662,
    "failure_alert_timing": {
        "before_first_violation": 32,
        "same_step": 55,
        "after_first_violation": 154,
        "no_recorded_violation": 26,
    },
    "violation_bearing_test_tasks": 268,
    "outcomes": "strict offline state replay; no online intervention",
}

COPY = {
    "en": {
        "kicker": "LLM AGENT RELIABILITY  /  RESEARCH POSTER",
        "subtitle": "Early failure warning\nfrom tool-call traces",
        "takeaway": "Early warning helps. Stopping still has a price.",
        "takeaway_sub": "Predict failure  •  Protect recovery  •  Report savings and success loss together",
        "intro": "Introduction",
        "intro_bullets": [
            "Tool errors may recover; normal returns\ncan still violate the task.",
            "Can runtime signals save resources\nwhile preserving task success?",
        ],
        "methods": "Methods",
        "cohort": "6,000 runs → 5,894 valid tasks",
        "split_names": ["TRAIN", "VALIDATION", "TEST"],
        "split_families": ["6 families", "2 families", "2 families"],
        "split_note": "Parent task families stay in one split.",
        "audit_note": "106 infra/protocol exclusions · 47,184 eligible prefixes",
        "env_bullet": "Real Kimi agent · ticket & incident workflows.\n20 templates; 5 conditions; ≤12 tool calls per run.",
        "success_bullet": "Success: required writes + correct final state;\nno unexpected or duplicate writes.",
        "pipeline": ["Tool\nevents", "22 runtime\nfeatures", "LR risk\nscore", "Continue\nor stop"],
        "feature_note": "Errors / retries · tool context · resources · writes\nOnly observed events; no future or verifier data.",
        "strategy": "Compare LR, KNN and rules. Select the threshold\non validation with a 5% false-stop budget.",
        "model_names": "LR: logistic regression · KNN: k-nearest neighbors",
        "training": "Results / Training evidence",
        "recover_label": "of training successes\nexperienced tool errors",
        "tokens_title": "Failed runs consume more tokens",
        "tokens_note": "Full trajectories · training only · n = 3,542",
        "success": "Success",
        "failure": "Failure",
        "tokens_unit": "Tokens (×1,000)",
        "workflows": ["Incident", "Ticket"],
        "box_note": "IQR boxes · 1.5 × IQR whiskers · all outliers",
        "early_note": "Yet step-3 means overlap: 4.2k–4.3k tokens.",
        "results": "Results / Main test",
        "test_note": "HELD-OUT TEST · 1,176 TASKS · OFFLINE REPLAY",
        "kpi_labels": ["Failure recall", "Cost saved", "Net successes"],
        "kpi_notes": ["267 / 321 failures", "Replay estimate", "855 → 841"],
        "false_stop": "17 / 855 successful tasks stopped (1.99%).",
        "auc_title": "Prediction quality",
        "auc_rows": ["All prefixes", "Step 3"],
        "auc_row_notes": ["n = 9,288 prefixes", "n = 1,176 tasks"],
        "auc_axis": "ROC AUC",
        "ap_note": "Full-prefix AP: LR 0.8030 · KNN 0.7295",
        "trade_title": "Resource gain, success loss",
        "equation_labels": ["Baseline", "Lost", "Rescued", "Replay"],
        "cost_line": "Estimated cost: ¥206.20 → ¥186.11",
        "cost_note": "¥20.09 saved in strict offline replay",
        "timing_title": "Most alerts follow the first violation",
        "timing_note": "Timing within the 267 alerted failures",
        "timing_legend": ["Before", "Same step", "After", "No violation"],
        "before_note": "Only 32 / 268 violation-bearing tasks (11.94%)\nwere warned before their first violation.",
        "discussion": "Discussion",
        "discussion_bullets": [
            "One LLM, two workflows, two test families;\nno online intervention experiment in this batch.",
            "Scoring follows the tool return; the same alert\ncannot undo a write that has already happened.",
        ],
        "conclusion_label": "CONCLUSION",
        "conclusion": "Useful for risk monitoring.\nNot yet a loss-free stop policy.",
        "footer": "FROZEN MAIN EXPERIMENT  /  22 OBSERVABLE FEATURES",
        "footer_sub": "Source: final research report, 6 Nov 2026  ·  Concept illustration: Seedream 5.5",
    },
    "zh": {
        "kicker": "大语言模型智能体可靠性  /  研究海报",
        "subtitle": "基于工具调用轨迹的\n智能体失败预警",
        "takeaway": "失败可预警，提前停止仍有代价。",
        "takeaway_sub": "预测失败  ·  保护恢复过程  ·  同时报告资源节省与成功损失",
        "intro": "引言",
        "intro_bullets": [
            "工具报错后仍可能恢复；正常返回也可能\n违反任务要求。",
            "仅靠运行时信号，能否在保留任务成功的\n同时节省资源？",
        ],
        "methods": "研究方法",
        "cohort": "6,000 次运行 → 5,894 条有效任务",
        "split_names": ["训练集", "验证集", "测试集"],
        "split_families": ["6 个家族", "2 个家族", "2 个家族"],
        "split_note": "同一父任务家族始终留在同一分区。",
        "audit_note": "排除 106 条设施／协议异常 · 47,184 条可决策前缀",
        "env_bullet": "真实 Kimi 模型；工单与服务事件两类工作流。\n20 个模板，5 种运行条件，每条最多 12 次工具调用。",
        "success_bullet": "严格成功：必要写入齐全、最终状态正确，\n且无非预期或重复写入。",
        "pipeline": ["工具\n事件", "22 项\n运行特征", "LR 失败\n概率", "继续\n或停止"],
        "feature_note": "错误／重试、调用背景、资源消耗、写入行为。\n仅用已观察事件，不含未来结果或验收信息。",
        "strategy": "比较 LR、KNN 与规则；在验证集 5% 误停预算\n内选择阈值，全部收益采用离线回放估算。",
        "model_names": "LR：逻辑回归 · KNN：K 近邻",
        "training": "结果 / 训练集探索",
        "recover_label": "的训练集成功任务\n曾出现工具错误",
        "tokens_title": "失败任务的完整轨迹消耗更多 Token",
        "tokens_note": "完整轨迹 · 仅训练集 · n = 3,542",
        "success": "成功",
        "failure": "失败",
        "tokens_unit": "Token（千）",
        "workflows": ["服务事件", "工单"],
        "box_note": "箱体为 IQR · 须为 1.5 × IQR · 保留全部外点",
        "early_note": "但第 3 步均值接近：约 4.2k–4.3k Token。",
        "results": "结果 / 主实验测试",
        "test_note": "独立测试集 · 1,176 条任务 · 离线回放",
        "kpi_labels": ["失败召回率", "费用节省", "成功数净变化"],
        "kpi_notes": ["267 / 321 条失败", "离线回放估算", "855 → 841"],
        "false_stop": "误停原本成功的任务 17 / 855 条（1.99%）。",
        "auc_title": "失败预测能力",
        "auc_rows": ["全前缀", "第 3 步"],
        "auc_row_notes": ["n = 9,288 条前缀", "n = 1,176 条任务"],
        "auc_axis": "ROC AUC",
        "ap_note": "全前缀 AP：LR 0.8030 · KNN 0.7295",
        "trade_title": "资源收益与成功代价",
        "equation_labels": ["原始成功", "损失", "挽救", "回放成功"],
        "cost_line": "估算费用：¥206.20 → ¥186.11",
        "cost_note": "严格离线回放估算节省 ¥20.09",
        "timing_title": "多数告警发生在首次违规之后",
        "timing_note": "267 条被告警失败任务的告警时点",
        "timing_legend": ["违规之前", "同一步", "违规之后", "无记录违规"],
        "before_note": "含违规任务中，仅 32 / 268 条（11.94%）\n在首次违规前收到告警。",
        "discussion": "讨论",
        "discussion_bullets": [
            "仅一个模型、两类工作流、两个测试家族；\n本批次未开展在线干预实验。",
            "评分发生在工具返回之后；当前告警无法\n撤销同一步已经发生的写入。",
        ],
        "conclusion_label": "结  论",
        "conclusion": "适合辅助风险监测，\n尚不能实现无成功损失的自动停止。",
        "footer": "冻结主实验  /  22 项可观察运行特征",
        "footer_sub": "来源：最终研究报告，2026 年 11 月 6 日  ·  概念插图：Seedream 5.5",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(TRAIN)
    assert len(data) == METRICS["train_tasks"]
    assert set(data["split"]) == {"train"}
    assert data["valid_for_analysis"].all()
    assert data.session_id.is_unique
    assert data.total_tokens.notna().all() and (data.total_tokens >= 0).all()
    assert data.label.value_counts().to_dict() == {0: 2570, 1: 972}
    assert int(((data.label == 0) & data.has_error).sum()) == 1316
    expected = {
        ("incident", 0): (1003, 14407.0),
        ("incident", 1): (766, 24945.5),
        ("ticket", 0): (1567, 12410.0),
        ("ticket", 1): (206, 24061.5),
    }
    observed = {}
    for (workflow, label), (n, median) in expected.items():
        values = data.loc[(data.workflow == workflow) & (data.label == label), "total_tokens"]
        assert (len(values), float(values.median())) == (n, median)
        observed[f"{workflow}_{label}"] = {
            "n": n,
            "median": median,
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    early = pd.read_csv(EARLY)
    step3 = early.loc[early.step == 3]
    assert len(step3) == 4 and step3.total_tokens_mean.between(4200, 4300).all()
    text = REPORT.read_text()
    for token in (
        "0.8413", "0.7749", "0.8030", "0.7295", "0.7422", "0.6099",
        "83.18%", "9.74%", "0.7380596938", "32/268", "−14",
    ):
        assert token in text, f"Metric absent from source report: {token}"
    assert sum(METRICS["failure_alert_timing"].values()) == 267
    assert 855 - 17 + 3 == METRICS["replay_success"]
    assert round(267 / 321 * 100, 2) == 83.18
    assert round(METRICS["saved_cost_cny"] / METRICS["baseline_cost_cny"] * 100, 2) == 9.74
    assert round(32 / 268 * 100, 2) == 11.94
    return data, {
        "training_boxplot_groups": observed,
        "step3_means": step3[["workflow", "label", "n", "total_tokens_mean"]].to_dict("records"),
    }


class Poster:
    def __init__(self, lang: str, data: pd.DataFrame):
        self.lang = lang
        self.c = COPY[lang]
        self.data = data
        self.fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=72, facecolor=BG)
        self.bg = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.bg.set(xlim=(0, W), ylim=(H, 0))
        self.bg.set_axis_off()
        self.text_bounds = []
        self.body_texts = []

    def font(self, size, bold=False, latin=False):
        path = (BOLD if bold else REGULAR) if self.lang == "en" or latin else CJK
        return FontProperties(fname=path, size=size)

    def t(self, x, y, text, size=26, color=INK, bold=False, w=None, h=None,
          align="left", latin=False, linespacing=1.22, url=None):
        artist = self.fig.text(
            x / W, 1 - y / H, text, fontproperties=self.font(size, bold, latin),
            color=color, va="top", ha=align, linespacing=linespacing,
            zorder=6, url=url,
        )
        self.text_bounds.append((artist, w, h, x, y))
        self.body_texts.append(text)
        return artist

    def rect(self, x, y, w, h, color, edge=None, lw=1, alpha=1):
        patch = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge or "none",
                          linewidth=lw, alpha=alpha, zorder=1)
        self.bg.add_patch(patch)
        return patch

    def line(self, x1, y1, x2, y2, color=LINE, lw=1.4):
        self.bg.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)

    def section(self, x, y, number, title, size=34):
        self.t(x, y + 1.4, number, 24, TEAL, bold=True, latin=True, w=16)
        self.t(x + 21, y, title, size, bold=True, w=COL - 21, h=15)
        self.line(x, y + 18, x + COL, y + 18, INK, 1.4)

    def bullet(self, x, y, text, size=26):
        self.rect(x + 0.5, y + 3.7, 2.1, 2.1, TEAL)
        self.t(x + 8, y, text, size, w=COL - 8, h=28)

    def ax(self, x, y, w, h):
        return self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H],
                                 facecolor=BG, zorder=3)

    def header(self):
        image = Image.open(ART).convert("RGB").crop((0, 280, 2512, 1410))
        pixels = np.asarray(image).astype(np.float32)
        height, width = pixels.shape[:2]
        horizontal = np.clip(np.arange(width) / (width * 0.22), 0, 1)
        vertical = np.minimum(
            np.clip(np.arange(height) / (height * 0.12), 0, 1),
            np.clip((height - 1 - np.arange(height)) / (height * 0.14), 0, 1),
        )
        alpha = vertical[:, None] * horizontal[None, :]
        background = np.array([246, 244, 239], dtype=np.float32)
        blended = (pixels * alpha[:, :, None] + background * (1 - alpha[:, :, None])).astype(np.uint8)
        image_ax = self.ax(345, 3, 249, 112)
        image_ax.imshow(blended, aspect="auto")
        image_ax.set_axis_off()
        self.rect(20, 9, 26, 2.3, TEAL)
        self.t(20, 18, self.c["kicker"], 22, MUTED, w=330)
        self.t(20, 39, "AgentSentinel-Lite", 84, INK, bold=True, latin=True, w=326)
        self.t(20, 79, self.c["subtitle"], 42, INK, w=330, h=38, linespacing=1.12)
        self.rect(20, 127, 554, 48, INK)
        self.rect(20, 127, 2.2, 48, TEAL)
        self.t(32, 136, self.c["takeaway"], 40 if self.lang == "en" else 37,
               WHITE, bold=True, w=530, h=19)
        self.t(32, 158, self.c["takeaway_sub"], 25, "#C9DFD8", w=530, h=12)
        self.line(297, 191, 297, 801, LINE, 1.1)

    def introduction_methods(self):
        c = self.c
        self.section(20, 190, "01", c["intro"])
        self.bullet(20, 219, c["intro_bullets"][0], 27)
        self.bullet(20, 248, c["intro_bullets"][1], 27)
        self.section(20, 284, "02", c["methods"])
        self.t(20, 313, c["cohort"], 34 if self.lang == "en" else 30, bold=True, w=270)
        for i, (name, number, family) in enumerate(zip(
            c["split_names"], ["3,542", "1,176", "1,176"], c["split_families"]
        )):
            x = 20 + i * 92
            self.rect(x, 336, 86, 45, PALE)
            self.t(x + 7, 342, name, 20, MUTED, w=73)
            self.t(x + 7, 351, number, 37, INK, bold=True, latin=True, w=73)
            self.t(x + 7, 368, family, 19.5, MUTED, w=73)
        self.t(20, 387, c["split_note"], 22.5, TEAL, bold=True, w=270)
        self.t(20, 399, c["audit_note"], 21.5 if self.lang == "en" else 21, MUTED, w=270)
        self.bullet(20, 418, c["env_bullet"], 25 if self.lang == "en" else 24)
        self.bullet(20, 446, c["success_bullet"], 25)
        for i, label in enumerate(c["pipeline"]):
            x = 20 + i * 70
            self.rect(x, 478, 60, 31, INK if i == 2 else PALE, edge=LINE, lw=1.0)
            self.t(x + 30, 483, label, 23, WHITE if i == 2 else INK,
                   bold=(i == 2), w=56, h=24, align="center", linespacing=1.08)
            if i < 3:
                self.bg.add_patch(FancyArrowPatch(
                    (x + 61, 493.5), (x + 69, 493.5), arrowstyle="-|>",
                    mutation_scale=15, linewidth=1.5, color=TEAL, zorder=2,
                ))
        self.t(20, 518, c["feature_note"], 23.5, MUTED, w=270, h=24)
        self.bullet(20, 546, c["strategy"], 24)
        self.t(20, 575, c["model_names"], 20, MUTED, w=270, h=10)

    def training_chart(self):
        c = self.c
        self.section(20, 590, "03", c["training"], size=32)
        self.t(20, 618, "51.21%", 50, TEAL, bold=True, latin=True, w=81)
        self.t(21, 640, "1,316 / 2,570", 19.5, MUTED, latin=True, w=82)
        self.t(106, 620, c["recover_label"], 25, INK, w=183, h=25)
        self.t(20, 661, c["tokens_title"], 27 if self.lang == "en" else 25,
               bold=True, w=270)
        self.t(20, 675, c["tokens_note"], 19.5, MUTED, w=181)
        for x, label, color in [(213, c["success"], TEAL), (252, c["failure"], AMBER)]:
            self.rect(x, 677, 4, 3.1, color)
            self.t(x + 6, 675, label, 18.5, MUTED, w=33)
        self.t(20, 688, c["tokens_unit"], 18.5, MUTED, w=80)
        self.t(284, 689, c["box_note"], 17.5, MUTED, w=190, align="right")
        axis = self.ax(42, 704, 242, 64)
        positions = [0.8, 1.4, 2.6, 3.2]
        groups = []
        for workflow, label in [("incident", 0), ("incident", 1), ("ticket", 0), ("ticket", 1)]:
            groups.append(self.data.loc[
                (self.data.workflow == workflow) & (self.data.label == label), "total_tokens"
            ].to_numpy() / 1000)
        boxes = axis.boxplot(
            groups, positions=positions, widths=0.39, patch_artist=True, showfliers=True,
            whis=1.5, manage_ticks=False,
            medianprops={"color": INK, "linewidth": 2.0},
            whiskerprops={"color": MUTED, "linewidth": 1.25},
            capprops={"color": MUTED, "linewidth": 1.25},
            flierprops={"marker": "o", "markersize": 3, "markeredgecolor": MUTED,
                        "markerfacecolor": BG, "markeredgewidth": 0.7},
        )
        for box, color in zip(boxes["boxes"], [TEAL, AMBER, TEAL, AMBER]):
            box.set(facecolor=color, edgecolor=color, alpha=0.78, linewidth=1.6)
        axis.set(xlim=(0.35, 3.65), ylim=(0, 33), yticks=[0, 10, 20, 30])
        axis.set_xticks([1.1, 2.9])
        axis.set_xticklabels([
            c["workflows"][0] + "\nn = 1,003 / 766",
            c["workflows"][1] + "\nn = 1,567 / 206",
        ], fontproperties=self.font(19), linespacing=1.12)
        for tick in axis.get_yticklabels():
            tick.set_fontproperties(self.font(19, latin=True))
        axis.tick_params(axis="both", length=0, pad=5, colors=MUTED)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=LINE, linewidth=0.8)
        for side in ["left", "right", "top", "bottom"]:
            axis.spines[side].set_visible(False)
        self.t(20, 791, c["early_note"], 23 if self.lang == "en" else 22,
               TEAL, bold=True, w=270, h=12)

    def main_results(self):
        c = self.c
        self.section(RIGHT, 190, "03", c["results"])
        self.t(RIGHT, 216, c["test_note"], 21, MUTED, w=270)
        for i, (value, label, note) in enumerate(zip(
            ["83.18%", "9.74%", "−14"], c["kpi_labels"], c["kpi_notes"]
        )):
            x = RIGHT + i * 93
            self.t(x, 235, value, 49, TEAL if i < 2 else AMBER,
                   bold=True, latin=True, w=87)
            self.t(x, 258, label, 25 if self.lang == "en" else 23,
                   INK, bold=True, w=87)
            self.t(x, 272, note, 20.5, MUTED, w=87)
        self.t(RIGHT, 290, c["false_stop"], 24, AMBER, bold=True, w=270)
        self.line(RIGHT, 307, 574, 307)
        self.t(RIGHT, 320, c["auc_title"], 30, bold=True, w=165)
        self.rect(493, 323, 4, 4, TEAL)
        self.t(500, 320, "LR", 21, MUTED, latin=True, w=18)
        self.rect(529, 323, 4, 4, GRAY)
        self.t(536, 320, "KNN", 21, MUTED, latin=True, w=34)
        axis = self.ax(408, 348, 150, 53)
        values = [(0.7749, 0.8413), (0.6099, 0.7422)]
        for row, (knn, lr) in enumerate(values):
            y = 1 - row
            axis.plot([knn, lr], [y, y], color=LINE, linewidth=4, zorder=2)
            axis.scatter([knn], [y], s=105, color=GRAY, edgecolors=BG, linewidths=1, zorder=3)
            axis.scatter([lr], [y], s=130, color=TEAL, edgecolors=BG, linewidths=1, zorder=4)
            axis.text(knn, y - 0.18, f"{knn:.4f}", color=MUTED,
                      fontproperties=self.font(21, latin=True), ha="center", va="top")
            axis.text(lr, y + 0.18, f"{lr:.4f}", color=TEAL,
                      fontproperties=self.font(22, bold=True, latin=True), ha="center", va="bottom")
        axis.set(xlim=(0.55, 0.90), ylim=(-0.55, 1.50), yticks=[],
                 xticks=[0.6, 0.7, 0.8, 0.9])
        for tick in axis.get_xticklabels():
            tick.set_fontproperties(self.font(20, latin=True))
        axis.tick_params(axis="x", length=0, pad=6, colors=MUTED)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color=LINE, linewidth=0.9)
        for side in ["left", "right", "top", "bottom"]:
            axis.spines[side].set_visible(False)
        for row, y in enumerate([357, 383]):
            self.t(RIGHT, y, c["auc_rows"][row], 26, bold=True, w=99)
            self.t(RIGHT, y + 13, c["auc_row_notes"][row], 20, MUTED, w=99)
        self.t(558, 418, c["auc_axis"], 18.5, MUTED, align="right", w=75, latin=True)
        self.t(RIGHT, 430, c["ap_note"], 23, MUTED, w=270)
        self.t(RIGHT, 455, c["trade_title"], 30, bold=True, w=270)
        for x, value, label, color in zip(
            [304, 374, 444, 514], ["855", "−17", "+3", "841"],
            c["equation_labels"], [INK, AMBER, TEAL, INK],
        ):
            self.t(x + 22, 477, value, 40, color, bold=True, latin=True, align="center", w=65)
            self.t(x + 22, 497, label, 19.5, MUTED, align="center", w=63)
        self.t(503, 481, "=", 29, MUTED, latin=True, align="center", w=10)
        self.t(RIGHT, 517, c["cost_line"], 25, bold=True, w=270)
        self.t(RIGHT, 531, c["cost_note"], 21.5, MUTED, w=270)
        self.t(RIGHT, 556, c["timing_title"], 28.5 if self.lang == "en" else 28,
               bold=True, w=270)
        self.t(RIGHT, 573, c["timing_note"], 22, MUTED, w=270)
        colors = [TEAL, AMBER, INK, GRAY]
        left = RIGHT
        for count, color in zip([32, 55, 154, 26], colors):
            width = COL * count / 267
            self.rect(left, 589, width, 16, color, edge=BG, lw=1)
            self.t(left + width / 2, 592, str(count), 24,
                   INK if color == AMBER else WHITE, bold=True, latin=True,
                   align="center", w=width - 1, h=13)
            left += width
        for x, y, label, color in zip(
            [304, 439, 304, 439], [614, 614, 627, 627], c["timing_legend"], colors,
        ):
            self.rect(x, y + 2.4, 4, 4, color)
            self.t(x + 7, y, label, 21.5, MUTED, w=124)
        self.t(RIGHT, 648, c["before_note"], 23.5, TEAL, bold=True, w=270, h=27)

    def discussion_conclusion(self):
        c = self.c
        self.section(RIGHT, 686, "04", c["discussion"], size=32)
        self.bullet(RIGHT, 714, c["discussion_bullets"][0], 23)
        self.bullet(RIGHT, 740, c["discussion_bullets"][1], 23)
        self.rect(RIGHT, 772, COL, 32, PALE, edge=TEAL, lw=2)
        self.rect(RIGHT, 772, 2.2, 32, TEAL)
        self.t(RIGHT + 10, 779, c["conclusion_label"], 19, TEAL, bold=True, w=79)
        self.t(RIGHT + 84, 779, c["conclusion"], 24.5 if self.lang == "en" else 22.5,
               INK, bold=True, w=178, h=24, linespacing=1.1)

    def footer(self):
        self.line(20, 815, 574, 815, INK, 1.2)
        self.t(20, 822, self.c["footer"], 17, MUTED, w=283)
        self.t(574, 820.5, "github.com/Flysheep1223/agentsentinel-lite", 20, INK,
               align="right", latin=True, w=265, url=URL)
        self.t(20, 832, self.c["footer_sub"], 16, MUTED, w=554, url=REPORT_URL)

    def build(self):
        self.header()
        self.introduction_methods()
        self.training_chart()
        self.main_results()
        self.discussion_conclusion()
        self.footer()
        self.fig.canvas.draw()
        return self.check_layout()

    def check_layout(self):
        renderer = self.fig.canvas.get_renderer()
        errors = []
        bounds = []
        factor = 25.4 / self.fig.dpi
        for artist, w, h, x, y in self.text_bounds:
            box = artist.get_window_extent(renderer)
            actual_w, actual_h = box.width * factor, box.height * factor
            description = artist.get_text().replace("\n", " / ")
            if w is not None and actual_w > w + 0.5:
                errors.append(f"Width {actual_w:.1f} > {w}: {description}")
            if h is not None and actual_h > h + 0.5:
                errors.append(f"Height {actual_h:.1f} > {h}: {description}")
            if box.x0 < -1 or box.x1 > self.fig.bbox.width + 1 or box.y0 < -1:
                errors.append(f"Text outside page: {description}")
            bounds.append({
                "text": artist.get_text(), "x_mm": x, "y_mm": y,
                "width_mm": round(actual_w, 2), "height_mm": round(actual_h, 2),
                "font_size_pt": artist.get_fontsize(),
            })
        if errors:
            raise ValueError(f"{self.lang} layout checks failed:\n" + "\n".join(errors))
        return bounds

    def export(self, preview_only=False):
        stem = OUT / f"AgentSentinel_A1_{self.lang.upper()}"
        preview = Path(str(stem) + "_preview.png")
        self.fig.savefig(preview, dpi=90, facecolor=BG)
        files = [preview]
        if not preview_only:
            pdf, svg, png = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
            self.fig.savefig(
                pdf, facecolor=BG, dpi=300,
                metadata={
                    "Title": f"AgentSentinel-Lite — {self.c['subtitle'].replace(chr(10), ' ')}",
                    "Subject": "Frozen main 22-feature experiment; offline early stopping evaluation",
                    "Keywords": "A1, research poster, LLM agent, failure prediction, offline replay",
                    "Creator": "Matplotlib; decorative illustration generated with Seedream 5.5",
                },
            )
            self.fig.savefig(svg, facecolor=BG, dpi=300,
                             metadata={"Title": "AgentSentinel-Lite", "Description": "Editable A1 poster"})
            self.fig.savefig(png, dpi=300, facecolor=BG)
            files.extend([pdf, svg, png])
        (OUT / f"poster_text_{self.lang}.txt").write_text("\n\n".join(self.body_texts) + "\n")
        plt.close(self.fig)
        return files


def check_exports(files):
    results = []
    for path in files:
        result = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix == ".png":
            with Image.open(path) as image:
                result["pixels"] = list(image.size)
                result["dpi"] = list(image.info.get("dpi", []))
                assert abs(image.width / image.height - W / H) < 0.0002
                if "preview" not in path.name:
                    expected = (int(W / 25.4 * 300), int(H / 25.4 * 300))
                    assert image.size == expected, (image.size, expected)
        elif path.suffix == ".pdf":
            content = path.read_bytes()
            assert b"/FontFile2" in content, "PDF must embed TrueType fonts."
            box = re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", content)
            assert box, "Missing PDF page bounds"
            numbers = [float(value) for value in box.group(1).split()]
            result["page_mm"] = [round(numbers[2] * 25.4 / 72, 3),
                                 round(numbers[3] * 25.4 / 72, 3)]
            assert result["page_mm"] == [W, H]
            assert len(re.findall(rb"/Type\s*/Page\b", content)) == 1
            result["embedded_fonts"] = True
        elif path.suffix == ".svg":
            text = path.read_text()
            assert "<text" in text, "Keep SVG text editable."
            result["editable_text_elements"] = text.count("<text")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--language", choices=["en", "zh", "both"], default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "savefig.bbox": None,
        "font.family": "Arial",
        "font.size": 24,
    })
    sources = [REPORT, TRAIN, EARLY, ART, OUT / "visual_v1/metadata.json"]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    data, evidence = verify_inputs()
    files = []
    layout_checks = {}
    for language in (["en", "zh"] if args.language == "both" else [args.language]):
        poster = Poster(language, data)
        layout_checks[language] = poster.build()
        exported = poster.export(args.preview_only)
        files.extend(exported)
        print(f"{language.upper()}: rendered {len(exported)} files; all text boxes fit.", flush=True)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    assert before == after, "Source inputs changed during build."
    outputs = check_exports(files)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "page_mm": [W, H],
        "print_dpi": 300,
        "preview_only": args.preview_only,
        "metrics": METRICS,
        "data_checks": evidence,
        "source_sha256": before,
        "source_inputs_unchanged": True,
        "script_sha256": sha256(Path(__file__)),
        "fonts": {"latin_regular": REGULAR, "latin_bold": BOLD, "chinese": CJK},
        "chart_method": "Raw training token distributions; 1.5 IQR whiskers; all fliers retained.",
        "chart_sample_labels": "Counts below each workflow are success / failure.",
        "auc_note": "Full-prefix observations are correlated within tasks; step 3 has one prefix per task.",
        "timing_denominators": "Bar: 267 alerted failures; pre-violation coverage: 268 violation-bearing tasks.",
        "generated_visual": "Conceptual decoration only, not data or an experimental result.",
        "outputs": outputs,
    }
    name = "preview_manifest.json" if args.preview_only else "poster_manifest.json"
    (OUT / name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUT / "layout_checks.json").write_text(json.dumps(layout_checks, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"outputs": outputs, "source_inputs_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

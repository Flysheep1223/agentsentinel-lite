"""Build a proposal-led, figure-first EDA research poster.

The narrative follows the proposal's three research questions and uses the completed
6,000-run study as evidence. EDA figures carry the argument; the frozen prediction
experiment and stopping evaluation answer what those signals can and cannot do.

Usage:
    .venv/bin/python scripts/make_eda_story_poster.py --preview-only --language zh
    .venv/bin/python scripts/make_eda_story_poster.py

Existing editions and all frozen study inputs remain untouched. No image-generation
request is made; the small existing Seedream illustration is decorative only.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_complete_poster import (
    ART,
    BG,
    COL,
    DARK_AMBER,
    FIG_DIR,
    FIGURES,
    LEFT,
    LINE,
    MUTED,
    PALE,
    PALE_BLUE,
    RIGHT,
    ROOT,
    TEAL,
    CompletePoster,
    check_exports,
    verify_supplementary_evidence,
)
from make_eda_poster import eda_evidence
from make_research_poster import INK, REPORT, TRAIN, URL, sha256, verify_inputs
from PIL import Image

OUT = ROOT / "docs/poster/agentsentinel_eda_story"
PROPOSAL = ROOT / "AgentSentinel_Lite_Proposal.md"
REPORT_URL = URL + "/blob/main/docs/RESEARCH_REPORT_6000_FINAL.md"
PROPOSAL_URL = URL + "/blob/main/AgentSentinel_Lite_Proposal.md"

COPY = {
    "zh": {
        "kicker": "探索性数据分析（EDA） · LLM AGENT RELIABILITY",
        "headline": "工具报错 ≠ 任务失败｜从运行轨迹理解风险、恢复与停止代价",
        "subnote": "以 Proposal 的三个研究问题为主线 · 6,000 次真实模型运行 · 训练集 EDA · 冻结测试",
        "stats": [("6,000", "采集运行"), ("5,894", "有效任务"),
                  ("47,184", "可决策前缀"), ("3,542", "训练集 EDA")],
        "design": "01｜研究设计：在第 t 步，只能根据已经发生的事判断最终失败",
        "data_title": "实际数据与严格切分",
        "data": (
            "20 模板 × 20 实例 × 5 条件 × 3 重复 = 6,000\n"
            "工单 + 服务事件；正常/超时/限流/Schema/部分响应\n"
            "训练/验证/测试 = 3,542 / 1,176 / 1,176；父家族 6/2/2\n"
            "同一模板及全部前缀不跨区；106 条无效运行只作质量审计"
        ),
        "prefix_title": "前缀预测，而非事后诊断",
        "formula": r"$p_{s,t}=P(Y_s=1 \mid X_{s,\leq t})$",
        "prefix": (
            "工具返回后才评分；只使用当前及过去事件。\n"
            "故障配置、未来步骤和验证器答案不作输入。"
        ),
        "eval_title": "预测与停止分开评价",
        "evaluation": (
            "错误/重试/重复、Token/费用/时间、工具与写入阶段\n"
            "LR + KNN + 固定规则；验证集限制成功任务误停预算\n"
            "同时报告失败召回、告警时机、费用节省和成功损失\n"
            "本批没有在线干预；所有停止收益均来自离线回放"
        ),
        "eda": "02｜EDA：五张图回答 Proposal 的核心问题",
        "scope": (
            "仅训练分区：3,542 个有效任务 / 27,367 条前缀；验证与测试不参与 EDA。"
            "补充图表属于事后探索，未用于改写冻结主实验；描述性关联不作因果解释。"
        ),
        "cards": [
            {
                "label": "RQ2 · 故障与任务结构",
                "title": "故障影响取决于工作流",
                "key": "45.9% / 8.7%",
                "takeaway": (
                    "部分响应：服务事件 45.9%，工单仅 8.7%。正常条件仍有 170 次失败；"
                    "模板失败率 0.67%–90.41%，所以必须分面并按父家族切分。"
                ),
            },
            {
                "label": "RQ1 + RQ2 · 错误恢复与完成时点",
                "title": "错误可恢复，成功也可能很晚",
                "key": "51.21%",
                "takeaway": (
                    "51.21% 的成功任务出现显式错误；读取错误恢复 4,031/5,926（68.02%）。"
                    "77 个成功直到第 12 步才首次达标：报错不能直接当作停止理由。"
                ),
            },
            {
                "label": "RQ1 · 完整轨迹与早期资源对照",
                "title": "完整资源差异 ≠ 早期可分",
                "key": "STEP 3",
                "takeaway": (
                    "完整 Token 中位数明显分离；但第 3 步均值：服务事件成功/失败 4,277/4,226，"
                    "工单 4,246/4,289。事后总消耗不能伪装成早期信号。"
                ),
            },
            {
                "label": "RQ1 · 固定第 3 步的错误 × 重复",
                "title": "简单阈值并不稳定",
                "key": "先看 n",
                "takeaway": (
                    "服务事件“0 错误”格中，有重复反而失败更少（15.8% vs 42.8%）；"
                    "工单一个 50% 格子仅 n=2。不能据此声称重复有保护作用。"
                ),
            },
            {
                "label": "RQ1 · 跨步趋势与样本组成",
                "title": "后期曲线已不是同一批任务",
                "key": "1,003 → 84",
                "takeaway": (
                    "服务事件成功任务的可决策样本到第 11 步仅剩 84，失败仍有 705。"
                    "后期成功曲线错误更多，不表示错误有益；必须同时看 surviving n。"
                ),
            },
        ],
        "synthesis_label": "EDA SYNTHESIS · 从发现到实验约束",
        "synthesis_title": "EDA 不是装饰，而是建模规则",
        "corr_title": "固定第 3 步 Spearman：信息高度重叠",
        "correlations": [("重试 ↔ 重复", "0.981"), ("Token ↔ 费用", "0.884"),
                         ("Token ↔ Reasoning", "0.894")],
        "implications": [
            ("任务异质性", "父家族分组 CV + 逐家族报告"),
            ("晚恢复 / 曾经达标", "告警、误停与成功损失分开"),
            ("早期弱分离 / 高相关", "联合特征 + 固定步评价 + 消融"),
        ],
        "quality": (
            "质量底线：有效分析字段无缺失、负值或非有限值；62 条时间与 31 条 Token 外点回查后保留。\n"
            "50/10 并发的无效率 8.00%/1.03%，但与日期和供应方状态混杂。"
        ),
        "model": "03｜从 EDA 到冻结主实验：风险可预测，无损停止未实现",
        "model_panels": [
            ("冻结模型与公平比较",
             "22 项可观察特征；每任务最多 6 条前缀\n"
             "LR C=1；KNN k=9；固定/步数/永不停规则\n"
             "19,094 条训练前缀；3 折父家族 CV\n"
             "验证集 5% 误停预算选阈值，测试只评一次"),
            ("预测质量 ≠ 违规预防", ""),
            ("费用节省 ≠ 无损收益", ""),
        ],
        "auc": "ROC AUC：全前缀 0.8413｜第 3 步 0.7422",
        "recall": "83.18%", "recall_label": "失败召回 267/321",
        "pre": "11.94%", "pre_label": "首次违规前覆盖 32/268",
        "timing": "267 次失败告警中，154 次晚于首次违规；55 次同步，32 次提前。",
        "saving": "9.74%", "saving_label": "估算费用节省",
        "false_stop": "1.99%", "false_stop_label": "误停 17/855",
        "net": "−14", "net_label": "严格成功 855 → 841",
        "baseline": "同一 5% 验证预算下，KNN 与规则均选择不停止；测试仅 2 个父家族。",
        "extension": (
            "后续 48 特征探索（复用已见测试集，非新盲测）：费用节省 5.39%→5.83%，误停 4→3，"
            "第 6 步前识别 14→60；严格成功仍低于原基线，不能替代冻结主结论。"
        ),
        "conclusion": (
            "EDA 结论｜错误不是标签，晚期差异不是早期信号，相关不是因果。\n"
            "研究结论｜运行信号适合风险监测并减少部分后续浪费；当前不能承诺违规前预防、无损自动停止或在线收益。\n"
            "汇报主线｜先理解失败与恢复，再评价预测；先保护可恢复的成功任务，再讨论节省。"
        ),
        "footer": "Proposal：AgentSentinel_Lite_Proposal.md｜结果：RESEARCH_REPORT_6000_FINAL.md（2026-11-06）",
        "footer_sub": "真实 LLM + 本地模拟工具环境 · 五张数据图均来自训练集 · Seedream 5.5 仅用于页首概念装饰，研究图与数字不由 AI 生成",
    },
    "en": {
        "kicker": "EXPLORATORY DATA ANALYSIS · LLM AGENT RELIABILITY",
        "headline": "Tool errors ≠ task failure | Reading risk, recovery and stopping costs from traces",
        "subnote": "Three proposal research questions · 6,000 real-model runs · training-only EDA · frozen test",
        "stats": [("6,000", "COLLECTED RUNS"), ("5,894", "VALID TASKS"),
                  ("47,184", "ELIGIBLE PREFIXES"), ("3,542", "TRAINING EDA")],
        "design": "01 | DESIGN: at step t, predict final failure using only observed events",
        "data_title": "Completed data and strict split",
        "data": (
            "20 templates × 20 instances × 5 conditions × 3 repeats = 6,000\n"
            "Ticket + incident; normal/timeout/rate-limit/schema/partial response\n"
            "Train/validation/test = 3,542/1,176/1,176; parent families 6/2/2\n"
            "No template or prefix crosses splits; 106 invalid runs audited separately"
        ),
        "prefix_title": "Prefix prediction, not hindsight",
        "formula": r"$p_{s,t}=P(Y_s=1 \mid X_{s,\leq t})$",
        "prefix": (
            "Score after a tool return; use current and past events only.\n"
            "Fault settings, future steps and verifier answers are never inputs."
        ),
        "eval_title": "Prediction and stopping are distinct",
        "evaluation": (
            "Errors/retries/repeats, tokens/cost/time, tool and write context\n"
            "LR + KNN + fixed rules; validation false-stop budget\n"
            "Report recall, timing, savings and success loss together\n"
            "No online intervention; every stopping gain is offline replay"
        ),
        "eda": "02 | EDA: five figures answer the proposal's core questions",
        "scope": (
            "TRAIN ONLY: 3,542 valid tasks / 27,367 prefixes; no validation/test EDA. "
            "Supplementary figures are post-hoc exploration and did not alter the frozen main experiment; associations are descriptive."
        ),
        "cards": [
            {
                "label": "RQ2 · FAULTS AND TASK STRUCTURE",
                "title": "Fault impact depends on workflow",
                "key": "45.9% / 8.7%",
                "takeaway": (
                    "Partial response: 45.9% incident vs 8.7% ticket failure. Normal conditions still have 170 failures; "
                    "template failure spans 0.67%–90.41%. Facet and split by parent family."
                ),
            },
            {
                "label": "RQ1 + RQ2 · RECOVERY AND COMPLETION",
                "title": "Errors recover; success can be late",
                "key": "51.21%",
                "takeaway": (
                    "51.21% of successes show explicit errors; 4,031/5,926 read errors recover (68.02%). "
                    "77 successes first complete at step 12: an error alone is not a stopping rule."
                ),
            },
            {
                "label": "RQ1 · FULL-RUN VERSUS EARLY RESOURCES",
                "title": "Full-run gaps are not early signals",
                "key": "STEP 3",
                "takeaway": (
                    "Full-run token medians separate, but step-3 means do not: incident success/failure 4,277/4,226; "
                    "ticket 4,246/4,289. Final consumption cannot stand in for early information."
                ),
            },
            {
                "label": "RQ1 · STEP-3 ERRORS × REPETITION",
                "title": "No stable simple threshold",
                "key": "CHECK n",
                "takeaway": (
                    "Within incident zero-error cells, repetition coincides with lower failure (15.8% vs 42.8%); "
                    "one 50% ticket cell has n=2. This does not show repetition is protective."
                ),
            },
            {
                "label": "RQ1 · CROSS-STEP COHORT CHANGE",
                "title": "Later curves contain different tasks",
                "key": "1,003 → 84",
                "takeaway": (
                    "Eligible incident successes fall to 84 at step 11 while 705 failures remain. "
                    "Higher late errors among successes are not beneficial effects; always report surviving n."
                ),
            },
        ],
        "synthesis_label": "EDA SYNTHESIS · FROM FINDINGS TO CONSTRAINTS",
        "synthesis_title": "EDA sets the modeling rules",
        "corr_title": "Step-3 Spearman: overlapping information",
        "correlations": [("Retry ↔ repeat", "0.981"), ("Tokens ↔ cost", "0.884"),
                         ("Tokens ↔ reasoning", "0.894")],
        "implications": [
            ("Task heterogeneity", "parent-family CV + family results"),
            ("Late recovery / completion", "separate alerts, false stops, success loss"),
            ("Weak early split / correlation", "joint features + fixed-step tests + ablation"),
        ],
        "quality": (
            "Quality floor: no missing, negative or non-finite values in valid analysis fields; 62 time and 31 token outliers audited and retained.\n"
            "Invalid rates at 50/10 workers: 8.00%/1.03%; date and provider state are confounded."
        ),
        "model": "03 | FROM EDA TO THE FROZEN TEST: risk is predictable; loss-free stopping is not",
        "model_panels": [
            ("Frozen model and fair comparison",
             "22 observable features; at most 6 prefixes/task\n"
             "LR C=1; KNN k=9; fixed/step/never-stop rules\n"
             "19,094 training prefixes; 3-fold parent-family CV\n"
             "5% validation false-stop budget; one frozen test"),
            ("Prediction is not prevention", ""),
            ("Savings are not loss-free gains", ""),
        ],
        "auc": "ROC AUC: all prefixes 0.8413 | step 3 0.7422",
        "recall": "83.18%", "recall_label": "Failure recall 267/321",
        "pre": "11.94%", "pre_label": "Pre-violation coverage 32/268",
        "timing": "Of 267 failure alerts, 154 occur after first violation; 55 at the same step and 32 before.",
        "saving": "9.74%", "saving_label": "Estimated cost saved",
        "false_stop": "1.99%", "false_stop_label": "False stops 17/855",
        "net": "−14", "net_label": "Strict successes 855 → 841",
        "baseline": "Under the same 5% validation budget, KNN and rules select no-stop; the test has only two parent families.",
        "extension": (
            "Exploratory 48-feature extension (seen test set, not a new blind test): savings 5.39%→5.83%; false stops 4→3; "
            "detections by step 6: 14→60. Strict successes remain below baseline; the frozen conclusion stands."
        ),
        "conclusion": (
            "EDA | Errors are not labels, late gaps are not early signals, and correlation is not causation.\n"
            "Study | Runtime signals support risk monitoring and reduce some later waste; violation prevention, loss-free stopping and online gains remain unproven.\n"
            "Presentation | Understand failure and recovery first; protect recoverable successes before claiming execution savings."
        ),
        "footer": "Proposal: AgentSentinel_Lite_Proposal.md | Results: RESEARCH_REPORT_6000_FINAL.md (6 Nov 2026)",
        "footer_sub": "Real LLM + local simulated tools · EDA figures use training data only · Seedream 5.5 created header decoration only, never study figures or numbers",
    },
}


class StoryPoster(CompletePoster):
    """A1/A0 canvas with proposal-led evidence cards instead of report-like prose."""

    def __init__(self, lang, data, evidence, paper="a1"):
        super().__init__(lang, data, evidence, paper)
        self.c = COPY[lang]

    def header(self):
        with Image.open(ART) as source:
            art = source.convert("RGB").crop((420, 300, 2512, 1390))
        pixels = np.asarray(art).astype(np.float32)
        height, width = pixels.shape[:2]
        horizontal = np.clip(np.arange(width) / (width * 0.24), 0, 1)
        vertical = np.minimum(
            np.clip(np.arange(height) / (height * 0.15), 0, 1),
            np.clip((height - 1 - np.arange(height)) / (height * 0.16), 0, 1),
        )
        alpha = vertical[:, None] * horizontal[None, :]
        blended = (pixels * alpha[:, :, None] + 255 * (1 - alpha[:, :, None])).astype(np.uint8)
        axis = self.ax(493, 8, 85, 49)
        axis.imshow(blended, aspect="auto")
        axis.set_axis_off()

        self.rect(16, 7, 28, 1.8, TEAL)
        self.t(16, 12, self.c["kicker"], 11.8, MUTED, w=465, h=5)
        self.t(16, 20, "AgentSentinel-Lite", 47, INK,
               bold=True, latin=True, w=466, h=21)
        self.t(16, 45, self.c["headline"], 20.0 if self.lang == "zh" else 16.3,
               TEAL, bold=True, w=465, h=11)
        self.t(16, 58, self.c["subnote"], 11.5, MUTED, w=465, h=5)
        for i, (value, label) in enumerate(self.c["stats"]):
            x = 16 + i * 142
            self.rect(x, 68, 136, 24, PALE if i != 3 else PALE_BLUE)
            self.rect(x, 68, 1.8, 24, TEAL if i != 3 else "#172D33")
            self.t(x + 7, 70.5, value, 23, bold=True, latin=True, w=72, h=10)
            self.t(x + 79, 75.2, label, 11.3 if self.lang == "zh" else 9.4,
                   MUTED, bold=(i == 3), w=52, h=8)

    def wide_section(self, y, title, emphasis=False):
        if emphasis:
            self.rect(16, y - 2, 562, 17.5, PALE)
            self.rect(16, y - 2, 2.4, 17.5, TEAL)
        self.t(23 if emphasis else 16, y, title, 20.0 if self.lang == "zh" else 17.4,
               bold=True, w=552, h=12)
        self.line(16, y + 16, 578, y + 16, "#172D33", 0.8)

    def panel(self, x, y, w, title, body, body_size=11.5):
        self.rect(x, y, w, 46, "#FAFCFB", edge=LINE, lw=0.55)
        self.rect(x, y, 2.0, 46, TEAL)
        self.t(x + 6, y + 4, title, 13.2 if self.lang == "zh" else 11.8,
               bold=True, w=w - 12, h=7)
        self.t(x + 6, y + 13, body, body_size, w=w - 12, h=29, linespacing=1.08)

    def study_design(self):
        self.wide_section(100, self.c["design"])
        self.panel(16, 121, 178, self.c["data_title"], self.c["data"],
                   11.1 if self.lang == "zh" else 9.5)
        self.rect(204, 121, 178, 46, "#FAFCFB", edge=LINE, lw=0.55)
        self.rect(204, 121, 2, 46, TEAL)
        self.t(210, 125, self.c["prefix_title"], 13.2 if self.lang == "zh" else 11.8,
               bold=True, w=166, h=7)
        self.t(210, 135, self.c["formula"], 17.0, TEAL, bold=True, latin=True, w=166, h=8)
        self.t(210, 146, self.c["prefix"], 11.1 if self.lang == "zh" else 9.8,
               w=166, h=16, linespacing=1.08)
        self.panel(392, 121, 186, self.c["eval_title"], self.c["evaluation"],
                   10.9 if self.lang == "zh" else 9.3)

    def card_base(self, x, y, w, h, card):
        self.rect(x, y, w, h, "#FFFFFF", edge=LINE, lw=0.65)
        self.rect(x, y, w, 2.0, TEAL)
        self.t(x + 6, y + 4, card["label"], 10.3 if self.lang == "zh" else 8.9,
               MUTED, bold=True, w=w - 12, h=5)
        self.t(x + 6, y + 11, card["title"], 17.1 if self.lang == "zh" else 14.7,
               bold=True, w=181 if self.lang == "zh" else 170, h=11)
        self.t(x + w - 6, y + 9.5, card["key"], 18.0 if self.lang == "zh" else 15.5,
               TEAL, bold=True, align="right",
               latin=self.lang == "en" or all(ord(char) < 128 for char in card["key"]),
               w=88, h=10)

    def takeaway(self, x, y, w, text, h=17, size=None):
        self.rect(x + 4, y, w - 8, h, PALE)
        self.t(x + 8, y + 2.0, text,
               size or (10.9 if self.lang == "zh" else 9.2),
               w=w - 16, h=h - 3, linespacing=1.06)

    def eda(self):
        self.wide_section(174, self.c["eda"], emphasis=True)
        self.t(16, 195, self.c["scope"], 10.9 if self.lang == "zh" else 9.2,
               MUTED, w=562, h=10, linespacing=1.05)
        cards = self.c["cards"]

        # RQ2: assigned condition by workflow.
        self.card_base(LEFT, 208, COL, 143, cards[0])
        self.image("conditions", LEFT + 20, 235, 235)
        self.takeaway(LEFT, 337, COL, cards[0]["takeaway"], h=12,
                      size=10.4 if self.lang == "zh" else 8.6)

        # Recovery and late completion: large completion-time figure.
        self.card_base(RIGHT, 208, COL, 143, cards[1])
        self.image("completion", RIGHT + 12.5, 235, 250)
        self.takeaway(RIGHT, 331, COL, cards[1]["takeaway"], h=18,
                      size=10.6 if self.lang == "zh" else 8.9)

        # Full-run resource separation versus fixed-step means.
        self.card_base(LEFT, 357, COL, 164, cards[2])
        self.image("tokens", LEFT + 12.5, 384, 250)
        self.takeaway(LEFT, 502, COL, cards[2]["takeaway"], h=17,
                      size=10.7 if self.lang == "zh" else 8.9)

        # Fixed-step joint failure-rate heatmap.
        self.card_base(RIGHT, 357, COL, 164, cards[3])
        self.image("step3", RIGHT + 32.5, 384, 210)
        self.takeaway(RIGHT, 502, COL, cards[3]["takeaway"], h=17,
                      size=10.7 if self.lang == "zh" else 8.9)

        # Survivor bias and cohort sizes.
        self.card_base(LEFT, 527, COL, 139, cards[4])
        self.image("survivors", LEFT + 12.5, 554, 250)
        self.takeaway(LEFT, 648, COL, cards[4]["takeaway"], h=16,
                      size=10.5 if self.lang == "zh" else 8.8)

        self.synthesis_card(RIGHT, 527, COL, 139)

    def synthesis_card(self, x, y, w, h):
        self.rect(x, y, w, h, "#F7FAF9", edge=TEAL, lw=0.8)
        self.rect(x, y, w, 2, TEAL)
        self.t(x + 6, y + 4, self.c["synthesis_label"], 10.0 if self.lang == "zh" else 8.5,
               MUTED, bold=True, w=w - 12, h=5)
        self.t(x + 6, y + 11, self.c["synthesis_title"], 17.2 if self.lang == "zh" else 15.2,
               bold=True, w=w - 12, h=10)
        self.t(x + 6, y + 25, self.c["corr_title"], 11.2 if self.lang == "zh" else 9.7,
               MUTED, w=w - 12, h=6)
        for i, (label, value) in enumerate(self.c["correlations"]):
            cx = x + 6 + i * 89
            self.rect(cx, y + 33, 83, 21, PALE_BLUE)
            self.t(cx + 41.5, y + 34, value, 17.5, bold=True, latin=True,
                   align="center", w=79, h=8)
            self.t(cx + 41.5, y + 44.5, label, 9.8 if self.lang == "zh" else 8.2,
                   MUTED, align="center", w=79, h=6)
        for i, (cause, effect) in enumerate(self.c["implications"]):
            cy = y + 60 + i * 16
            self.rect(x + 6, cy, 88 if self.lang == "zh" else 92, 13, PALE)
            self.t(x + 10, cy + 2.5, cause, 10.6 if self.lang == "zh" else 8.8,
                   bold=True, w=82 if self.lang == "zh" else 86, h=8)
            self.t(x + 101, cy + 1.2, "→", 14, TEAL, bold=True, w=12, h=8)
            self.t(x + 115, cy + 2.0, effect, 10.5 if self.lang == "zh" else 8.5,
                   w=w - 121, h=9)
        self.t(x + 6, y + 110, self.c["quality"], 9.9 if self.lang == "zh" else 8.2,
               MUTED, w=w - 12, h=24, linespacing=1.04)

    def model(self):
        self.wide_section(673, self.c["model"], emphasis=True)
        xs = [16, 204, 392]
        widths = [178, 178, 186]
        for x, w, (title, _) in zip(xs, widths, self.c["model_panels"]):
            self.t(x, 695, title, 13.2 if self.lang == "zh" else 11.4,
                   bold=True, w=w, h=7)
        self.line(198, 694, 198, 753, LINE, 0.6)
        self.line(386, 694, 386, 753, LINE, 0.6)
        self.t(16, 705, self.c["model_panels"][0][1], 10.9 if self.lang == "zh" else 9.2,
               w=178, h=34, linespacing=1.07)
        self.rect(16, 741, 178, 13, PALE_BLUE)
        self.t(22, 744, self.c["auc"], 11.0 if self.lang == "zh" else 9.5,
               bold=True, w=166, h=7)

        self.t(204, 704, self.c["recall"], 23, TEAL, bold=True, latin=True, w=84, h=10)
        self.t(292, 704, self.c["pre"], 23, TEAL, bold=True, latin=True,
               align="right", w=90, h=10)
        self.t(204, 716, self.c["recall_label"], 10.0 if self.lang == "zh" else 8.5,
               MUTED, w=84, h=7)
        self.t(292, 716, self.c["pre_label"], 9.7 if self.lang == "zh" else 8.0,
               MUTED, align="right", w=90, h=9)
        self.t(204, 731, self.c["timing"], 10.4 if self.lang == "zh" else 8.7,
               w=178, h=21, linespacing=1.06)

        metric_data = [
            (self.c["saving"], self.c["saving_label"], TEAL),
            (self.c["false_stop"], self.c["false_stop_label"], TEAL),
            (self.c["net"], self.c["net_label"], DARK_AMBER),
        ]
        for i, (value, label, color) in enumerate(metric_data):
            x = 392 + i * 63
            self.t(x + 29, 703, value, 21.5, color, bold=True, latin=True,
                   align="center", w=58, h=10)
            self.t(x + 29, 715.5, label, 9.2 if self.lang == "zh" else 7.6,
                   MUTED, align="center", w=58, h=10)
        self.t(392, 731, self.c["baseline"], 10.1 if self.lang == "zh" else 8.5,
               w=186, h=21, linespacing=1.05)

        self.rect(16, 758, 562, 15, "#F5F7F7")
        self.t(22, 760, self.c["extension"], 10.2 if self.lang == "zh" else 8.6,
               MUTED, w=550, h=11, linespacing=1.04)
        self.rect(16, 779, 562, 32, PALE, edge=TEAL, lw=1.0)
        self.rect(16, 779, 2.5, 32, TEAL)
        self.t(25, 783, self.c["conclusion"], 12.5 if self.lang == "zh" else 10.7,
               bold=True, w=544, h=23, linespacing=1.13)

    def footer(self):
        self.line(16, 820, 578, 820, "#172D33", 0.75)
        self.t(16, 824, self.c["footer"], 9.4, MUTED, w=455, h=5, url=REPORT_URL)
        self.t(578, 824, "github.com/Flysheep1223/agentsentinel-lite", 9.5,
               align="right", latin=True, w=210, h=5, url=URL)
        self.t(16, 832, self.c["footer_sub"], 9.0, MUTED, w=562, h=5)

    def build(self):
        self.header()
        self.study_design()
        self.eda()
        self.model()
        self.footer()
        self.fig.canvas.draw()
        return self.check_layout()

    def export(self, preview_only=False):
        paper = self.paper.upper()
        stem = OUT / f"AgentSentinel_EDA_Story_{paper}_{self.lang.upper()}"
        files = []
        if self.paper == "a1":
            preview = Path(str(stem) + "_preview.png")
            self.fig.savefig(preview, dpi=90, facecolor=BG)
            files.append(preview)
            (OUT / f"poster_text_{self.lang}.txt").write_text(
                "\n\n".join(self.body_texts) + "\n"
            )
        if not preview_only:
            suffixes = [".pdf", ".svg", ".png"] if self.paper == "a1" else [".pdf", ".svg"]
            for suffix in suffixes:
                path = stem.with_suffix(suffix)
                kwargs = {"dpi": 300, "facecolor": BG}
                if suffix == ".pdf":
                    kwargs["metadata"] = {
                        "Title": "AgentSentinel-Lite — EDA evidence and early-stopping trade-offs",
                        "Subject": "Proposal-led EDA narrative using the completed 6,000-run study",
                        "Keywords": "EDA, LLM agent, recovery, failure prediction, early stopping",
                        "Creator": "Matplotlib; existing Seedream art used only as header decoration",
                    }
                elif suffix == ".svg":
                    kwargs["metadata"] = {
                        "Title": "AgentSentinel-Lite — EDA story poster",
                        "Description": f"Editable {paper} poster with proposal-led EDA evidence cards",
                    }
                self.fig.savefig(path, **kwargs)
                files.append(path)
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
        "font.family": "Arial", "font.size": 16.5,
    })
    proposal = PROPOSAL.read_text()
    for token in ["Research Questions", "EDA and Data Visualization Plan", "错误中断率", "首次违反任务约束"]:
        assert token in proposal, f"Proposal narrative element missing: {token}"
    sources = [
        PROPOSAL, REPORT, TRAIN, ROOT / "artifacts/expanded_eda_v2/fixed_step_summary.csv",
        ART, ART.parent / "metadata.json",
        *[FIG_DIR / name for name, _ in FIGURES.values()],
    ]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    data, distribution_checks = verify_inputs()
    evidence = eda_evidence(data)
    supplementary = verify_supplementary_evidence()
    files, layout = [], {}
    languages = ["en", "zh"] if args.language == "both" else [args.language]
    papers = ["a1"] if args.preview_only else ["a1", "a0"]
    for language in languages:
        for paper in papers:
            poster = StoryPoster(language, data, evidence, paper)
            layout[f"{language}_{paper}"] = poster.build()
            exported = poster.export(args.preview_only)
            files.extend(exported)
            print(f"{language.upper()} {paper.upper()}: rendered {len(exported)} files; text boxes fit.", flush=True)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    assert before == after, "Source inputs changed during build."
    outputs = check_exports(files)
    text_counts = {}
    for language in languages:
        text = (OUT / f"poster_text_{language}.txt").read_text()
        text_counts[language] = {
            "whitespace_delimited_words": len(text.split()),
            "non_whitespace_characters": len(re.sub(r"\s", "", text)),
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": "Proposal-led, figure-first EDA story poster",
        "narrative": [
            "Tool errors are not task-failure labels",
            "Full-run resource gaps are not evidence of early separability",
            "Recovery and late completion constrain early stopping",
            "Changing surviving cohorts prevent naive trend interpretation",
            "Prediction, violation prevention and loss-free stopping are distinct outcomes",
        ],
        "paper_mm": {"A1": [594, 841], "A0": [841, 1189]},
        "print_dpi_a1_png": 300,
        "preview_only": args.preview_only,
        "source_inputs_unchanged": True,
        "source_sha256": before,
        "script_sha256": sha256(Path(__file__)),
        "eda_evidence": evidence,
        "distribution_checks": distribution_checks,
        "supplementary_checks": supplementary,
        "scope": {
            "eda": "Training split only; five original figures plus an EDA-to-model synthesis card",
            "main_experiment": "Frozen 22-feature LR; independent test; strict offline replay",
            "extension": "48 features; seen-test exploratory reevaluation, not a new blind test",
            "online_evaluation": "Not conducted in this batch; no online stopping gains claimed",
            "concept_art": "Existing Seedream 5.5 header decoration only; no new generation call",
        },
        "text_counts": text_counts,
        "outputs": outputs,
    }
    name = "preview_manifest.json" if args.preview_only else "poster_manifest.json"
    (OUT / name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "layout_checks.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "text_counts": text_counts,
                      "source_inputs_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

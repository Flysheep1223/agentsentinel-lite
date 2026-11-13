"""Build dense bilingual A1/A0 posters covering the complete AgentSentinel study.

The EDA occupies most of the layout and embeds six reviewed report figures without
redrawing them. A1 PNG is exported at 300 dpi; A0 PDF/SVG are enlarged print/vector
editions. Seedream art is reused only as a small conceptual header illustration.

Usage:
    .venv/bin/python scripts/make_complete_poster.py --preview-only --language zh
    .venv/bin/python scripts/make_complete_poster.py

No image-generation request is made. All source data, figures, reports and prior
poster editions are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_eda_poster import eda_evidence
from make_research_poster import (
    ART,
    BOLD,
    CJK,
    INK,
    LINE,
    MUTED,
    REGULAR,
    REPORT,
    ROOT,
    TEAL,
    TRAIN,
    URL,
    sha256,
    verify_inputs,
)
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

OUT = ROOT / "docs/poster/agentsentinel_complete_eda"
FIG_DIR = ROOT / "docs/figures/research_report_6000"
EARLY_TEST_REPORT = ROOT / "artifacts/early_lr_test_v1/report.md"
SUPPLEMENT = ROOT / "artifacts/expanded_eda_supplement_v1"
WORKERS = ROOT / "artifacts/expanded_eda_v2/quality_by_workers.csv"
SPEARMAN = ROOT / "artifacts/expanded_eda_v2/step3_spearman.csv"
W, H = 594.0, 841.0
LEFT, RIGHT, COL = 16.0, 303.0, 275.0
BG = "#FFFFFF"
WHITE = "#FFFFFF"
PALE = "#EAF2EF"
PALE_BLUE = "#EEF3F5"
GRAY = "#8D9C9D"
DARK_AMBER = "#9B641F"
REPORT_URL = URL + "/blob/main/docs/RESEARCH_REPORT_6000_FINAL.md"
FIGURES = {
    "failure": ("01_failure_composition.png", (1540, 700)),
    "conditions": ("05_condition_failure.png", (2376, 1039)),
    "tokens": ("06_token_boxplot.png", (2378, 1096)),
    "completion": ("04_success_completion_step.png", (1540, 560)),
    "step3": ("03_step3_error_repeat.png", (991, 543)),
    "survivors": ("02_error_trends_survivors.png", (1540, 560)),
}

COPY = {
    "zh": {
        "kicker": "探索性数据分析（EDA） · 大语言模型智能体可靠性",
        "subtitle": "基于运行时工具事件的智能体失败预测与提前停止研究",
        "subnote": "完整项目海报｜真实模型轨迹 · 训练集 EDA · 冻结主实验 · 后续探索",
        "stats": [("6,000", "采集运行"), ("5,894", "有效任务"),
                  ("47,184", "可决策前缀"), ("3,542", "训练集 EDA")],
        "s1": "研究背景与目标",
        "intro": (
            "工具报错后仍可能恢复，正常返回也可能写错对象或执行越界操作。项目仅利用当前及过去\n"
            "可观察的工具事件，逐步预测最终失败，并把预测质量、告警时机、费用节省与成功保护\n"
            "放在同一框架。核心问题：哪些失败模式与早期信号相关？提前停止能否减少无效开销，\n"
            "同时保留暂时受阻但仍可成功的任务？预测、违规前预防与无损停止是三个不同目标。"
        ),
        "s2": "研究方法 / 系统、数据与验收",
        "pipeline": ["完整工具\n事件轨迹", "可观察\n运行特征", "逐步失败\n概率预测", "离线回放\n继续或停止"],
        "environment": (
            "真实 kimi-k3-1-pro-261024；本地模拟服务台，含工单与服务事件读写工作流。\n"
            "20 模板 × 20 参数实例 × 5 条件 × 3 重复 = 6,000，归属 10 个父任务家族。\n"
            "条件：正常、超时、限流、Schema 变化、部分响应；每任务≤12 次工具调用、≤24 次模型请求。\n"
            "工具、业务状态和故障由代码构造；采集时不启用风险停止，保留完整轨迹供离线回放。"
        ),
        "partition_header": ["分区", "家族/模板", "有效", "成功", "失败", "无效", "前缀"],
        "partition_rows": [
            ["训练", "6 / 12", "3,542", "2,570", "972", "58", "27,367"],
            ["验证", "2 / 4", "1,176", "997", "179", "24", "10,529"],
            ["测试", "2 / 4", "1,176", "855", "321", "24", "9,288"],
            ["合计", "10 / 20", "5,894", "4,422", "1,472", "106", "47,184"],
        ],
        "protocol": (
            "有效任务成功率 75.03%。严格成功＝必要写入齐全＋最终状态匹配＋无非预期／重复历史写入。\n"
            "每会话保存 session.json、events.jsonl（原始 SSE）、trajectory.json、result.json；审计并封印哈希。\n"
            "已知总费用 ¥1,006.602469，103 次请求费用未知且不补零；无效运行保留支出但排除业务建模。\n"
            "同一家族的模板、实例、条件、重复及全部前缀不跨区；标签、故障元数据、隐藏验收和未来信息均不作特征。"
        ),
        "s3": "研究结果 / 探索性数据分析（EDA）",
        "quality": (
            "范围：仅训练集 3,542 任务／27,367 前缀；质量另含 58 条无效尝试，验证与测试不参与 EDA。\n"
            "22 项特征缺失、负值、非有限值均为 0；ID／步骤键无重复，28,202 条工具事件无完全重复行。\n"
            "4,702 次真实同参重试不删除；62 条耗时与 31 条 Token 外点经 6 任务／24 文件回查后保留。"
        ),
        "a": "A｜失败证据与模板异质性（图 1）",
        "failure_text": (
            "972 次失败：违规且缺写入 411、仅违规 253、仅缺写入 277、仅状态不匹配 31（互斥）。\n"
            "757/972＝77.88% 耗尽 12 步；664 次含违规写入，二者重叠；31 次状态失败均为顺序错误。\n"
            "模板失败率 0.67%–90.41%，两个服务事件家族贡献 67.70% 失败；随机拆分前缀会泄露结构。"
        ),
        "b": "B｜运行条件 × 工作流（图 2）",
        "condition_text": (
            "正常条件仍失败 170 次，其中 159 次含违规；故障不是失败的必要条件，Schema 变化也不必然更差。\n"
            "部分响应失败率：服务事件 45.9%、工单 8.7%；限流／超时预算失败占 91.74%／94.47%。\n"
            "条件是分配设置而非实际触发；组间为描述性关联，不作统一“故障危险等级”或因果排名。"
        ),
        "c": "C｜完整资源差异明显，早期差异弱（图 3）",
        "resource_text": (
            "Token 中位数（成功/失败）：服务事件 14,407/24,945.5，工单 12,410/24,061.5；\n"
            "耗时中位数 67.37/92.61 秒、56.90/86.52 秒；工具步数中位数 7/12、6/12。\n"
            "但第 3 步 Token 均值四组仅 4,226–4,289，尚无违规写入；事后差异不等于早期可分。\n"
            "Token、步骤与墙钟时间的条件排序并不一致；外点保留，长任务仍可能成功。"
        ),
        "s3c": "研究结果 / EDA（续）",
        "d": "D｜错误恢复与完成时点（图 4）",
        "recovery_text": (
            "2,570 次成功中 1,316 次（51.21%）出现显式工具错误，548 次（21.32%）错误≥3。\n"
            "读取错误恢复 4,031/5,926＝68.02%，中位 1 步；不完整政策响应补全 590/804＝73.38%，中位 2 步。\n"
            "恢复指后续同工具同参数正常返回／字段补齐，不等于任务成功；未恢复也受剩余步数影响。\n"
            "77 次成功直到第 12 步才首次达标；80 次最终失败曾达标后被额外写入破坏。"
        ),
        "e": "E｜早期阈值与相关结构（图 5）",
        "early_text": (
            "第 3 步错误×重复没有一致的单调关系：服务事件“0 错误/0 重复”失败 42.8%，\n"
            "“0 错误/有重复”15.8%；工单某 50% 格子仅 n=2。不能反推重复有保护作用。\n"
            "Spearman：重试/重复 0.981，Token/费用 0.884，Token/推理 0.894；分组输出/推理 0.999/0.998。\n"
            "常量列相关性未定义而非 0；相关特征的系数不是独立或因果效应，需联合建模与消融。"
        ),
        "f": "F｜逐步趋势、存活样本与批次混杂（图 6）",
        "survivor_text": (
            "服务事件成功可决策样本由早期 1,003 降至第 11 步 84，失败仍有 705；后期已是不同人群。\n"
            "报告每步 n 与共同队列，避免把存活偏差当成新增信息或性能单调提升。\n"
            "并发 50/10 无效率 8.00%/1.03%，但日期、恢复阶段和供应商状态混杂；模拟超时不等于真实网络等待。"
        ),
        "s4": "预测模型与提前停止 / 冻结主实验",
        "model_text": (
            "22 项特征：进度/写入、错误/重试、Token/费用、时间、最近窗口、工具背景；LR C=1，无类别权重。\n"
            "每任务最多等距抽 6 前缀，共 19,094 拟合行；三折父家族 CV，预处理仅在训练折拟合。\n"
            "KNN k=9 距离加权；固定 OR、步数、永不停止为对照。验证集 5% 误停预算内选最低费用策略。\n"
            "主阈值 0.7380596938；工具返回后才评分，同一步已发生写入不可撤销。"
        ),
        "model_header": ["模型", "AP", "任务等权 AP", "ROC AUC", "Brier", "第3步 AUC"],
        "model_rows": [
            ["LR", "0.8030", "0.7538", "0.8413", "0.1395", "0.7422"],
            ["KNN", "0.7295", "0.6774", "0.7749", "0.1702", "0.6099"],
        ],
        "kpis": [("83.18%", "失败召回 267/321"), ("9.74%", "离线费用节省"), ("−14", "成功净变化 855→841")],
        "main_text": (
            "独立测试 1,176 任务／9,288 前缀：误停 17/855＝1.99%；费用 ¥206.20→¥186.11。\n"
            "严格回放 855−17＋3＝841。告警平均提前 3.82 步；32 早于违规、55 同步、154 之后、26 无违规。\n"
            "真正首次违规前覆盖 32/268＝11.94%，不是 83.18% 的失败召回。5% 预算下 KNN 与规则均选择不停。\n"
            "阶段-only 节省 6.96%；去累计写入节省 5.37% 且净＋2（次要消融），不得据此事后换主模型。\n"
            "全部 17 次误停及几乎全部节省集中于服务事件；两个测试家族限制外推，预测能力不等于干预价值。"
        ),
        "extension": "后续探索 / 48 项特征与既有测试集复评",
        "extension_text": (
            "新增同参调用历史、工具序列、跨步恢复、公开响应完整性；任务等权拟合。22 项对照 C=10，48 项 C=0.1。\n"
            "共享辅助 LR 预测当前达标，两个 Ridge 预测剩余费用/Token，均不进入主 LR；规则为≥90%＋连续 2 步。\n"
            "误停 4→3；费用节省 5.39%→5.83%；失败召回均 237/321；第 6 步前识别 14→60；回放 851→852＜855。\n"
            "验证第 6 步 AUC 0.80856→0.86201，但第 3 步 Brier 0.12525→0.15199；测试全前缀 AUC 0.83404→0.82522。\n"
            "案例：0 显式错误但政策多次不完整，第 8 步停止，原轨迹第 12 步成功；仍难区分暂缺信息与最终失败。\n"
            "该复评复用已经看过的测试集，仅是探索性证据，不是新的独立盲测，也不表示所有门槛都改善。"
        ),
        "s5": "讨论、结论与局限",
        "conclusion": (
            "结论：运行信号可辅助风险监测并减少部分后续浪费，但尚未证明无损自动停止、首次违规预防或在线完成率提升。\n"
            "局限：单一 LLM、两类模拟工作流、6 训练/2 测试家族；验收不含读取充分性和最终回复质量；全部为离线回放。\n"
            "下一步：新独立家族、预注册成功保护阈值、扩展验收；比较告警／延迟决策／完成检测，保留晚恢复难例。"
        ),
        "footer_left": "来源：RESEARCH_REPORT_6000_FINAL.md（2026-11-06）｜参考：ReliabilityBench 2601.06112；AgentRx 2602.02475",
        "footer_right": "github.com/Flysheep1223/agentsentinel-lite",
        "footer_sub": "EDA 仅使用训练分区 · 停止收益均为离线估算 · Seedream 5.5 仅生成页首概念插图，不生成实验图或数字",
    },
    "en": {
        "kicker": "EXPLORATORY DATA ANALYSIS · LLM AGENT RELIABILITY",
        "subtitle": "Failure prediction and early stopping from runtime tool events",
        "subnote": "COMPLETE STUDY · REAL MODEL TRACES · TRAINING-ONLY EDA · FROZEN TEST · EXTENSION",
        "stats": [("6,000", "COLLECTED RUNS"), ("5,894", "VALID TASKS"),
                  ("47,184", "ELIGIBLE PREFIXES"), ("3,542", "TRAINING EDA")],
        "s1": "Introduction & objectives",
        "intro": (
            "Tool errors can recover; normal returns can still violate the task. Using only current and past observable events,\n"
            "the study updates final-failure risk after each tool return and evaluates prediction, alert timing, cost and success\n"
            "protection together. Can early stopping reduce wasted work without sacrificing recoverable runs? Prediction,\n"
            "pre-violation warning and loss-free stopping are distinct objectives, not interchangeable outcomes."
        ),
        "s2": "Methods / system, data & acceptance",
        "pipeline": ["Complete tool\nevent traces", "Observable\nruntime features", "Stepwise failure\nprobability", "Offline replay\ncontinue or stop"],
        "environment": (
            "Real kimi-k3-1-pro-261024 in a LOCAL SIMULATED service desk: ticket and incident read/write workflows.\n"
            "20 templates × 20 parameter instances × 5 conditions × 3 repeats = 6,000 runs in 10 parent families.\n"
            "Conditions: normal, timeout, rate limit, schema change, partial response; ≤12 tool and ≤24 LLM calls/run.\n"
            "Tools, states and faults are programmatic; no risk stopping during collection. Full traces support replay."
        ),
        "partition_header": ["Split", "Fam./temp.", "Valid", "Success", "Failure", "Invalid", "Prefixes"],
        "partition_rows": [
            ["Train", "6 / 12", "3,542", "2,570", "972", "58", "27,367"],
            ["Validation", "2 / 4", "1,176", "997", "179", "24", "10,529"],
            ["Test", "2 / 4", "1,176", "855", "321", "24", "9,288"],
            ["Total", "10 / 20", "5,894", "4,422", "1,472", "106", "47,184"],
        ],
        "protocol": (
            "Valid-task success: 75.03%. Strict success = required writes + final state + no unexpected/duplicate writes.\n"
            "Each session retains session.json, events.jsonl (raw SSE), trajectory.json and result.json; hashes are audited.\n"
            "Known collection cost: ¥1,006.602469; 103 requests have unknown cost, never imputed as zero.\n"
            "Families never cross splits. Labels, fault metadata, hidden verifier answers and future events are not features."
        ),
        "s3": "Results / Exploratory Data Analysis (EDA)",
        "quality": (
            "TRAIN ONLY: 3,542 valid tasks and 27,367 prefixes; quality adds 58 invalid attempts. No validation/test EDA.\n"
            "All 22 features: zero missing, negative or non-finite values; unique ID/step keys; no duplicate rows in 28,202 events.\n"
            "4,702 genuine same-argument retries are not log duplicates. 62 time and 31 token outliers audited (6 tasks/24 files), retained."
        ),
        "a": "A | Failure evidence and task heterogeneity (Fig. 1)",
        "failure_text": (
            "The 972 failures partition into violation+missing 411, violation-only 253, missing-only 277 and state-only 31.\n"
            "757/972 (77.88%) exhausted 12 steps; 664 contained violations (overlapping evidence). All 31 state failures were order errors.\n"
            "Template failure spans 0.67%–90.41%; two incident families contribute 67.70%. Random prefix splits would leak structure."
        ),
        "b": "B | Assigned condition × workflow (Fig. 2)",
        "condition_text": (
            "Normal conditions still produced 170 failures, 159 with violations; faults are not necessary for failure.\n"
            "Partial-response failure: incident 45.9% vs ticket 8.7%. Budget failures form 91.74%/94.47% of rate-limit/timeout failures.\n"
            "Conditions are assigned, not guaranteed triggers. These are descriptive strata, not a universal causal hazard ranking."
        ),
        "c": "C | Large full-run resource gaps, weak early separation (Fig. 3)",
        "resource_text": (
            "Median tokens, success/failure: incident 14,407/24,945.5; ticket 12,410/24,061.5.\n"
            "Median seconds: 67.37/92.61 and 56.90/86.52; median tool steps: 7/12 and 6/12.\n"
            "Yet step-3 token means span only 4,226–4,289 across four groups, before any applied violation.\n"
            "Full-trajectory gaps do not prove early separability. Condition rankings differ across tokens, steps and time."
        ),
        "s3c": "Results / EDA (continued)",
        "d": "D | Error recovery and completion timing (Fig. 4)",
        "recovery_text": (
            "Among 2,570 successes, 1,316 (51.21%) had explicit tool errors and 548 (21.32%) had ≥3 errors.\n"
            "Read errors recovered in 4,031/5,926 events (68.02%; median 1 step); incomplete policies completed in 590/804 (73.38%; 2 steps).\n"
            "Recovery means a later same-tool/same-argument normal return or completed fields—not task success; follow-up is budget-limited.\n"
            "77 successes first completed at step 12; 80 eventual failures once completed, then deteriorated after extra writes."
        ),
        "e": "E | Early thresholds and correlation structure (Fig. 5)",
        "early_text": (
            "Step-3 error × repetition has no consistent monotonic relation: incident 0-error/0-repeat failure is 42.8%,\n"
            "versus 15.8% with repetition; a 50% ticket cell has n=2. This does not show repetition is protective.\n"
            "Spearman: retry/repetition 0.981; tokens/cost 0.884; tokens/reasoning 0.894; output/reasoning 0.999/0.998 by workflow.\n"
            "Undefined constant-column correlations are not zero. Correlated coefficients are not independent causal effects."
        ),
        "f": "F | Cross-step trends, survivors and batch confounding (Fig. 6)",
        "survivor_text": (
            "Eligible incident successes fall from 1,003 early to 84 at step 11, while 705 failures remain: populations change.\n"
            "Report surviving n and common cohorts; do not confuse survivor composition with new information or monotonic gains.\n"
            "Invalid rates at 50/10 workers are 8.00%/1.03%, confounded by date, restart phase and provider state; simulated timeout ≠ network wait."
        ),
        "s4": "Prediction & intervention / frozen main experiment",
        "model_text": (
            "22 observable features: progress/writes, errors/retries, tokens/cost, time, recent window and tool context; LR C=1, unweighted.\n"
            "At most 6 evenly spaced prefixes/task give 19,094 fit rows; three-fold parent-family CV; fold-only preprocessing.\n"
            "KNN k=9 distance-weighted; fixed-OR, step and never-stop baselines. Validation selects lowest cost within a 5% false-stop budget.\n"
            "Main probability threshold: 0.7380596938. Scoring follows tool return; it cannot undo the just-completed write."
        ),
        "model_header": ["Model", "AP", "Task-wt AP", "ROC AUC", "Brier", "Step-3 AUC"],
        "model_rows": [
            ["LR", "0.8030", "0.7538", "0.8413", "0.1395", "0.7422"],
            ["KNN", "0.7295", "0.6774", "0.7749", "0.1702", "0.6099"],
        ],
        "kpis": [("83.18%", "Failure recall 267/321"), ("9.74%", "Offline cost saved"), ("−14", "Net successes 855→841")],
        "main_text": (
            "Held-out test: 1,176 tasks/9,288 prefixes; 17/855 false stops (1.99%); cost ¥206.20→¥186.11.\n"
            "Strict replay: 855−17+3=841. Alerts average 3.82 steps early: 32 before, 55 same-step, 154 after violation, 26 without.\n"
            "True pre-violation coverage is 32/268=11.94%, not the 83.18% failure recall. At 5%, KNN/rules select no-stop.\n"
            "Stage-only saves 6.96%; removing write count saves 5.37% with net +2 successes (secondary ablation, not a new main model).\n"
            "All 17 false stops and nearly all savings concentrate in incidents. Two test families limit generalization; AUC ≠ intervention value."
        ),
        "extension": "Exploratory extension / 48 features, seen test set",
        "extension_text": (
            "Adds same-argument history, call sequences, recovery and public response completeness; task-weighted fitting. Comparator C=10; 48-feature C=0.1.\n"
            "Shared auxiliary LR predicts current completion; two Ridge heads predict remaining cost/tokens, never main-LR inputs. Rule: ≥90% for 2 steps.\n"
            "False stops 4→3; savings 5.39%→5.83%; recall stays 237/321; detections by step 6 rise 14→60; replay 851→852 (<855).\n"
            "Validation step-6 AUC 0.80856→0.86201, but step-3 Brier 0.12525→0.15199; test full-prefix task-weighted AUC 0.83404→0.82522.\n"
            "Case: zero explicit errors but incomplete policies; stopped at step 8, original succeeds at step 12. Temporary information lack remains ambiguous.\n"
            "This reuses an already seen test set: exploratory reevaluation, not a new independent blind test or universal improvement."
        ),
        "s5": "Discussion, conclusion & limitations",
        "conclusion": (
            "Conclusion: runtime signals support risk monitoring and reduce some later waste.\n"
            "Loss-free stopping, violation prevention and online gains remain unproven. Limits: one LLM, two simulated workflows;\n"
            "six train/two test families; acceptance omits reading adequacy/final-reply quality; all outcomes are offline replay.\n"
            "Next: independent families, preregistered protection, richer acceptance, warnings/deferred decisions/completion detection."
        ),
        "footer_left": "Source: RESEARCH_REPORT_6000_FINAL.md (6 Nov 2026) | ReliabilityBench arXiv:2601.06112; AgentRx arXiv:2602.02475",
        "footer_right": "github.com/Flysheep1223/agentsentinel-lite",
        "footer_sub": "EDA uses training only · All stopping gains are offline estimates · Seedream 5.5 created only the conceptual header art, never study figures or numbers",
    },
}


class CompletePoster:
    """Canvas in A1 design millimetres, optionally enlarged to exact A0 paper."""

    def __init__(self, lang: str, data: pd.DataFrame, evidence: dict, paper: str = "a1"):
        self.lang = lang
        self.c = COPY[lang]
        self.data = data
        self.evidence = evidence
        self.paper = paper
        self.physical_w, self.physical_h = (594.0, 841.0) if paper == "a1" else (841.0, 1189.0)
        self.sx, self.sy = self.physical_w / W, self.physical_h / H
        self.font_scale = math.sqrt(self.sx * self.sy)
        self.fig = plt.figure(
            figsize=(self.physical_w / 25.4, self.physical_h / 25.4), dpi=72, facecolor=BG
        )
        self.bg = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.bg.set(xlim=(0, W), ylim=(H, 0))
        self.bg.set_axis_off()
        self.text_bounds = []
        self.body_texts = []

    def font(self, size, bold=False, latin=False):
        path = (BOLD if bold else REGULAR) if self.lang == "en" or latin else CJK
        return FontProperties(
            fname=path, size=size * self.font_scale, weight="bold" if bold else "normal"
        )

    def t(self, x, y, text, size=16.5, color=INK, bold=False, w=None, h=None,
          align="left", latin=False, linespacing=1.14, url=None):
        artist = self.fig.text(
            x / W, 1 - y / H, text, fontproperties=self.font(size, bold, latin),
            color=color, va="top", ha=align, linespacing=linespacing,
            zorder=6, url=url,
        )
        self.text_bounds.append((artist, w, h, x, y))
        self.body_texts.append(text)
        return artist

    def rect(self, x, y, w, h, color, edge=None, lw=0.8, alpha=1, zorder=1):
        patch = Rectangle(
            (x, y), w, h, facecolor=color, edgecolor=edge or "none",
            linewidth=lw * self.font_scale, alpha=alpha, zorder=zorder,
        )
        self.bg.add_patch(patch)
        return patch

    def line(self, x1, y1, x2, y2, color=LINE, lw=0.8):
        self.bg.plot([x1, x2], [y1, y2], color=color, lw=lw * self.font_scale, zorder=2)

    def section(self, x, y, number, title, emphasis=False):
        if emphasis:
            self.rect(x, y - 1.5, COL, 16.5, PALE)
            self.rect(x, y - 1.5, 2.2, 16.5, TEAL)
        self.t(x + 6, y + 1.0, number, 13.5, TEAL, bold=True, latin=True, w=16, h=6)
        self.t(x + 25, y, title, 21.5 if self.lang == "zh" else 20.5,
               bold=True, w=COL - 28, h=12)
        self.line(x, y + 16, x + COL, y + 16, INK, 0.85)

    def subhead(self, x, y, title):
        self.rect(x, y + 2.1, 3.2, 3.2, TEAL)
        self.t(x + 7, y, title, 18.5 if self.lang == "zh" else 17.5,
               bold=True, w=COL - 7, h=9)

    def ax(self, x, y, w, h):
        return self.fig.add_axes(
            [x / W, 1 - (y + h) / H, w / W, h / H], facecolor=BG, zorder=3
        )

    def image(self, key, x, y, w):
        name, expected = FIGURES[key]
        path = FIG_DIR / name
        with Image.open(path) as source:
            image = source.convert("RGB")
            assert image.size == expected
            pixels = np.asarray(image)
        h = w * expected[1] / expected[0]
        axis = self.ax(x, y, w, h)
        axis.imshow(pixels, aspect="auto", interpolation="none", resample=False)
        axis.set_axis_off()
        self.rect(x - 0.5, y - 0.5, w + 1, h + 1, "none", edge=LINE, lw=0.55, zorder=4)
        return h

    def table(self, x, y, widths, header, rows, height, font_size=12.5):
        assert abs(sum(widths) - COL) < 0.01
        row_h = height / (len(rows) + 1)
        self.rect(x, y, COL, row_h, INK)
        for row_index in range(len(rows)):
            if row_index % 2 == 0:
                self.rect(x, y + row_h * (row_index + 1), COL, row_h, PALE_BLUE)
        for r, values in enumerate([header] + rows):
            left = x
            for c, (value, width) in enumerate(zip(values, widths)):
                if c > 0:
                    self.line(left, y, left, y + height, WHITE if r == 0 else LINE, 0.35)
                is_header = r == 0
                align = "left" if c == 0 else "center"
                tx = left + 2 if c == 0 else left + width / 2
                self.t(
                    tx, y + r * row_h + (row_h - font_size * 0.3528) / 2 - 0.2,
                    str(value), font_size - (0.8 if is_header else 0),
                    WHITE if is_header else INK, bold=is_header or (r == len(rows) and self.lang == "zh"),
                    w=width - 3, h=row_h - 0.8, align=align,
                    latin=self.lang == "en" or all(ord(char) < 128 for char in str(value)),
                    linespacing=1.0,
                )
                left += width
            self.line(x, y + r * row_h, x + COL, y + r * row_h,
                      INK if r == 0 else LINE, 0.4)
        self.rect(x, y, COL, height, "none", edge=INK, lw=0.55, zorder=5)

    def header(self):
        with Image.open(ART) as source:
            art = source.convert("RGB").crop((250, 280, 2512, 1410))
        pixels = np.asarray(art).astype(np.float32)
        height, width = pixels.shape[:2]
        horizontal = np.clip(np.arange(width) / (width * 0.18), 0, 1)
        vertical = np.minimum(
            np.clip(np.arange(height) / (height * 0.12), 0, 1),
            np.clip((height - 1 - np.arange(height)) / (height * 0.14), 0, 1),
        )
        alpha = vertical[:, None] * horizontal[None, :]
        background = np.array([255, 255, 255], dtype=np.float32)
        blended = (pixels * alpha[:, :, None] + background * (1 - alpha[:, :, None])).astype(np.uint8)
        image_ax = self.ax(469, 8, 109, 54)
        image_ax.imshow(blended, aspect="auto")
        image_ax.set_axis_off()
        self.rect(16, 8, 24, 1.8, TEAL)
        self.t(16, 13, self.c["kicker"], 13.2, MUTED, w=438, h=6)
        self.t(16, 22, "AgentSentinel-Lite", 54, INK, bold=True, latin=True, w=440, h=23)
        self.t(16, 47, self.c["subtitle"], 22.5 if self.lang == "zh" else 21.5,
               INK, bold=True, w=445, h=11)
        self.t(16, 60, self.c["subnote"], 12.5, MUTED, w=445, h=6)
        for i, (value, label) in enumerate(self.c["stats"]):
            x = 16 + i * 142
            self.rect(x, 70, 136, 25, PALE if i != 3 else PALE_BLUE)
            self.rect(x, 70, 1.8, 25, TEAL if i != 3 else INK)
            self.t(x + 8, 73, value, 24, INK, bold=True, latin=True, w=71, h=11)
            self.t(x + 81, 77, label, 12.2, MUTED, bold=(i == 3), w=49, h=10)

    def introduction_methods(self):
        c = self.c
        self.section(LEFT, 102, "01", c["s1"])
        self.t(LEFT, 121, c["intro"], 16.0 if self.lang == "zh" else 15.5,
               w=COL, h=25, linespacing=1.12)
        self.section(LEFT, 148, "02", c["s2"])
        for i, label in enumerate(c["pipeline"]):
            x = LEFT + i * 70
            self.rect(x, 167, 61, 18, INK if i == 2 else PALE, edge=LINE, lw=0.55)
            self.t(x + 30.5, 169.2, label, 12.8, WHITE if i == 2 else INK,
                   bold=(i == 2), w=57, h=14, align="center", linespacing=1.05)
            if i < 3:
                self.bg.add_patch(FancyArrowPatch(
                    (x + 62, 176), (x + 68.5, 176), arrowstyle="-|>",
                    mutation_scale=10 * self.font_scale, linewidth=0.8 * self.font_scale,
                    color=TEAL, zorder=2,
                ))
        self.t(LEFT, 189, c["environment"], 15.0 if self.lang == "zh" else 14.3,
               w=COL, h=26, linespacing=1.10)
        widths = [37, 43, 38, 39, 35, 32, 51]
        self.table(LEFT, 219, widths, c["partition_header"], c["partition_rows"], 32, 12.2)
        self.t(LEFT, 255, c["protocol"], 13.8 if self.lang == "zh" else 13.0,
               MUTED, w=COL, h=32, linespacing=1.10)

    def eda_left(self):
        c = self.c
        self.section(LEFT, 293, "03", c["s3"], emphasis=True)
        self.t(LEFT, 312, c["quality"], 13.4 if self.lang == "zh" else 12.4,
               MUTED, w=COL, h=16, linespacing=1.06)
        self.subhead(LEFT, 330, c["a"])
        self.t(LEFT, 342, c["failure_text"], 15.2 if self.lang == "zh" else 13.3,
               w=COL, h=21, linespacing=1.08)
        self.image("failure", LEFT + 7.5, 367, 260)
        self.subhead(LEFT, 495, c["b"])
        self.t(LEFT, 508, c["condition_text"], 15.2 if self.lang == "zh" else 13.8,
               w=COL, h=22, linespacing=1.08)
        self.image("conditions", LEFT + 7.5, 535, 260)
        self.subhead(LEFT, 657, c["c"])
        self.t(LEFT, 670, c["resource_text"], 14.8 if self.lang == "zh" else 13.4,
               w=COL, h=27, linespacing=1.07)
        self.image("tokens", LEFT + 17.5, 702, 240)

    def eda_right(self):
        c = self.c
        self.section(RIGHT, 102, "03", c["s3c"], emphasis=True)
        self.subhead(RIGHT, 121, c["d"])
        self.t(RIGHT, 133, c["recovery_text"], 14.0 if self.lang == "zh" else 12.7,
               w=COL, h=25, linespacing=1.06)
        self.image("completion", RIGHT + 19.5, 162, 236)
        self.subhead(RIGHT, 255, c["e"])
        self.t(RIGHT, 268, c["early_text"], 14.0 if self.lang == "zh" else 12.7,
               w=COL, h=25, linespacing=1.06)
        self.image("step3", RIGHT + 23.5, 298, 228)
        self.subhead(RIGHT, 430, c["f"])
        self.t(RIGHT, 443, c["survivor_text"], 14.0 if self.lang == "zh" else 12.3,
               w=COL, h=21, linespacing=1.06)
        self.image("survivors", RIGHT + 24.5, 468, 226)

    def model_and_extension(self):
        c = self.c
        self.section(RIGHT, 561, "04", c["s4"])
        self.t(RIGHT, 580, c["model_text"], 13.8 if self.lang == "zh" else 12.4,
               w=COL, h=27, linespacing=1.06)
        widths = [34, 36, 55, 45, 40, 65]
        self.table(RIGHT, 610, widths, c["model_header"], c["model_rows"], 24, 12.4)
        for i, (value, label) in enumerate(c["kpis"]):
            x = RIGHT + i * 93
            self.rect(x, 639, 86, 20, PALE if i < 2 else "#FBF2E6")
            self.t(x + 43, 640.5, value, 23, TEAL if i < 2 else DARK_AMBER,
                   bold=True, latin=True, align="center", w=82, h=10)
            self.t(x + 43, 652, label, 11.4, MUTED, align="center", w=82, h=5)
        self.t(RIGHT, 663, c["main_text"], 13.2 if self.lang == "zh" else 11.9,
               w=COL, h=34, linespacing=1.05)
        self.subhead(RIGHT, 702, c["extension"])
        self.t(RIGHT, 715, c["extension_text"], 13.0 if self.lang == "zh" else 11.1,
               w=COL, h=48, linespacing=1.05)
        self.section(RIGHT, 770, "05", c["s5"])
        self.rect(RIGHT, 789, COL, 24, PALE, edge=TEAL, lw=1.1)
        self.rect(RIGHT, 789, 2.0, 24, TEAL)
        self.t(RIGHT + 7, 792, c["conclusion"], 12.6 if self.lang == "zh" else 11.0,
               INK, bold=True, w=COL - 13, h=18, linespacing=1.06)

    def footer(self):
        self.line(16, 820, 578, 820, INK, 0.75)
        self.t(16, 824, self.c["footer_left"], 9.8, MUTED, w=390, h=5, url=REPORT_URL)
        self.t(578, 824, self.c["footer_right"], 10.4, INK, align="right",
               latin=True, w=180, h=5, url=URL)
        self.t(16, 832, self.c["footer_sub"], 9.8, MUTED, w=562, h=5)
        self.line(297, 102, 297, 813, LINE, 0.7)

    def build(self):
        self.header()
        self.introduction_methods()
        self.eda_left()
        self.eda_right()
        self.model_and_extension()
        self.footer()
        self.fig.canvas.draw()
        return self.check_layout()

    def check_layout(self):
        renderer = self.fig.canvas.get_renderer()
        errors, bounds = [], []
        factor_x = 25.4 / (self.fig.dpi * self.sx)
        factor_y = 25.4 / (self.fig.dpi * self.sy)
        for artist, w, h, x, y in self.text_bounds:
            box = artist.get_window_extent(renderer)
            actual_w, actual_h = box.width * factor_x, box.height * factor_y
            description = artist.get_text().replace("\n", " / ")
            if w is not None and actual_w > w + 0.65:
                errors.append(f"Width {actual_w:.1f} > {w}: {description}")
            if h is not None and actual_h > h + 0.65:
                errors.append(f"Height {actual_h:.1f} > {h}: {description}")
            if (box.x0 < -1 or box.x1 > self.fig.bbox.width + 1 or
                    box.y0 < -1 or box.y1 > self.fig.bbox.height + 1):
                errors.append(f"Text outside page: {description}")
            bounds.append({
                "text": artist.get_text(), "x_mm": x, "y_mm": y,
                "width_mm": round(actual_w, 2), "height_mm": round(actual_h, 2),
                "font_size_pt_a1_equivalent": round(artist.get_fontsize() / self.font_scale, 2),
            })
        if errors:
            raise ValueError(
                f"{self.lang.upper()} {self.paper.upper()} layout checks failed:\n" + "\n".join(errors)
            )
        return bounds

    def export(self, preview_only=False):
        paper = self.paper.upper()
        stem = OUT / f"AgentSentinel_Complete_EDA_{paper}_{self.lang.upper()}"
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
                        "Title": "AgentSentinel-Lite — Complete study with training-only EDA",
                        "Subject": "6,000 runs; six original EDA figures; frozen 22-feature test and 48-feature exploration",
                        "Keywords": "EDA, LLM agent, failure prediction, early stopping, A1, A0",
                        "Creator": "Matplotlib; conceptual header illustration previously generated by Seedream 5.5",
                    }
                elif suffix == ".svg":
                    kwargs["metadata"] = {
                        "Title": "AgentSentinel-Lite — Complete EDA research poster",
                        "Description": f"Editable {paper} poster; six original report figures; training-only EDA",
                    }
                self.fig.savefig(path, **kwargs)
                files.append(path)
        plt.close(self.fig)
        return files


def verify_supplementary_evidence():
    summary = json.loads((SUPPLEMENT / "summary.json").read_text())
    assert summary["sessions"] == 3542 and summary["prefixes"] == 27367
    assert summary["tool_events"] == 28202 and summary["extreme_sessions_checked"] == 6
    assert summary["raw_files_checked_before_and_after"] == 24
    assert set(summary["duplicates"].values()) == {0}

    quality = pd.read_csv(SUPPLEMENT / "missing_and_numeric_quality.csv")
    valid = quality[quality.table.isin(["valid_sessions", "eligible_prefixes"])]
    assert (valid[["missing", "nonfinite_nonmissing", "negative"]] == 0).all().all()
    attempts = quality[quality.table == "all_train_attempts"].set_index("column")
    assert int(attempts.loc["cost_estimate", "missing"]) == 58
    assert int(attempts.loc["reasoning_tokens", "missing"]) == 4

    outliers = pd.read_csv(SUPPLEMENT / "tukey_outliers.csv")
    time_outliers = int(outliers.loc[outliers.metric == "elapsed_seconds", "above_fence"].sum())
    token_rows = outliers[outliers.metric == "total_tokens"]
    token_outliers = int(token_rows[["below_fence", "above_fence"]].sum().sum())
    assert (time_outliers, token_outliers) == (62, 31)

    recovery = pd.read_csv(SUPPLEMENT / "recovery_comparison.csv")
    read = recovery[(recovery.kind == "explicit_read_error") & (recovery.grouping == "all")].iloc[0]
    incomplete = recovery[(recovery.kind == "incomplete_policy_response") & (recovery.grouping == "all")].iloc[0]
    assert (int(read.recovered_events), int(read.events), read.recovered_delay_median) == (4031, 5926, 1.0)
    assert (int(incomplete.recovered_events), int(incomplete.events), incomplete.recovered_delay_median) == (590, 804, 2.0)

    spearman = pd.read_csv(SPEARMAN, index_col=0)
    assert round(spearman.loc["retry_count", "recent_repeat_ratio"], 3) == 0.981
    assert round(spearman.loc["total_tokens", "cost_estimate"], 3) == 0.884
    assert round(spearman.loc["total_tokens", "reasoning_tokens"], 3) == 0.894
    assert spearman.loc["step"].isna().all()

    workers = pd.read_csv(WORKERS).set_index("workers")
    assert round(workers.loc[50, "invalid_rate"] * 100, 2) == 8.00
    assert round(workers.loc[10, "invalid_rate"] * 100, 2) == 1.03

    report = REPORT.read_text()
    early = EARLY_TEST_REPORT.read_text()
    for token in [
        "0.8413", "0.7422", "83.18%", "9.74%", "0.7380596938", "32/268",
        "19,094", "0.80856", "0.86201", "0.12525", "0.15199", "0.83404", "0.82522",
        "14次增至60次",
    ]:
        assert token in report, f"Metric absent from final report: {token}"
    for token in ["5.39%", "5.83%", "0e8b3d93fc1a4109bfdb2f1ee649c3ab"]:
        assert token in early, f"Metric absent from extension report: {token}"
    for name, expected in FIGURES.values():
        with Image.open(FIG_DIR / name) as image:
            assert image.size == expected
    art_metadata = json.loads((ART.parent / "metadata.json").read_text())
    assert art_metadata["returned_model"] == "doubao-seedream-5-5-261028"
    return {
        "quality": "22 features complete; no negative/non-finite values in valid analysis records",
        "outliers": {"elapsed_seconds": time_outliers, "total_tokens": token_outliers, "retained": True},
        "recovery": {"read_events": "4031/5926", "incomplete_policy_events": "590/804"},
        "step3_spearman": {"retry_repeat": 0.981, "tokens_cost": 0.884, "tokens_reasoning": 0.894},
        "batch_invalid_rates": {"50_workers": 0.08, "10_workers": 0.010303030303030303},
        "six_original_figures_verified": True,
        "seedream_art": "reused; conceptual decoration only; no new generation call",
    }


def check_exports(files):
    results = []
    for path in files:
        result = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        expected_mm = [594.0, 841.0] if "_A1_" in path.name else [841.0, 1189.0]
        if path.suffix == ".png":
            with Image.open(path) as image:
                result["pixels"] = list(image.size)
                result["dpi"] = list(image.info.get("dpi", []))
                assert abs(image.width / image.height - expected_mm[0] / expected_mm[1]) < 0.0002
                if "preview" not in path.name:
                    expected = tuple(int(value / 25.4 * 300) for value in expected_mm)
                    assert image.size == expected, (image.size, expected)
        elif path.suffix == ".pdf":
            content = path.read_bytes()
            assert b"/FontFile2" in content, "PDF must embed TrueType fonts."
            box = re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", content)
            assert box, "Missing PDF page bounds"
            numbers = [float(value) for value in box.group(1).split()]
            result["page_mm"] = [round(numbers[2] * 25.4 / 72, 3), round(numbers[3] * 25.4 / 72, 3)]
            assert all(abs(a - b) < 0.01 for a, b in zip(result["page_mm"], expected_mm))
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
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "savefig.bbox": None,
        "font.family": "Arial", "font.size": 16.5,
    })
    sources = [
        REPORT, TRAIN, ROOT / "artifacts/expanded_eda_v2/fixed_step_summary.csv",
        ART, ART.parent / "metadata.json", EARLY_TEST_REPORT,
        SUPPLEMENT / "summary.json", SUPPLEMENT / "missing_and_numeric_quality.csv",
        SUPPLEMENT / "tukey_outliers.csv", SUPPLEMENT / "recovery_comparison.csv",
        WORKERS, SPEARMAN, *[FIG_DIR / name for name, _ in FIGURES.values()],
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
            poster = CompletePoster(language, data, evidence, paper)
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
        "revision": "Dense complete-project poster; EDA is the largest section",
        "paper_mm": {"A1": [594, 841], "A0": [841, 1189]},
        "print_dpi_a1_png": 300,
        "preview_only": args.preview_only,
        "source_inputs_unchanged": True,
        "source_sha256": before,
        "script_sha256": sha256(Path(__file__)),
        "fonts": {"latin_regular": REGULAR, "latin_bold": BOLD, "chinese": CJK},
        "eda_evidence": evidence,
        "distribution_checks": distribution_checks,
        "supplementary_checks": supplementary,
        "scope": {
            "eda": "Training split only; six original report figures; validation/test excluded",
            "main_experiment": "Pre-frozen 22-feature LR; independent test; strict offline replay",
            "extension": "48 features; reuses seen test set; exploratory, not a new blind test",
            "concept_art": "Seedream 5.5 header only; never a source of data or results",
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

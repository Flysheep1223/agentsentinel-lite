# 文档导航

更新：2026-11-06。当前研究已定稿，唯一分析批次为6,000次采集，
其中5,894条有效任务、47,184条可决策前缀。默认只预测，本阶段没有继续训练或在线实验待办。

## 从哪里开始

| 目的 | 文档 |
| --- | --- |
| 第一次了解项目、向合作者讲解 | [合作者指南与讲稿](COLLABORATOR_GUIDE.md) |
| 查完整方法、结果及研究局限 | [最终研究报告](RESEARCH_REPORT_6000_FINAL.md) |
| 查看安装、复核和预测示例 | [仓库README](../README.md) |
| 确认已经完成哪些工作 | [完成记录](NEXT_STEPS_EXPANDED.md) |
| 了解保留目录、依赖及清理范围 | [仓库结构与清理记录](REPOSITORY_LAYOUT.md) |

`NEXT_STEPS_EXPANDED.md` 沿用原文件名，内容已是完成记录。
当前结果与研究边界以最终报告为准，阶段文档中的“本轮”指各自实验阶段。

## 按研究阶段查证据

| 阶段 | 结果说明 | 操作说明 | 冻结协议 |
| --- | --- | --- | --- |
| 数据审计 | [审计报告](../artifacts/expanded_dataset_v1/audit_report.md) | [审计与加载](EXPANDED_DATA_PIPELINE.md) | [采集协议](../data/expanded_20261031_6000_v3_auto/protocol.json) |
| 训练EDA | [发现与建模依据](EXPANDED_TRAINING_EDA.md) | 同左，含复现命令 | [分析计划](../artifacts/expanded_eda_v2/analysis_plan.json) |
| 预先冻结的主实验 | [LR/KNN/规则结果](EXPANDED_MODEL_RESULTS.md) | [主实验复核](EXPANDED_MODEL_USAGE.md) | [主实验方案](EXPANDED_MODEL_PROTOCOL.md) |
| 逐步预测v1，历史探索 | [四个预测头与完成停止](STEPWISE_LR_RESULTS.md) | [v1复现](STEPWISE_LR_USAGE.md) | [v1方案](STEPWISE_LR_PROTOCOL.md) |
| 最终任务预测v2，22列对照 | [最终成功预测](TASK_OUTCOME_LR_RESULTS.md) | [22列模型使用](TASK_OUTCOME_LR_USAGE.md) | [v2方案](TASK_OUTCOME_LR_PROTOCOL.md) |
| 早期LR，48列探索 | [开发验证结果](EARLY_LR_RESULTS.md) | [48列模型使用](EARLY_LR_USAGE.md) | [早期特征方案](EARLY_LR_PROTOCOL.md) |
| 冻结模型的既有测试集复评 | [测试复评结果](EARLY_LR_TEST_RESULTS.md) | [复评产物复核](EARLY_LR_TEST_USAGE.md) | [预测前方案](EARLY_LR_TEST_PROTOCOL.md) |

主实验包含一次预先冻结的独立测试；其后的模型开发与同一测试集复评属于探索。
开发验证中的成功/失败分母为997/179，测试中的分母为855/321，两者不能混用。

## 模型入口与版本

| 用途 | Python模块 | 产物 | 默认行为 |
| --- | --- | --- | --- |
| 使用48列LR进行事件序列预测 | `agentsentinel.early_lr` | `artifacts/early_lr_v1` | 只预测 |
| 使用22列最终结果对照模型 | `agentsentinel.task_outcome` | `artifacts/stepwise_lr_v2` | 只预测 |
| 复现历史v1完成停止研究 | `agentsentinel.stepwise` | `artifacts/stepwise_lr_v1` | 使用停止政策，含完成分支 |
| 复核主实验、读取已有结果 | `agentsentinel.expanded_comparison` | `artifacts/expanded_model_comparison_v1` | 通过 `verify` 复核 |
| 复核既有测试集比较 | `agentsentinel.early_test` | `artifacts/early_lr_test_v1` | 通过 `verify` 复核 |

48列与22列模型的运行时输入格式不同，使用对应说明中的示例。
停止政策名 `risk@0`、`risk@0.01` 表示选择时的误停预算，不是通用概率门槛：
22列v2原政策在验证集选择，48列探索及测试复评的政策在训练OOF选择，
同名政策的实际阈值可能不同，应查各自产物和使用说明。
模型版本较新不表示所有指标更好，全部停止收益仍是离线估算。

## 历史与冻结材料

[原始研究提案](../AgentSentinel_Lite_Proposal.md)保留最初的问题与方法设计。
其中600条计划、预算和在线对照安排属于历史方案，不能作为当前数据或待执行计划。
[论文索引](../papers/README.md)及6份PDF保留引用时的版本，不因项目结束删除参考文献。

5份 `*_PROTOCOL.md`、产物内7份报告及5份协议快照均被实验封印引用，
正文、文件名和路径保持原样。它们中的未来时态、旧目录说明和工程测试数量是阶段记录。
例如封印内主实验报告仍提到旧批次报告保留，EDA报告仍写着“下一步建模”；
这些历史表述不代表当前仓库状态，当前范围以本文和仓库结构说明为准。

各阶段的 `*_RESULTS.md` 解释具体实验，最终报告汇总研究结论，二者用途不同，保留独立文件。
重复的全量审计段落已合并；旧报告与无关数据已在前次仓库清理中删除。
本次文档整理更新入口、状态和示例命令，不改写实验数值、原始数据或封印。

## 维护约定

- 已有产物优先运行 `verify`；`build`、`train`、`run` 示例只用于另行复现，输出目录必须不存在。
- Python命令使用项目 `.venv`。本地复核不需要API；文档中的复现说明不代表要重新执行实验。
- 327、349、372、395、410等测试计数保留为对应阶段记录；最近代码回归为数据清理后的409项通过、1项按原条件跳过，见仓库清理记录。
- 新结论更新最终报告和对应解释文档；冻结协议及封印内报告保持原样，不能重算哈希来掩盖修改。

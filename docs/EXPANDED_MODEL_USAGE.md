# 6,000条数据主实验：使用与复核

建模入口为 `agentsentinel.expanded_comparison`，显式使用独立的 schema 4 数据集加载器。
本阶段已完成并纳入[最终报告](RESEARCH_REPORT_6000_FINAL.md)；以下以已有产物复核为主。
科学口径见[拟合前方案](EXPANDED_MODEL_PROTOCOL.md)，数据来源见
[审计与加载说明](EXPANDED_DATA_PIPELINE.md)。

## 命令

本次结果目录为 `artifacts/expanded_model_comparison_v1`。已有产物使用校验命令：

```bash
.venv/bin/python -m agentsentinel.expanded_comparison verify \
  --output artifacts/expanded_model_comparison_v1
```

从冻结数据完整复现时，输出目录必须尚不存在：

```bash
.venv/bin/python -m agentsentinel.expanded_comparison run \
  --dataset artifacts/expanded_dataset_v1 \
  --eda artifacts/expanded_eda_v2 \
  --output artifacts/expanded_model_comparison_reproduction
```

该命令不加载 `.env`、不请求模型、不调用采集运行器。它先核对原始文件哈希，
保存分析方案、源码快照与依赖版本，再进行训练与验证。所有23项政策及验证阶段的
在线研究建议冻结后，写入不可覆盖的测试启动记录，才读取测试私有诊断并生成测试预测。
末尾再次核对原始文件、冻结记录与产物，生成最终 `manifest.json`。

复现相同数据不构成新的独立实验。已有测试结果不能用于调参后再次宣称独立测试。
执行中断时保留目录用于排查，不删除启动记录或覆盖已产生的测试结果。
源码或依赖改变会使校验失败；应恢复对应源码快照和依赖，不能改写冻结哈希。

## 文件与解释

| 文件/目录                                           | 用途                                                |
| --------------------------------------------------- | --------------------------------------------------- |
| `analysis_plan.json`、`source_snapshot/`        | 拟合前协议、代码、测试、依赖及哈希                  |
| `training_sample.csv`、`cv/`                    | 每Session最多6条拟合样本、逐折评分与最佳候选OOF预测 |
| `train/`、`validation/`                         | 全训练前缀预测、验证预测与冻结阈值的回放结果        |
| `thresholds/`、`policies.json`                  | 全部验证候选曲线与最终模型/规则阈值                 |
| `models/`                                         | 已拟合模型，只接收对应特征allowlist                 |
| `validation_online_recommendation.json`           | 测试前记录的建议，不是收费执行授权                  |
| `all_policies_frozen.json`、`test_started.json` | 所有模型政策冻结与单次测试屏障                      |
| `test/summary.csv`、`test/groups.csv`           | 每政策及模板/家族/工作流/条件/终止原因的完整分母    |
| `test/fixed_steps.csv`、`test/results.json`     | 固定早期指标、分类/校准、敏感性与父家族bootstrap    |
| `test/*_sessions.csv`                             | 首次告警、当步已消耗资源、停止反事实及违规前判断    |
| `test/diagnostic_*.csv`                           | 私有验证答案的事后解释，严禁输入风险模型            |
| `figures/`、`report.md`、`manifest.json`      | 独立PNG/SVG图件、报告与完整性封印                   |

严格误停是原成功Session出现告警，无论此时是否已经完成写入。
状态反事实单独计算尚未完成的成功损失、已完成后停止，以及原失败在停止时满足验证的情况。
二者回答不同问题，均需保留；状态反事实不能替代真实在线收益。

违规前告警要求告警步严格小于首次违规步。同一步已发生的模型请求、写入和费用不能撤销；
第12步不属于可告警前缀。纯顺序错误另作敏感性分析。

测试只有两个父家族。分组bootstrap保留整个家族，其区间仅表示有限家族组成的变化。
费用缺失统一切换Token选择目标，不补零；空评价子集输出0计数和null比率。

## 主实验完成时的工程验证记录

- 全量回归：327项通过；新增专项及相关离线回归：30项通过。
- `ruff check src tests`：通过。
- 覆盖父家族隔离、全部留出前缀评分、Session等权、特征隔离、费用缺失、
  规则平局、完成/违规时点、空子集、原始证据门禁、冻结哈希与不可覆盖测试。
- 使用合成小样本完整执行训练、23项政策冻结、测试及PNG/SVG报告生成，
  仅用于工程验证，不计入真实实验分母。

上述计数属于主实验阶段；最近代码回归见[仓库清理记录](REPOSITORY_LAYOUT.md)。

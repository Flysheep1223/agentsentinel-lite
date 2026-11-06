# 冻结LR的既有测试集复评入口

方案见[预测前冻结方案](EARLY_LR_TEST_PROTOCOL.md)。本入口仅对
`artifacts/early_lr_v1` 中的 `frozen_v2` 与 `history_payload` 做离线推断。
整个task原始最终验收结果是主标签，当前验收事实仅用于回放评价。
旧开发入口仍然拒绝test，不需要修改或绕过其分区限制。
复评已完成，当前应校验已有产物；研究结论见[最终报告](RESEARCH_REPORT_6000_FINAL.md)。

## 执行与复核

使用项目 `.venv`，不读取 `.env` 或调用付费API。已有结果使用以下命令复核，不产生新预测：

```bash
.venv/bin/python -m agentsentinel.early_test verify \
  --output artifacts/early_lr_test_v1 --check-sources
```

复评执行前曾通过410项回归测试及全量ruff检查。新增15项测试覆盖test分区边界、
冻结模型纯推断、原22列一致性、最终标签隔离、严格验收回放、
缺失费用、连续证据、不可覆盖和原文件/模型/源码封印篡改检测。
真实冻结模型在合成事件流上的预测与原运行时监测器已核对一致。

最近代码回归为仓库数据清理后的409项通过、1项旧模型相关测试按原条件跳过，
详见[清理记录](REPOSITORY_LAYOUT.md)。工程回归与研究结果评价分别解释。

仅在另行复现时使用以下命令；输出目录必须尚不存在。
已有 `artifacts/early_lr_test_v1` 保持原样，不能覆盖或删除后重跑：

```bash
.venv/bin/python -m agentsentinel.early_test run \
  --previous artifacts/early_lr_v1 \
  --output artifacts/early_lr_test_reproduction --workers 6
```

运行会先检查既有实验封印，记录方案、依赖、源码快照和数据集指纹，
复制冻结的模型与政策，写入 `models_policies_frozen.json` 和
`test_started.json`，再读取本轮test数据并生成预测。
结束时复核既有实验链、当前源码以及消费的原始文件；未完成的输出目录不能作为结果使用。
verify不产生新预测。

本轮固定主政策 `risk@0.01`：失败概率≥90%，连续两步确认。
固定次要政策 `risk@0`：失败概率≥99%，连续两步确认。
另外保存预先指定的80%门槛与只预测参考；90%同门槛比较与主政策共用结果。
两模型共享冻结的当前验收和剩余资源预测头。费用预测不可用时不停止，
步骤间隔重置连续证据，第12步不再产生新决定。
此次评价不会拟合模型、校准概率、选择候选或调整门槛。

## 产物

| 路径 | 内容 |
|---|---|
| `report.md` | 主比较与评价口径 |
| `prediction_metrics.csv` | 全前缀、固定步、共同队列和各分组的预测指标 |
| `calibration.csv` | 成功概率的10箱校准数据 |
| `early_brier.csv` | 第3/6步Brier的平均值 |
| `policy_results.csv` | 停止率、误停、失败覆盖、严格成功损失/挽救、时间、费用与Token |
| `policy_groups.csv` | 父家族、模板、工作流和条件分组政策统计 |
| `test/*_predictions.csv` | 每个模型的逐前缀概率与评价上下文 |
| `replay/*.csv` | 每条task的首次停止与严格验收回放结果 |
| `diagnostics/*_false_stop_steps.csv` | 所有误停task的逐步特征、可决策步概率和验收事实 |
| `diagnostics/*_examples.json` | 每个模型/政策按词典序首个误停task |
| `private/` | 原始最终标签、原验证器回放事实、全步骤特征 |
| `consumed_source_hashes.json` | 本轮消费的原始trajectory/result哈希 |
| `manifest.json` | 完成状态与全部产物封印 |

`private/` 与预测CSV中的标签、ID、验收和未来资源事实只用于核对与评价。
预测调用严格投影到冻结模型的22/48列运行时特征，不把整个评价表传给模型。
固定时点只包含存在该前缀的task；共同队列限制为同时存在第3/6/9步的task。
全部前缀按task等权，不让长轨迹主导统计。

## 解释口径

成功任务误停率=被停止的原成功task/全部原成功task，显示为百分比；
即使停止时已完成，也计入误停。严格成功损失另按停止时原验证器判定，
挽救原失败task不能抵扣误停。
平均/中位停止步只以停止task为分母。
费用与Token节省以全部有效test原轨迹为分母；任一所需费用缺失时，
总费用节省保持未知，不按零补齐或临时切换政策。

这些是既有轨迹上的离线回放估计，不含监测器开销。测试集之前已用于第二轮模型评价，
因此本轮称为既有测试集复评，不声称新的盲测。少数父任务家族限制了泛化结论，
不能根据此次test结果再挑选模型或门槛。

# 整个任务成功预测：22列v2使用说明

本文件对应22列对照模型，入口是`agentsentinel.task_outcome`，默认只预测。模型目录为
`artifacts/stepwise_lr_v2`。旧`agentsentinel.stepwise`保留用于复现v1，
其宽松状态完成目标和默认完成停止行为不适用于新版。
48列事件序列模型见[早期LR说明](EARLY_LR_USAGE.md)；两者的输入格式与政策阈值不能混用。
本阶段已完成，以下训练命令仅用于另行复现。

## 每一步在预测什么

同一个已结束Session的所有前缀，主标签均为`task_success=1-原label`。
例如第1步超时、第2步重试、第8步完成并最终通过原验收：这些前缀的主标签全部是1。
模型在运行时看不到结局，只能根据当前22列可观测特征更新最终成功概率。
概率可以先降后升；每步重新预测不等于每步重训参数。

| 输出 | 含义 |
| --- | --- |
| `p_task_success` | 原Agent策略继续执行时，整个task最终通过原验收的预测概率 |
| `p_task_failure` | `1-p_task_success` |
| `auxiliary.p_acceptance_now` | 当前已经通过同一原验收器的预测概率 |
| `auxiliary.remaining_cost_cny` | 从现在到原自然结束的预计剩余人民币费用 |
| `auxiliary.remaining_tokens` | 从现在到原自然结束的预计剩余Token |
| `decision` | 默认恒为`continue`；显式风险政策才可能返回`stop` |

原验收检查必要写入、完整状态和历史额外/重复写入，不独立核验必要读取与最终回复质量。
历史错写修复后仍可能失败，沿用原规则；辅助头不改变成功定义，也不成为主模型输入。
剩余费用已扣除当步开销；不可用的资源头输出`null`，不会把未知值当0。
概率存在校准误差，不是保证，不代表任务已经不可恢复。

## 直接运行默认预测

在仓库根目录使用`.venv`：

```bash
.venv/bin/python -m agentsentinel.task_outcome verify \
  --output artifacts/stepwise_lr_v2 --check-inputs

.venv/bin/python -m agentsentinel.task_outcome predict \
  --output artifacts/stepwise_lr_v2 \
  --features artifacts/stepwise_lr_v2/example_features.json
```

特征文件为**一个Session**的JSON数组，每行恰好22列`FEATURE_COLUMNS`。
只在可继续的工具返回之后输入；本批允许第1～11步，第12步用于最终结果监督，
不再产生停止决策。步数必须递增；不要把多个Session拼成同一个序列。
运行时不传入task标签、验收结果、任务ID、模板、答案或未来费用。

## 在Python中接入

```python
from agentsentinel.features import FeatureBuilder
from agentsentinel.task_outcome import load_monitor

builder = FeatureBuilder(workflow="ticket")
monitor = load_monitor("artifacts/stepwise_lr_v2")

# observable_event来自已有运行器的工具返回日志，包含当步累计用量。
# 每个Session创建独立builder和monitor。
features = builder.add(observable_event)
prediction = monitor.observe(features)
print(prediction["p_task_success"])
```

调用方在创建特征前检查运行器的可继续状态，例如`decision_eligible`。
`observe()`返回概率和建议，本身不会调用Agent、修改任务状态或执行停止动作。
使用序列化模型前只接受本项目本地生成且可信的joblib产物；加载时验证文件封印。

## 可选风险政策

```bash
.venv/bin/python -m agentsentinel.task_outcome predict \
  --output artifacts/stepwise_lr_v2 \
  --features artifacts/stepwise_lr_v2/example_features.json \
  --policy 'risk@0'
```

`risk@0`与`risk@0.01`分别是在已使用的验证集上按0%和1%原成功误停预算选择的政策。
只有最终成功概率低、预计剩余资源满足门槛、连续证据达到要求时才给出停止建议。
没有“当前完成概率高就停止”的分支。步数有间隔会重置连续证据，停止后拒绝继续观察。
无论停止时是否已经通过验收，原本最终成功的Session被停止均计作误停。
实际验收成功损失/挽救另外统计，不能抵消误停预算。
验证预算不是新任务的性能保证；运行器的自动停止集成不包含在本轮修改中。

## 重新离线训练

```bash
# 已有v2禁止覆盖；重新训练必须使用新的版本目录。
.venv/bin/python -m agentsentinel.task_outcome train \
  --dataset artifacts/expanded_dataset_v1 \
  --old-comparison artifacts/expanded_model_comparison_v1 \
  --previous artifacts/stepwise_lr_v1 \
  --output artifacts/task_outcome_reproduction
```

方案先于拟合冻结，保留原父家族分区与三折CV。
两个LR按Brier选参，两个Ridge按MAE选参；每Session最多6个拟合前缀，按Session等权。
该训练入口只对train/validation制作新预测；原测试已曝光，不用于开发选参。
已有的测试复评使用独立入口，见[复评说明](EARLY_LR_TEST_USAGE.md)。
训练、验证和预测均不加载API配置、不发送收费请求。
输出目录存在时直接拒绝；旧模块、旧产物、原始数据和标签保持原样。

## 主要产物

| 文件 | 用途 |
| --- | --- |
| `report.md` | 主预测、辅助指标和可选政策分开的报告 |
| `prediction_metrics.csv` | 总体、固定步、共同人群及家族/模板/工作流/条件指标 |
| `calibration.csv` | 各模型、各分组的概率校准分箱 |
| `heads.json`、`models/`、`cv/` | 模型、固定候选与分组CV记录 |
| `validation/predictions.csv` | 主预测、辅助预测及离线真实目标 |
| `validation/comparator_predictions.csv` | 与旧冻结验证预测对齐的主目标比较 |
| `prediction_examples.csv` | 按预定条件和Session ID选择的完整逐步示例 |
| `policies.json`、`policy_comparison.csv`、`replay/` | 可选风险政策和逐Session回放 |
| `private/` | 原始轨迹重放生成的离线监督目标，不输入运行时模型 |
| `analysis_plan.json`、`source_snapshot/`、`manifest.json` | 拟合前方案、源码及输出封印 |

详细方法见[拟合前方案](TASK_OUTCOME_LR_PROTOCOL.md)，实际结果见
[新版结果](TASK_OUTCOME_LR_RESULTS.md)。当前验证集已反复使用，后续泛化评价需要新的独立任务。

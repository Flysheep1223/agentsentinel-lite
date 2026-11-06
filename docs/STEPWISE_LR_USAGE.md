# 逐步LR预测：使用与复核

> 以下入口用于复现6,000条数据上的历史v1，其默认会使用完成停止分支。
> 最终任务结果预测可使用[22列v2](TASK_OUTCOME_LR_USAGE.md)或[48列LR](EARLY_LR_USAGE.md)，均默认只预测。
> 本阶段已完成，v1用于保留研究过程及上游依赖，不是当前推荐的运行入口。

本轮使用原22列运行时特征，增加当前完成、继续成功、剩余人民币成本、
剩余Token四个目标。前两个使用Logistic Regression，后两个使用Ridge线性回归。
每步重新判断，模型参数离线拟合；尚未做在线增量学习，也未自动修改Agent行动。

方案见[拟合前协议](STEPWISE_LR_PROTOCOL.md)，结果见[开发结果](STEPWISE_LR_RESULTS.md)。
独立目录为`artifacts/stepwise_lr_v1`；原始数据与第二轮比较产物不覆盖。

## 复核与复现

所有命令使用项目`.venv`，不读取`.env`，不调用模型API。

```bash
# 校验已有产物，以及实际消费的原文件、源码和上游比较封印：
.venv/bin/python -m agentsentinel.stepwise verify \
  --output artifacts/stepwise_lr_v1 --check-inputs

# 复现必须使用尚不存在的独立目录：
.venv/bin/python -m agentsentinel.stepwise train \
  --dataset artifacts/expanded_dataset_v1 \
  --old-comparison artifacts/expanded_model_comparison_v1 \
  --output artifacts/stepwise_lr_v1_reproduction --workers 6
```

训练入口只提取train/validation原始轨迹。封印加载器可读全表校验一致性，
但新目标、拟合、预测和策略选择不访问test分区。入口拒绝覆盖已有目录；
中断产物保留，检查原因后新建版本，不能伪造完成封印。

预测头使用固定父家族三折CV、每Session最多六个等距拟合前缀、Session等权评分。
参数与政策候选在拟合前协议中列出。全部验证收益属于开发性回放估算，
不能将反复使用的验证集称为新的独立测试。

## 本地逐步预测

产物包含一条真实验证Session的22列特征示例，可直接运行：

```bash
.venv/bin/python -m agentsentinel.stepwise predict \
  --output artifacts/stepwise_lr_v1 \
  --policy combined@0 \
  --features artifacts/stepwise_lr_v1/example_features.json
```

输入是**同一个Session**的JSON数组，每项恰好包含`FEATURE_COLUMNS`的22列。
输出为逐行JSON：四个预测值、`continue`或`stop`、原因与各分支连续命中次数。
工具调用后仍可继续的位置才送入控制器；本批第12步不再是决策点。
同一Session步数必须递增，间隔重置连续证据，停止后拒绝新决策。

Python集成方式：

```python
from pathlib import Path
from agentsentinel.features import FeatureBuilder
from agentsentinel.stepwise import load_controller
from agentsentinel.stepwise_policy import Controller

# 从自己生成的可信本地产物加载一次；joblib不用于加载不可信文件。
bundle = load_controller(Path("artifacts/stepwise_lr_v1"), "combined@0")

# 每个Session分别初始化，避免不同任务共享步数和连续证据。
controller = Controller(bundle.heads, bundle.policy)
builder = FeatureBuilder(workflow="ticket")

# 在现有运行器的工具返回处：
# features = builder.add(tool_end_event)  # 包含实际累计totals、耗时及工具返回字段
# if tool_end_event["decision_eligible"]:
#     result = controller.observe(features)
#     if result["decision"] == "stop":
#         ...  # 由调用方执行停止；本轮未接入收费在线运行器
```

`Controller.observe()`拒绝额外字段，任务ID、私有答案、真实当前完成、
未来结果与剩余真实成本均不能作为输入。四个头固定读取22列allowlist。
创建新的控制器可复用只读模型对象；每个Session维护自己的控制器和FeatureBuilder。

政策包括：

- `combined@0`：主开发政策，完成分支和低继续收益分支，验证完成损失预算0%。
- `completion_only@0`：仅完成分支，预算0%。
- `combined@0.01`、`completion_only@0.01`：次要开发政策，预算1%。

完成分支优先，两个分支独立累计连续证据。
若人民币目标不完整，整次选策统一改用Token资源门槛，不把未知费用当零。
验证中没有符合预算且更省资源的候选时，会保留永不停止。
0%是当前验证样本约束，不能承诺新任务零误伤。

## 产物说明

| 文件 | 用途 |
| --- | --- |
| analysis_plan.json / source_snapshot/ | 拟合前方案、依赖、代码和特征边界 |
| consumed_source_hashes.json | 仅train/validation实际消费的原文件哈希 |
| session_targets.csv / all_step_targets.csv / prefix_targets.csv | 私有监督目标及合规诊断，不直接喂入模型 |
| models/ / heads.json | 四个头及其参数、哈希、CV/验证性能和校准 |
| cv/ | 全部候选折评分、最佳候选完整留出前缀预测 |
| train/predictions.csv / validation/predictions.csv | 对齐到Session/step的目标与预测 |
| policy_candidates.csv / policies.json | 验证候选曲线与选中政策 |
| replay/ / comparison.csv / policy_groups.csv | 逐Session回放、旧LR对照及分组结果 |
| target_by_step.csv | 每步存活分母与目标分布 |
| example_features.json / prediction_examples.csv | 可运行输入示例和可读预测诊断 |
| report.md / figures/ / manifest.json | 开发报告、图与完整性封印 |

“当前完成”要求必要写入次数全部满足且完整状态相同；历史写错后修复可重新完成。
原严格合规标签另存。必要读取和最终回复质量未独立验证，不能宣称覆盖全部任务语义。
“继续成功”只描述原执行策略的结局，不是某次告警之后任务已经无法挽救。
Ridge预测值钳制为非负，未知费用保留缺失；预测值不是计费预算保证。

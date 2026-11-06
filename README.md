# AgentSentinel-Lite

**基于运行日志的 LLM Agent 失败预测与提前停止研究。**

项目研究工具调用中的结构化运行信号，能否在任务结束前预测最终失败，
并在限制成功任务误停的前提下减少后续执行成本。Logistic Regression（LR）为主模型，
KNN、固定规则、步数规则和永不停止为对照。

Agent 使用真实 LLM，在本地模拟的工单和服务事件环境中执行任务。
模型请求与执行轨迹来自真实运行；工具、业务状态和故障由代码构造，不连接真实业务系统。
结论限于当前模型、任务家族和运行设置。

## 阅读入口

- [合作者入门说明与讲解稿](docs/COLLABORATOR_GUIDE.md)：从具体任务解释 Agent、前缀预测和实验结果。
- [6,000次采集最终研究报告](docs/RESEARCH_REPORT_6000_FINAL.md)：当前研究的完整结论与证据。
- [已完成工作记录](docs/NEXT_STEPS_EXPANDED.md)：数据审计、主实验和后续探索的完成情况。
- [仓库结构与清理记录](docs/REPOSITORY_LAYOUT.md)：保留目录、依赖关系和已移除内容。
- [全部文档导航](docs/README.md)：按研究阶段查找结果、操作说明和冻结协议，并区分模型版本。

## 当前数据与结论

最终报告仅采用 `data/expanded_20261031_6000_v3_auto`：
20个结构模板 × 每模板20个实例 × 5种运行条件 × 3次重复，共6,000次采集。

| 分区 | 有效任务 |  成功 |  失败 | 排除的无效运行 |
| ---- | -------: | ----: | ----: | -------------: |
| 训练 |    3,542 | 2,570 |   972 |             58 |
| 验证 |    1,176 |   997 |   179 |             24 |
| 测试 |    1,176 |   855 |   321 |             24 |
| 合计 |    5,894 | 4,422 | 1,472 |            106 |

有效任务生成47,184条可决策前缀。全部采集的已知费用经审计为¥1,006.602469，
另有103次请求费用未知；未知费用不记作零。

预先冻结的主LR在独立测试中识别267/321次失败（83.18%），
误停17/855次成功任务（1.99%），离线估算节省费用9.74%；
严格状态回放的成功任务数净减少14次。

后续48列LR在同一既有测试集、90%失败概率连续两步门槛下，
相对探索阶段的22列对照，误停由4次降至3次，费用节省由5.39%升至5.83%。
这属于后续探索，不能当作新的独立盲测；主门槛下仍净损失3次成功任务。

运行信号能够识别部分失败并减少后续开销，但可靠的早期预警、违规前预防和无损自动停止尚未实现。
研究已定稿，本阶段不再扩展实验。当前默认只预测，每步更新整个任务最终通过原验收的概率。
所有停止收益均为离线回放估算，本批次没有在线干预实测证据。

## 保留的目录

| 路径                                               | 用途                               |
| -------------------------------------------------- | ---------------------------------- |
| `data/expanded_20261031_6000_v3_auto/`           | 6,000条原始轨迹、设计、协议与账本  |
| `artifacts/expanded_dataset_v1/`                 | 审计、有效任务与前缀表、原文件哈希 |
| `artifacts/expanded_eda_v2/`                     | 当前训练集EDA、图表与轨迹证据      |
| `artifacts/expanded_model_comparison_v1/`        | 冻结主实验、LR/KNN/规则比较与消融  |
| `artifacts/stepwise_lr_v1/`、`stepwise_lr_v2/` | 6,000条数据上的逐步预测探索及依赖  |
| `artifacts/early_lr_v1/`、`early_lr_test_v1/`  | 48列LR探索与既有测试集复评         |
| `src/agentsentinel/`、`tests/`                 | 实现、兼容代码与回归测试           |
| `docs/`                                          | 当前报告、使用说明与冻结方案       |
| `papers/`、`AgentSentinel_Lite_Proposal.md`    | 研究参考材料与原始提案             |

模型产物相互引用并校验源码、协议和上游文件。名称中的 `v1` 不代表旧批次；
上述目录共同构成当前6,000条研究的复核链，不应单独删除。
旧采集与报告的少量兼容源码仍被封印校验引用，保留它们以保证现有结果可复核；
当前使用以下离线入口。

## 环境与离线检查

依赖由 `uv.lock` 固定，所有 Python 命令使用项目 `.venv`。
以下命令只读取本地数据，不发送模型请求。

```bash
uv sync --python 3.12 --extra dev --frozen
.venv/bin/pytest -q
.venv/bin/ruff check src tests

# 数据集封印及全部原始文件哈希
.venv/bin/python -m agentsentinel.expanded_dataset verify \
  --dataset artifacts/expanded_dataset_v1 --verify-raw

# 主实验及上游依赖
.venv/bin/python -m agentsentinel.expanded_comparison verify \
  --output artifacts/expanded_model_comparison_v1

# 后续探索、既有测试复评及消费的原始文件
.venv/bin/python -m agentsentinel.early_test verify \
  --output artifacts/early_lr_test_v1 --check-sources
```

离线审计和预测无需 API 密钥。真实模型配置使用本地 `.env`，
通过 `load_dotenv(override=True)` 加载；变量说明见 `.env.example`，不要提交真实密钥。

## 使用已有预测模型

以下示例读取产物中的工单事件，逐步输出最终成功/失败概率及辅助预测，默认返回 `continue`：

```bash
.venv/bin/python -m agentsentinel.early_lr predict \
  --output artifacts/early_lr_v1 \
  --events artifacts/early_lr_v1/example_events.json \
  --workflow ticket
```

输入必须是同一个任务从第1步开始的有序工具事件，工作流与任务一致。
高成功概率不表示任务已经完成，风险概率也不等于任务不可恢复。
输入字段与接口见[早期LR使用说明](docs/EARLY_LR_USAGE.md)；
22列模型的目标定义见[整个任务预测说明](docs/TASK_OUTCOME_LR_USAGE.md)。

## 复核原则

- 同一模板及其参数变体保持冻结分区，交叉验证按父任务家族分组，不能随机拆分前缀。
- 特征只使用当前及过去可观察的工具事件；标签、最终验证结果和故障注入元数据不进入模型。
- 每个会话保留 `session.json`、`events.jsonl`、`trajectory.json`、`result.json`，原始SSE随事件保存。
- 原始数据、协议、模型和产物封印保持原样；不能为通过检查而改写冻结哈希。
- 分别报告预测质量、告警时机、费用节省和成功损失；既有测试集探索与独立主实验分开解释。

进一步阅读：[数据管线](docs/EXPANDED_DATA_PIPELINE.md)、
[主实验结果](docs/EXPANDED_MODEL_RESULTS.md)、
[训练EDA](docs/EXPANDED_TRAINING_EDA.md)、
[早期LR复评结果](docs/EARLY_LR_TEST_RESULTS.md)。

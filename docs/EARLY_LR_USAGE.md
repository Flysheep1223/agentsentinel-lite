# 48列早期LR：使用与复核

本轮离线开发方案见 [EARLY_LR_PROTOCOL.md](EARLY_LR_PROTOCOL.md)。
最终成功预测仍以整段task的原始验收结果为标签；当前验收、剩余费用、剩余Token为独立辅助输出。
模型目录为 `artifacts/early_lr_v1`，输入是按顺序累积的可观察工具事件。
本阶段开发验证与[既有测试集复评](EARLY_LR_TEST_RESULTS.md)均已完成，当前默认只预测。
22列对照模型的特征数组接口见[task_outcome说明](TASK_OUTCOME_LR_USAGE.md)。

## 已有模型验证

```bash
.venv/bin/python -m agentsentinel.early_lr verify \
  --output artifacts/early_lr_v1 --check-inputs
```

核验模型、源码快照、分析方案、训练选策封印、原始轨迹和既有v2结果。
产物封印后不要修改其中的文件；需要重跑时使用新的输出目录。

## 默认只预测

每个task创建一个监测器，从第1个工具调用开始，按顺序传入事件。
命令行示例输入的工作流见产物 `example_context.json`：

```bash
.venv/bin/python -m agentsentinel.early_lr predict \
  --output artifacts/early_lr_v1 \
  --events artifacts/early_lr_v1/example_events.json \
  --workflow ticket
```

`--workflow` 必须与输入task一致，可取 `ticket` 或 `incident`。
输入为同一个task的JSON数组；输出为每步一行JSON。

```python
from agentsentinel.early_lr import load_monitor
from agentsentinel.early_features import project_event

monitor = load_monitor("artifacts/early_lr_v1", workflow="incident")
for raw_tool_event in tool_events:
    result = monitor.observe(project_event(raw_tool_event))
    print(result)
```

默认没有停止政策，无论概率高低都会返回 `continue`。
输出包含 `p_task_success`、`p_task_failure` 和三个辅助预测；
预测“成功概率高”不表示当前已经完成。

## 显式启用风险停止

在上述命令后增加 `--policy 'risk@0.01'`，或：

```python
monitor = load_monitor(
    "artifacts/early_lr_v1", workflow="incident", policy="risk@0.01"
)
```

`risk@0` 和 `risk@0.01` 分别是在训练OOF上按0%和1%成功任务误停预算选择的政策。
名称表示训练选择约束，不保证验证或未来task也满足该预算。
连续两步最终成功概率低于门槛，且预测资源满足门槛时才停止。
费用预测不可用时不自动停止；第12步不产生新的停止决定。
应用方收到 `decision="stop"` 后应终止该task的后续执行。

## 输入边界

`observe()` 严格接受以下字段：

```text
step, tool, arguments, response, status, error, schema_error, is_write,
latency_ms, elapsed_ms, totals
```

`totals` 只接受：

```text
input_tokens, output_tokens, reasoning_tokens, cached_tokens, total_tokens,
cost_estimate, cost_cny, cost_usd, currency
```

缺失用量使用JSON `null`，不能填0冒充已知值。
步骤必须从1开始连续递增；不允许把中途前缀当作完整历史重新开始。
`project_event()` 会从存档事件中移除额外元数据；输出的原始ID和正文只用于同参相等性、
公开读取字段完整性检查，不作为模型列。标签、验收结果和故障注入信息均不参与预测。

## 另行复现时的重新拟合

本阶段不再改进或重训模型。以下保留离线复现方式，不作为当前待办。

```bash
.venv/bin/python -m agentsentinel.early_lr train \
  --previous artifacts/stepwise_lr_v2 \
  --output artifacts/early_lr_reproduction \
  --workers 6
```

输出目录必须不存在。所有Python操作使用项目 `.venv`。
训练继承既有父家族分组和原始标签；每Session最多6个拟合前缀。
先完成训练OOF的模型与政策选择，再生成新验证预测；
不加载API凭据，不发出收费请求，也不生成新test预测。

本轮结果包含开发集重复使用，不能作为独立泛化证据。
原轨迹回放的费用节省不含监测器开销，不能直接等同于线上收益。

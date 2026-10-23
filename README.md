# AgentSentinel-Lite

按照 [项目提案](AgentSentinel_Lite_Project_Proposal.md) 实现的首版实验工程：在工具调用后，使用结构化运行指标预测最终失败风险，并比较固定规则、Logistic Regression 与 KNN 的停止策略。

目前已具备离线闭环与真实模型接口。`scripted` 是用于工程验证的确定性脚本，包含人为设置的循环、提前放弃和错误写入行为；其分数、估算 Token 和耗时**不能作为 LLM 研究结论**。真实模型采集需要在 `.env` 中填写接口配置。

## 安装与快速验证

所有 Python 命令使用项目 `.venv`；依赖版本由 `uv.lock` 固定。

```bash
uv sync --python 3.12 --extra dev --frozen
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/sentinel demo
```

`demo` 默认执行 40 个实例 × 5 种条件 × 3 次重复，即 600 个本地 Session，随后完成日志检查、特征提取、分组切分、训练、阈值选择、回放和报告生成。没有 API 请求。默认输出到新的 `data/demo_<时间>/`，亦可通过 `--output` 指定空目录。

已运行的工程演示位于 `data/demo_verified/`。主报告为 `reports/evaluation.json`，图表包括 `training_eda.png`、`prediction_curves.png` 和 `stopping_comparison.png`。

## 真实模型试运行

将 `.env.example` 复制为 `.env`，填写以下字段；加载使用 `load_dotenv(override=True)`。

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL=your-model
```

支持 OpenAI-compatible **Chat Completions SSE**，要求工具调用、`parallel_tool_calls=false`、`stream_options.include_usage`。当前使用 `max_tokens` 参数；仅支持其他参数形式的供应商需要适配。接口密钥不写入 Session 日志。

```bash
.venv/bin/sentinel collect \
  --backend openai --phase pilot \
  --limit-tasks 4 --conditions none timeout \
  --repeats 1 --workers 2 --output data/pilot_001
```

这会执行 8 个真实模型 Session。采集目录内 `experiment_<id>.json` 记录有效运行数、完成率、总 Token、耗时和按样本均值估算的 600 次运行预算。小规模任务的分布可能不具代表性，正式采集前应再覆盖两个工作流与全部故障条件。

价格字段 `INPUT_PRICE_PER_MILLION`、`OUTPUT_PRICE_PER_MILLION` 未设置时，美元成本保留为 `null`，阈值选择使用 Token 作为资源代理。Reasoning Token 属于供应商 completion Token 的一部分，不重复加到总量。缺少 reasoning/cache 字段保留缺失；缺少输入或输出用量会中止该 Session，并排除出训练数据。

Token 上限按供应商已报告用量阻止**下一次**请求，并限制单次最大输出；正在执行的请求仍可能因输入 Token 使总量超过上限。此上限不是 API 账单的精确预授权上限。

## 正式实验流程

先根据试运行调整任务、硬预算和故障强度，随后冻结配置，使用独立目录开始正式采集。`pilot` 数据不能进入正式训练，`scripted` 后端不能标记为 `collection`。

```bash
.venv/bin/sentinel tasks --output data/tasks.json

.venv/bin/sentinel collect \
  --backend openai --phase collection --repeats 3 --workers 4 \
  --fault-probability 0.7 --persistent-probability 0.3 \
  --output data/collection_001

.venv/bin/sentinel dataset \
  --runs data/collection_001 --output artifacts/dataset_001

.venv/bin/sentinel eda \
  --dataset artifacts/dataset_001 --output artifacts/eda_001

.venv/bin/sentinel train \
  --dataset artifacts/dataset_001 --output artifacts/models_001

.venv/bin/sentinel evaluate \
  --dataset artifacts/dataset_001 --models artifacts/models_001 \
  --output artifacts/test_001

.venv/bin/sentinel online \
  --backend openai --models artifacts/models_001 \
  --output data/online_001 --repeats 3 --workers 4 --budget 0.05
```

训练和最终测试是独立命令。训练集内采用按模板分组的交叉验证，验证集选择 5% 和 10% 错误中断预算的策略。测试结果不能用于继续修改本次实验的阈值。再次运行测试只会重新评价冻结策略，不自动调参。

`online` 仅运行冻结清单的测试任务：8 个实例 × 5 种条件 × 3 次重复 × 4 种策略，默认另需 480 个 Session 的预算。四种策略共享初始状态与故障种子，LLM 随机生成的轨迹仍可能不同。在线数据不会进入基线训练集。

消融实验使用相同数据和分区：

```bash
.venv/bin/sentinel train --dataset artifacts/dataset_001 \
  --output artifacts/models_no_tokens --ablation no_tokens
.venv/bin/sentinel train --dataset artifacts/dataset_001 \
  --output artifacts/models_no_errors --ablation no_errors
```

分别用 `evaluate` 评价这些冻结模型。`no_errors` 同时去掉错误计数、错误状态和重试计数，避免保留直接反映错误的代理字段。

## 数据与评价口径

- 两个本地工作流：工单处理、服务事件排查；共 20 个操作组合模板，每个模板两个参数变体。通知也仅写入本地模拟状态。
- 故障包括超时、限流、Schema 变化、部分响应，作用于读取工具；部分响应可以通过结构校验但缺失业务必需字段。暂不注入写操作“已提交但返回超时”的不确定状态。
- 超时和限流目前是立即返回的模拟错误，不等待真实网络超时；所记录延迟是本地实测执行时间。要研究真实超时等待造成的时间收益，需在正式试运行阶段加入并冻结延迟模型。
- 验证器同时检查最终状态、缺失操作、非预期写入和重复操作；最终状态恢复正确也不能抹掉违规写入历史。
- 按模板分层到两个工作流，切分为 12/4/4 个模板，即 360/120/120 个 Session。同一模板的故障运行和重复执行始终属于同一区。
- 模型特征使用白名单，不包含故障标签、随机种子、任务实例 ID、最终长度、剩余步骤、验证结果或思维链。Schema 错误仅来自当时实际执行的结构检查。
- 训练每个 Session 最多取 12 个均匀位置前缀；验证和测试回放保留所有可继续执行的决策位置。没有可用前缀的运行仍计入 Session 分母。
- 重试定义为：紧接上一条失败调用，且工具名和规范化参数完全一致。最近三步重复比例为重复签名数量除以窗口长度。
- Agent 的自然终止以没有工具调用的最终模型回复为准。验证器不会在中途告诉 Agent 或控制器“已经成功”。达到硬步骤、Token 或模型调用预算后的事件不计作预警样本。
- 阈值先满足验证集错误中断预算，再最小化执行成本；成本缺失时最小化 Token。并列时依次选择误中断更少、失败召回更多的策略。`threshold: null` 表示从不提前停止。
- PR-AUC 使用 Average Precision 定义。前缀指标仅作诊断，主要决策指标按 Session 计算，并报告模板 Bootstrap 区间、首次告警的步骤/Token/时间提前量和严格的违例前拦截率。
- 在写操作之后同一步告警，不能算作违例前拦截。回放中的成功误中断按“基线成功且产生停止”计数；在线完成率由实际停止时的状态验证。这两个指标含义不同。
- 离线评分延迟来自验证样本的单行测量，是估计值；在线记录实际特征提取和评分时间。预处理、序列化和并发调度的完整开销以在线结果为准。
- 当前成本是统一输入/输出单价下的估计，未应用供应商的缓存折扣、阶梯价格或额外费用。

## 文件结构

```text
src/agentsentinel/
  tasks.py          # 任务实例、模板与业务完成条件
  environment.py    # 本地工具、故障注入、确定性验证
  backends.py       # 脚本适配器与 OpenAI-compatible SSE 适配器
  harness.py        # 执行循环、预算、日志与在线监控
  features.py       # 在线/离线共用的因果前缀特征
  dataset.py        # 日志检查、模板切分、数据冻结与训练集 EDA
  modeling.py       # Pipeline、模型选择、规则、消融、冻结与测试
  evaluation.py     # Session 回放、阈值、提前量与 Bootstrap
  experiments.py    # 多线程采集及独立在线对照
  cli.py            # 命令行入口
tests/
docs/
```

每个 Session 目录：

```text
events.jsonl        # 原始事件；真实接口的 SSE 响应体字节以 Base64 保存
session.json        # 任务、初始状态、配置、种子、阶段、策略等元数据
trajectory.json     # 可读对话、工具调用、最终状态和写操作记录
result.json         # 验证结果、终止原因、Token、耗时及监控开销
```

拼接 `events.jsonl` 中 `kind == "llm_sse"` 的 `body_base64` 解码结果，可重建响应体；多轮调用以 `llm_start`/`llm_end` 分隔。原始数据与导出的 `prefixes.csv`、`sessions.csv` 分开保存。

`split.json` 固定分区和数据哈希，`policies.json` 固定模型、阈值、Agent 配置与故障规则。损坏日志会报错；基础设施失败、在线运行和启用策略的运行会明确排除。模型使用本项目生成的本地 joblib 文件。

后续实验安排与待完成事项见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

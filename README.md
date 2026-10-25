# AgentSentinel-Lite

在工具调用后使用结构化运行指标预测最终失败风险，比较固定规则、Logistic Regression 与 KNN 的停止策略。

**当前完成第一步：任务结构、采集前分区、交叉验证和标签对齐。尚未完成正式采集准备。** Session 级 120 秒截止、12 次调用／30,000 Token 默认预算、总费用 B 与完整费用核算仍待下一步落实。不要把下面的真实模型命令视为已获准启动收费实验。

`scripted` 是工程验证脚本，包含人为设置的循环、提前放弃和错误写入；其分数、估算 Token 和耗时不能作为真实 LLM 研究结论。本轮未调用付费 API。

## 安装与本地验证

所有 Python 命令使用项目 `.venv`，依赖由 `uv.lock` 固定。

```bash
uv sync --python 3.12 --extra dev --frozen
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/sentinel demo
```

`demo` 先冻结设计，再执行 40 实例 × 五种条件 × 三次重复，共 600 个脚本 Session，随后加工数据、训练、选择阈值、回放并生成报告，无 API 请求。默认输出到新的 `data/demo_<时间>/`；用 `--output` 指定其他空目录。工程验证的临时产物不作为正式数据交付。

## 任务结构与标签

两个工作流各五个结构模板，每模板四个参数实例，共 10 模板、40 实例。结构差异实际影响读取依赖与允许写入对象，不仅是操作组合或编号变化。

| 工单模板 | 依赖结构 | 服务事件模板 | 依赖结构 |
| --- | --- | --- | --- |
| ticket-00 | 单工单→服务状态→更新原工单 | incident-00 | 单服务状态与策略→事件 |
| ticket-01 | 原工单→关联工单→关联服务→写关联工单 | incident-01 | 服务→响应工单→关联服务→事件与工单备注 |
| ticket-02 | 读取两候选工单并比较影响分→只修改胜者 | incident-02 | 比较候选服务严重程度和错误数→只处理胜者 |
| ticket-03 | 按原服务状态分支，必要时转向升级工单 | incident-03 | 健康时通知原服务，否则处理依赖服务 |
| ticket-04 | 沿父工单链定位终点→只修改终点 | incident-04 | 汇总原服务和两个依赖的状态及策略→聚合事件 |

真实 Agent 仅接收公开请求及工具结果，不接收初始状态、预期写入或最终验证结果。脚本后端同样从已观察响应解析依赖。

标签由全部完成条件、完整写入历史和最终状态决定：全部满足且没有违规记成功，否则失败。终止原因独立记录；最后一次允许调用完成任务仍可成功，风险停止前已经完成也可成功。重复或非预期写入不能被最终状态恢复掩盖。基础设施无效运行单独排除，不直接作为任务失败训练样本。

## 采集前冻结设计

```bash
.venv/bin/sentinel freeze --output data/design_001.json --seed 42
```

冻结文件包含完整任务目录、每工作流 3／1／1 模板的训练／验证／测试分区（合计 6／2／2）、三个显式 CV 留出折、pilot 和 online 的预选实例及指纹。每折留出两个工作流各一个训练模板。此时不冻结需在 pilot 后确定的模型配置与故障强度。

- 设计文件不允许覆盖；采集前验证其指纹及与当前任务目录的一致性。
- 每个训练模板预选排序首个实例，共 6 个 pilot 实例；每个测试模板预选前两个实例，共 4 个 online 实例。
- 每批采集在提交 Session 前保存 `design.json`，实验清单及 Session 记录 `design_hash`。
- 输出目录必须为空或仅含相同设计，拒绝混批与覆盖。
- 数据加工复用冻结分区，模板缺数据只报告覆盖不足，不重新分配。
- 旧版20模板日志、无设计哈希的历史日志不能混入新版数据。不要事后给旧日志补造预冻结记录。

`tasks --output <路径>` 只导出任务目录，不替代 `freeze`。

## 真实模型 pilot（预算控制完成后执行）

在 `.env` 中配置接口、准确模型 ID 和价格，使用 `load_dotenv(override=True)`：

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL=your-model
INPUT_PRICE_PER_MILLION=供应商输入单价
OUTPUT_PRICE_PER_MILLION=供应商输出单价
```

支持 OpenAI-compatible Chat Completions SSE，要求工具调用、`parallel_tool_calls=false`、`stream_options.include_usage`；当前发送 `max_tokens`，供应商参数差异需先适配。密钥不写入日志。

```bash
.venv/bin/sentinel collect \
  --design data/design_001.json --backend openai --phase pilot \
  --workers 2 --output data/pilot_001
```

pilot 固定为 6 个预选训练实例 × 五条件 × 一次，共30次；数据不可进入正式训练。`--limit-tasks` 仅适用于 demo，不允许截取列表代替 pilot 选样。单独的小规模协议探测入口尚未实现。

Reasoning Token 属于供应商 completion Token 的一部分，不重复累加；缺少 reasoning/cache 保留缺失；缺少输入或输出 usage 时中止并排除该运行。缺少价格时成本为 `null`，使用 Token 作为资源代理。

目前 Token 上限按已报告用量阻止下一次请求，并限制单次输出；本次请求仍可能因输入用量超限，不是账单预授权。现有费用汇总只覆盖有效分析运行，总费用核算及 B 控制仍待实现。

## 正式实验流程（后续阶段）

在训练模板 pilot 后冻结预算、故障强度和准确模型配置，核对总费用 B；分区及测试选样不根据结果调整。使用独立目录采集。

```bash
.venv/bin/sentinel collect \
  --design data/design_001.json --backend openai --phase collection \
  --workers 4 --fault-probability 0.7 --persistent-probability 0.3 \
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
  --output data/online_001 --workers 4 --budget 0.05
```

正式 collection 强制40实例、五条件、三次重复，共600次，拒绝 scripted 后端。计划训练／验证／测试分别360／120／120次；实际有效样本数另行报告。固定 CV 折缺模板或出现单类别时明确报错，不按标签重新分折。

online 复用冻结策略中的完整设计，仅用4个预选测试实例 × 五条件 × 两次重复 × 四策略，共160个新 Session，包含重新运行的完整基线。共享初始状态、故障规则和环境种子不能保证 LLM 轨迹相同。80次缩减及取消决策记录尚未实现；不得查看测试结果后临时改变选样。

验证集选择5%／10%错误中断预算下的阈值，测试只评价冻结策略。CLI 的 `--budget` 是错误中断率预算，不是 API 费用上限。

消融使用同一冻结数据：

```bash
.venv/bin/sentinel train --dataset artifacts/dataset_001 \
  --output artifacts/models_no_tokens --ablation no_tokens
.venv/bin/sentinel train --dataset artifacts/dataset_001 \
  --output artifacts/models_no_errors --ablation no_errors
```

分别通过 `evaluate` 评价。删除 Token／成本不等于删除全部长度信息；`no_errors` 同时移除错误计数、错误状态及重试计数。

## 数据与评价边界

- 四类故障为超时、限流、Schema变化和部分响应，作用于读取工具。超时／限流当前立即返回模拟错误，不模拟真实等待；写操作提交后返回超时不在当前范围。
- 特征使用白名单，不含故障标签、任务 ID、未来长度、验证结果或思维链；Schema错误仅来自当时结构检查。
- 每个 Session 训练前缀最多12个；验证／测试保留全部可继续位置，无前缀运行仍计入 Session 分母。
- 自然终止以无工具调用的最终模型回复为准；验证器不在中途告知成功。耗尽已实现的步骤、Token、模型调用预算后不产生告警样本。
- 首次告警按 Session 计算；同一步写入后告警不算违例前拦截。回放误中断与在线实际完成率分别报告。
- PR-AUC 使用 Average Precision；阈值先满足误中断预算，再最小化成本或 Token，包含永不停止策略。
- 当前 EDA 仅含故障失败率与 Token 箱线图简版；五类完整 EDA、逐模板报告、KNN规模耗时等仍待补齐。
- 离线评分延迟为估计，在线记录特征提取和评分时间；成本使用统一输入／输出单价，未覆盖缓存折扣等完整供应商计费规则。

每个 Session 保存 `events.jsonl`、`session.json`、`trajectory.json`、`result.json`。SSE 原始响应体以 Base64 保留；按 `llm_start`／`llm_end` 分隔重建。原始日志与 `prefixes.csv`／`sessions.csv` 分开保存。

`split.json` 保存原设计与分析文件哈希；`policies.json` 保存原设计、模型、阈值、Agent配置和故障规则。加载时校验一致性。

本轮进度与后续工作见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

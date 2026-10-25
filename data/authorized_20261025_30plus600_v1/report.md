# 30 Pilot + 600 正式 Session 采集报告

## 结论与交付

采集已完成：先完成独立30个训练模板pilot，审查后冻结原配置，再完成独立600个正式Session。共630个唯一Session、5,093次真实模型请求；没有补跑无效样本、复用pilot充数、自动API重试或执行未授权的160次在线对照。

实验根目录：`/Users/flysheep/Desktop/agentsentinel-lite/data/authorized_20261025_30plus600_v1`。

**最终费用和未知用量以本目录 `summary.json` 为准。** `pilot/summary.json`、`collection/summary.json` 保留运行器原始统计；其中3个协议异常请求的usage已从原始SSE恢复并在根目录summary作追加对账，未修改原始Session、标签、有效性或冻结配置。

| 阶段 | Session | 分析有效 | 有效成功 | 有效失败 | 无效 | 实际请求 | 已知费用（CNY，对账后） | 仍未知usage请求 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot | 30 | 29 | 28 | 1 | 1 | 214 | 4.4543492 | 1 |
| 正式 | 600 | 593 | 478 | 115 | 7 | 4,879 | 93.4349350 | 4 |
| 合计 | 630 | 622 | 506 | 116 | 8 | 5,093 | 97.8892842 | 5 |

以上费用包含全部无效Session的已知支出，按用户给定价格估算，并非供应商最终账单。5次截止取消请求没有完整usage，额外费用未知，不能算作零。本轮无金额上限。既有5次试采不属于本轮630个Session，未覆盖其数据，费用不混入本表。

## 冻结设计与执行

- 模型：`kimi-k3-1-pro-261024`；接口：`https://ark.cn-beijing.volces.com/api/v3`。从本地配置读取，密钥不进入配置快照或实验产物。
- 10个结构模板，每模板4实例；预冻结训练/验证/测试模板数6/2/2。正式40实例 × 5条件 × 3次 = 600，分区360/120/120。
- pilot为6个预选训练模板各1实例 × 5条件 = 30；仅pilot审核用于冻结，没有使用验证/测试结果调参。
- 每Session最多12次工具调用、累计30,000输入+输出token、端到端120秒；最多24次模型请求、单次输出上限1024。reasoning属于输出，不重复计数。
- HTTP整体生命周期受剩余Session时长限制，流式心跳不能绕过120秒墙钟截止。token按返回usage累计检查，单个请求可能越过累计阈值；达到阈值后不再执行工具或下一请求。
- temperature 1.0；推理配置为供应商默认；没有发送额外thinking/reasoning覆盖参数。
- 故障概率0.7、持久概率0.3、瞬态窗口2，目标工具`get_service_status`和`lookup_policy`；种子42和既有派生规则未改变。timeout/rate_limit注入立即返回模拟错误，并非实际等待，没有额外sleep。
- pilot并发2、正式并发4；每Session独立环境。唯一计划、原子ledger及阶段独占锁；歧义started记录不自动重跑。
- pilot批次活动耗时968.005秒；正式批次活动耗时9,446.025秒（约2小时37分26秒）。正式Session耗时累计37,635.829秒，不能将并发累计时长当作批次墙钟时长。

冻结与审核证据：`design.json`、`protocol.json`、`pilot_review.json`、`frozen_collection.json`，以及阶段内`plan.json`、`ledger.json`。

- design SHA256：`761086b799fec580bc10854d4f3aa926f7145c26362d4c3fee0c997b8f747310`
- protocol SHA256：`a11db3ea7e3a2d5519b48965db4977a155028f95666905bc2e811da2c6bc6867`

## Pilot审核

29个分析有效：28成功、1工具步预算失败；1个在120秒截止时取消最后回复，usage缺失但实际写操作已完成，保留success=true、label=0，同时valid_for_analysis=false。无截断。20个Session实际触发故障，55次故障事件，18个有效成功属于故障恢复。

自动freeze入口拒绝含无效样本的pilot；本次对该唯一deadline样本作明确人工审核后写入审核记录及冻结文件。未补跑、未调任务或故障强度、未改变预算/输出上限，也未将无效样本伪装成有效。

## 正式结果

终止原因：completed 487、step_budget 50、output_truncated 56、time_budget 4、backend_error 3。没有token_budget终止。**终止原因不是成功标签**：标签由真实完成条件、完整写操作历史与最终状态决定，正常回复也可能任务失败，预算终止也不直接覆盖真实结果。

| 条件 | Session | 有效成功 | 有效失败 | 无效 | 实际触发故障Session |
|---|---:|---:|---:|---:|---:|
| 正常 | 120 | 99 | 21 | 0 | 0 |
| timeout | 120 | 90 | 28 | 2 | 112 |
| rate_limit | 120 | 91 | 28 | 1 | 109 |
| schema_change | 120 | 94 | 25 | 1 | 110 |
| partial_response | 120 | 104 | 13 | 3 | 78 |

正式共409个Session触发故障、1,157次故障事件；316个有效成功发生在触发过故障的Session。输出截断56次全部如实保留，不能作为正常完成；没有因为正式失败或截断比例调整冻结配置。

| 分区 | 实际Session | 分析有效 | 有效成功 | 有效失败 | 无效 |
|---|---:|---:|---:|---:|---:|
| train | 360 | 356 | 285 | 71 | 4 |
| validation | 120 | 118 | 106 | 12 | 2 |
| test | 120 | 119 | 87 | 32 | 1 |

7个正式无效Session：4个HTTP过程触及Session墙钟截止；3个供应商违反`parallel_tool_calls=false`返回多个工具调用，被拒绝执行并记为backend_error。这3个并非HTTP授权失败，不应写成网络或授权错误。没有触发“最近10个中3个非deadline基础设施异常”暂停阈值，也没有401/403记录。

## 用量与费用对账

| 已知token | pilot | 正式（SSE对账后） | 合计 |
|---|---:|---:|---:|
| 输入 | 304,646 | 7,325,374 | 7,630,020 |
| 输出（含推理） | 51,267 | 1,071,957 | 1,123,224 |
| 缓存输入（输入子集） | 255,744 | 6,383,590 | 6,639,334 |
| 推理（输出子集） | 37,229 | 748,046 | 785,275 |

计费CNY/百万token：非缓存输入13、缓存输入1.3、输出68。已知usage响应中没有缓存或reasoning字段缺失；5个取消请求的全部用量仍未知。缓存或reasoning是子集，不能重复加到总token。

运行器在拒绝并行工具回复时先抛协议异常，原始Session统计未接纳该请求usage；最终审计从已保存SSE恢复3个请求：新增已知输入3,737、输出693、缓存3,328、推理246、费用¥0.0567674。因此正式原始阶段summary的¥93.3781676应加该项，得到¥93.4349350；原“未知usage 7”对账后剩4。根目录summary列出每个恢复请求的Session ID、请求序号、token及金额，可复核。原始记录与有效性保持不变。

## 完整性、源码与测试

最终全量审计通过：630个唯一Session、每个阶段唯一计划、正式每实例15个Session、每条件120个Session；每Session保存`session.json`、`events.jsonl`、`trajectory.json`、`result.json`。对所有工具调用重放，核对响应、全部写历史、最终状态与标签；核对模型请求计数、工具调用预算及冻结代码哈希。未发现完整性错误。实际API密钥对全部实验文件和解码SSE扫描通过。

本轮代码范围：
- `src/agentsentinel/config.py`：默认预算、Session时长、CNY及缓存计费配置。
- `src/agentsentinel/backends.py`：完整HTTP墙钟取消、finish_reason和usage保存、原始SSE。
- `src/agentsentinel/harness.py`：预算边界、截断、有效性、已知/未知费用分离。
- `src/agentsentinel/collection_run.py`：唯一计划、恢复保护、阶段冻结、审计与汇总。
- `tests/test_collection_budgets.py`：预算、计费、超时、截断及630会话离线计划恢复测试。

离线测试实际结果：121 passed（末次8.66秒）；ruff通过；git diff --check通过。没有commit，没有恢复用户删除的旧文件，没有改写实验中被冻结的源码。既有README/状态文档与旧USD建模管线未全面迁移；CNY运行的旧`cost_usd`字段为null，不填人民币。后续分析应使用本轮CNY字段及根目录费用对账，而非把旧USD字段当作零费用。

## 文件索引

- `summary.json`：最终总汇总、对账后费用与token、无效样本清单、未完成usage请求及审计结果。
- `pilot/report.md`、`pilot/summary.json`、`pilot/sessions.csv`：pilot阶段原始汇总。
- `collection/report.md`、`collection/summary.json`、`collection/sessions.csv`：正式阶段原始汇总；费用差异见上述追加对账。
- `pilot/<session_id>/`、`collection/<session_id>/`：完整原始数据（含Base64原始SSE）。

本次不执行模型训练、在线160次对照，也不对失败或无效Session补采。

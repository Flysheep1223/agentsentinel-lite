# 真实 LLM 五完整会话试采报告

> 仅真实 LLM 五会话试采，不能作为正式 pilot、collection 或训练数据；不得送入数据构建/训练流程。

- 批次状态：completed；计划 5，会话落盘 5，未启动 0。
- 有效成功 4；有效失败 1；基础设施无效 0。
- 模型：`kimi-k3-1-pro-261024`；实际温度：1.0。
- 冻结设计：seed=42，`761086b799fec580bc10854d4f3aa926f7145c26362d4c3fee0c997b8f747310`。
- 从冻结的六个训练模板各一实例名单取前五个，覆盖 incident、ticket；全部正常 condition=none，无故障注入。
- 通过现有 run_session 完整 Agent→LLM→工具→LLM 回环，顺序执行、每会话独立初态；并非五个 HTTP 请求。
- 共 31 次实际请求启动，输入 42638、输出 9703、总计 52341 tokens。
- 缓存输入 32256；reasoning tokens 7655（已包含在输出中，不重复计费；None 表示未知）。
- 累计会话耗时 314.259 秒。人民币费用估计 ¥0.836703，不是实际账单；原始 cost_usd 保持 null。

## 会话结果

|任务|结果|工具步骤|请求|输入|输出|缓存|Reasoning|秒|估计 CNY|终止|自然 stop|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|incident-01-0|success|6|7|9656|2528|6912|2037|82.180|0.216562|completed|True|
|incident-02-0|failure|7|8|11557|2582|8960|2092|87.946|0.220985|completed|False|
|incident-03-0|success|3|4|4741|1511|3328|1248|42.978|0.125443|completed|True|
|ticket-00-0|success|3|4|4644|692|3328|442|23.751|0.068490|completed|True|
|ticket-02-0|success|7|8|12040|2390|9728|1836|77.405|0.205222|completed|True|

## 预算与计费边界

- 仅本批 5 个 Session，不自动补跑失败/无效样本，不运行 30 pilot 或 600 collection；本轮不设置金额阈值。
- 运行时覆盖：max_steps=12、max_model_calls=24、max_tokens=30000、max_completion_tokens=1024、request_timeout=60.0 秒；不修改 .env 或源码。
- 无自动 API 重试。基础设施错误（认证、模型不存在、参数/协议不兼容、缺 usage 或其他 backend_error）立即停止后续会话；任务失败、工具错误和预算终止保留原始标签。
- 尚无硬或软 120 秒 Session 截止；HTTP timeout 是各网络阶段/读取等待超时，持续流式响应并非严格墙钟上限。
- 总 token 预算在模型返回后检查，单次输入及返回可使实际累计超过阈值；max_completion_tokens 限制每次输出。
- task success 与终止方式正交：工具最终状态正确才算成功；completed 仅表示回环停止，可能是模型提前结束或输出截断，并不等同于任务完成。自然结束另检查最后 SSE finish_reason=stop；预算终止单列。
- 人民币估算公式：(未缓存输入×13 + 缓存输入×1.3 + 输出×68)/1,000,000。缓存计数缺失时该次输入按全额估算；reasoning 已计入 output，不再相加。
- 缓存计数缺失请求数：0；若 usage/费用未知则总额为 null/未知，不记为 0；可核算小计 ¥0.836703。
- 原始 session.json、events.jsonl、trajectory.json、result.json 保持 harness 格式与标签。events 中 llm_sse 为可重建原始响应的 Base64 块，可能含 reasoning_content；本报告不展开思维内容。
- phase=demo、experiment_id 含 real_llm_probe；研究禁用标记见 metadata.json 与 summary.json。当前 dataset 构建器并未对 demo 强制拒收，因此本目录必须与正式研究数据隔离，不调用构建/训练入口。

## incident-01-0

- Session：`7947058336ca4da0b57084f546879431`。
- 原始标签 label=0；valid_for_analysis=True（仅 harness 技术有效性，不表示可用于正式研究）。
- 最终状态匹配=True；缺失写入=0；违规=0；预算终止=False。
- 缺失写入：`[]`
- 违规：`[]`
|步骤|工具|状态|工具耗时 ms|会话累计 ms|
|---:|---|---|---:|---:|
|1|get_service_status|ok|0.124|8787.826|
|2|get_ticket|ok|0.100|13813.669|
|3|get_service_status|ok|0.144|27289.544|
|4|lookup_policy|ok|0.090|31525.101|
|5|record_incident|ok|0.097|49667.653|
|6|append_note|ok|0.174|71436.377|

|请求|输入|输出|缓存|Reasoning|LLM 耗时 ms|finish_reason|
|---:|---:|---:|---:|---:|---:|---|
|1|993|302|0|243|8787.446|tool_calls|
|2|1122|76|768|17|5025.533|tool_calls|
|3|1274|419|1024|359|13475.469|tool_calls|
|4|1390|60|1024|1|4235.166|tool_calls|
|5|1503|537|1280|447|18142.248|tool_calls|
|6|1627|795|1280|708|21768.274|tool_calls|
|7|1747|339|1536|262|10742.958|stop|

## incident-02-0

- Session：`fa84afa5ea20402f81c4238fb055ea2d`。
- 原始标签 label=1；valid_for_analysis=True（仅 harness 技术有效性，不表示可用于正式研究）。
- 最终状态匹配=False；缺失写入=2；违规=2；预算终止=False。
- 缺失写入：`[{"tool": "record_incident", "arguments": {"service_id": "S-incident-02-0-a", "severity": "info", "action": "monitor"}}, {"tool": "send_notification", "arguments": {"service_id": "S-incident-02-0-a", "channel": "audit", "body": "service:S-incident-02-0-a:healthy:monitor"}}]`
- 违规：`[{"step": 6, "tool": "record_incident", "arguments": {"service_id": "S-incident-02-0-b", "severity": "info", "action": "monitor"}, "reason": "unexpected_or_duplicate_write"}, {"step": 7, "tool": "send_notification", "arguments": {"service_id": "S-incident-02-0-b", "channel": "audit", "body": "service:S-incident-02-0-b:healthy:monitor"}, "reason": "unexpected_or_duplicate_write"}]`
|步骤|工具|状态|工具耗时 ms|会话累计 ms|
|---:|---|---|---:|---:|
|1|get_service_status|ok|0.102|12367.816|
|2|get_service_status|ok|0.090|17383.102|
|3|get_service_status|ok|0.082|21683.800|
|4|get_ticket|ok|0.089|23988.577|
|5|lookup_policy|ok|0.077|33532.793|
|6|record_incident|ok|0.116|51073.608|
|7|send_notification|ok|0.201|55231.183|

|请求|输入|输出|缓存|Reasoning|LLM 耗时 ms|finish_reason|
|---:|---:|---:|---:|---:|---:|---|
|1|998|386|512|327|12367.474|tool_calls|
|2|1138|124|768|64|5014.947|tool_calls|
|3|1254|61|1024|1|4300.405|tool_calls|
|4|1370|59|1024|1|2304.506|tool_calls|
|5|1519|284|1280|225|9543.910|tool_calls|
|6|1632|542|1280|452|17540.402|tool_calls|
|7|1756|102|1536|1|4156.992|tool_calls|
|8|1890|1024|1536|1021|32713.945|length|

## incident-03-0

- Session：`018c70ebcce3463e8b932803c350b7c9`。
- 原始标签 label=0；valid_for_analysis=True（仅 harness 技术有效性，不表示可用于正式研究）。
- 最终状态匹配=True；缺失写入=0；违规=0；预算终止=False。
- 缺失写入：`[]`
- 违规：`[]`
|步骤|工具|状态|工具耗时 ms|会话累计 ms|
|---:|---|---|---:|---:|
|1|get_service_status|ok|0.115|14310.237|
|2|lookup_policy|ok|0.123|23166.698|
|3|send_notification|ok|0.108|37574.631|

|请求|输入|输出|缓存|Reasoning|LLM 耗时 ms|finish_reason|
|---:|---:|---:|---:|---:|---:|---|
|1|1000|521|512|462|14309.839|tool_calls|
|2|1129|292|768|234|8856.054|tool_calls|
|3|1240|562|1024|463|14407.532|tool_calls|
|4|1372|136|1024|89|5402.918|stop|

## ticket-00-0

- Session：`cdc29b5c21404530920a7982a578fff3`。
- 原始标签 label=0；valid_for_analysis=True（仅 harness 技术有效性，不表示可用于正式研究）。
- 最终状态匹配=True；缺失写入=0；违规=0；预算终止=False。
- 缺失写入：`[]`
- 违规：`[]`
|步骤|工具|状态|工具耗时 ms|会话累计 ms|
|---:|---|---|---:|---:|
|1|get_ticket|ok|0.111|9417.341|
|2|get_service_status|ok|0.088|15047.018|
|3|update_ticket|ok|0.089|20069.942|

|请求|输入|输出|缓存|Reasoning|LLM 耗时 ms|finish_reason|
|---:|---:|---:|---:|---:|---:|---|
|1|962|331|512|273|9416.915|tool_calls|
|2|1111|147|768|88|5629.262|tool_calls|
|3|1225|142|1024|54|5022.634|tool_calls|
|4|1346|72|1024|27|3680.167|stop|

## ticket-02-0

- Session：`2b992f4baefe47ba82cf4d87402ad412`。
- 原始标签 label=0；valid_for_analysis=True（仅 harness 技术有效性，不表示可用于正式研究）。
- 最终状态匹配=True；缺失写入=0；违规=0；预算终止=False。
- 缺失写入：`[]`
- 违规：`[]`
|步骤|工具|状态|工具耗时 ms|会话累计 ms|
|---:|---|---|---:|---:|
|1|get_ticket|ok|0.130|17676.763|
|2|get_ticket|ok|0.089|27743.445|
|3|get_ticket|ok|0.101|31838.198|
|4|get_service_status|ok|0.091|35935.470|
|5|lookup_policy|ok|0.091|51414.330|
|6|update_ticket|ok|0.089|63132.841|
|7|append_note|ok|0.070|68716.965|

|请求|输入|输出|缓存|Reasoning|LLM 耗时 ms|finish_reason|
|---:|---:|---:|---:|---:|---:|---|
|1|997|684|512|626|17676.343|tool_calls|
|2|1172|309|768|250|10066.293|tool_calls|
|3|1324|60|1024|1|4094.297|tool_calls|
|4|1476|61|1280|1|4096.909|tool_calls|
|5|1592|572|1280|513|15478.537|tool_calls|
|6|1705|368|1536|279|11718.171|tool_calls|
|7|1827|88|1536|1|5583.792|tool_calls|
|8|1947|248|1792|165|8687.612|stop|

## 核验

```json
{
  "checked_sessions": 5,
  "sessions": [
    {
      "task_id": "incident-01-0",
      "design_hash_match": true,
      "complete_harness_artifacts": true,
      "offline_tool_replay_matches": true,
      "verifier_label_unchanged": true,
      "no_fault_injection": true
    },
    {
      "task_id": "incident-02-0",
      "design_hash_match": true,
      "complete_harness_artifacts": true,
      "offline_tool_replay_matches": true,
      "verifier_label_unchanged": true,
      "no_fault_injection": true
    },
    {
      "task_id": "incident-03-0",
      "design_hash_match": true,
      "complete_harness_artifacts": true,
      "offline_tool_replay_matches": true,
      "verifier_label_unchanged": true,
      "no_fault_injection": true
    },
    {
      "task_id": "ticket-00-0",
      "design_hash_match": true,
      "complete_harness_artifacts": true,
      "offline_tool_replay_matches": true,
      "verifier_label_unchanged": true,
      "no_fault_injection": true
    },
    {
      "task_id": "ticket-02-0",
      "design_hash_match": true,
      "complete_harness_artifacts": true,
      "offline_tool_replay_matches": true,
      "verifier_label_unchanged": true,
      "no_fault_injection": true
    }
  ],
  "secret_scan_actual_key_and_decoded_sse": "passed",
  "request_count_reconciled": true,
  "original_labels_preserved": true,
  "source_env_readme_previous_probe_unchanged": true
}
```

交付目录：`/Users/flysheep/Desktop/agentsentinel-lite/sample_sessions_20261025_211146_823696`。每请求 raw usage、步骤延迟与总计详见 summary.json。

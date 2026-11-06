# 6,000条数据：离线审计与加载

本入口接收独立的 `data/expanded_20261031_6000_v3_auto`，从 6000 个计划项重新核对原始证据。
不读取 `.env`，不调用模型 API，不启动采集运行器，不修改原始文件。
数据审计与本阶段研究均已完成，研究结论见[最终报告](RESEARCH_REPORT_6000_FINAL.md)。
旧采集与无关试采已清除，保留范围见[仓库说明](REPOSITORY_LAYOUT.md)。

## 已完成的全量审计

2026-11-06：6000 个 Session、24000 个会话文件和 7 个控制文件通过核验，
无完整性不一致；源文件二次哈希一致。有效数据为 5894 个 Session、47184 个可决策前缀，
成功/失败为 4422/1472，排除 106 个基础设施无效 Session。
3 个请求从原始 SSE 恢复 usage，已知费用从 ¥1006.5229576 对账到 ¥1006.6024690，
另有 103 次请求费用未知。详见[审计报告](../artifacts/expanded_dataset_v1/audit_report.md)。
审计阶段新增 24 项测试、221 项采集与完整性回归测试通过；当时独立 CLI 加载验证和 lint 通过。
批次映射覆盖 500 条协议记录为 50 并发的运行、5500 条显式 10 并发的运行。

## 命令

已有 `artifacts/expanded_dataset_v1` 使用校验命令：

```bash
# 校验输出文件封印、冻结设计、标签关联、分组与特征契约：
.venv/bin/python -m agentsentinel.expanded_dataset verify \
  --dataset artifacts/expanded_dataset_v1

# 额外重新读取全部原始文件并核对 SHA-256：
.venv/bin/python -m agentsentinel.expanded_dataset verify \
  --dataset artifacts/expanded_dataset_v1 --verify-raw \
  --source data/expanded_20261031_6000_v3_auto
```

仅在另行重建时使用以下命令，输出目录必须尚不存在：

```bash
.venv/bin/python -m agentsentinel.expanded_dataset build \
  --source data/expanded_20261031_6000_v3_auto \
  --output artifacts/expanded_dataset_reproduction \
  --workers 6
```

审计发生错误时保留 `audit_plan.json` 和能完成的错误证据，
逐会话错误见 `integrity_issues.json`；不生成可加载的 `manifest.json`。
完成时保存所有源文件 SHA-256，并再次读取验证源文件在审计期间未变。
源目录迁移后可以用 `--source` 指定新位置；账本中的旧绝对路径按 `collection/<session_id>` 解析。

## 分析接口

```python
from pathlib import Path

from agentsentinel.expanded_dataset import load_expanded_dataset

data = load_expanded_dataset(Path("artifacts/expanded_dataset_v1"))
sessions = data["sessions"]
prefixes = data["prefixes"]
feature_columns = data["manifest"]["feature_columns"]

train = prefixes.loc[prefixes["split"].eq("train")]
X_train = train.loc[:, feature_columns]
y_train = train["label"]  # 1=失败，0=成功
parent_groups = train["parent_family"]
cv_folds = data["design"]["cv_folds"]
```

此示例只加载与选择数据，不训练模型。不能直接对 `prefixes` 做全列拟合：
其中还包含标签和分组键。交叉验证沿用 `cv_folds`；不能随机打散前缀或拆散父模板家族。
同一 Session 的多个前缀相关，采样、权重和指标应采用已冻结的 Session 级评价方案。
加载器可保留没有可决策前缀的有效任务；当前5,894条有效任务均有可决策前缀。

数据集封印使用独立的 `schema_version=4` / `expanded_runtime_dataset_v1`，
其中冻结设计仍为 schema 3。旧 schema 2 兼容入口不用于当前批次；
主实验已通过 `agentsentinel.expanded_comparison` 显式接入此加载器，
并采用独立冻结方案；命令与产物说明见[模型复核入口](EXPANDED_MODEL_USAGE.md)。

## 产物与使用范围

| 文件 | 内容 | 使用范围 |
| --- | --- | --- |
| `sessions.csv` | 有效 Session、最终标签、资源用量、前缀数量 | 任务级统计与标签关联 |
| `prefixes.csv` | 有效且可决策的工具调用前缀、22 个可观察特征 | 仅 allowlist 内特征可输入风险模型 |
| `all_sessions.csv` | 全部尝试；`observed_label` 保留环境验证结果 | 审计，不作为训练数据源 |
| `excluded_sessions.csv` | 基础设施无效尝试及排除原因 | 排除核对 |
| `requests.csv` | 每次请求的 recorded/recovered/unknown usage、CNY 估算 | 费用对账 |
| `batch_sessions.csv` | 实际执行批次、并发、时间、修订 | 批次效应检查，不作为模型特征 |
| `design.json` / `parent_design.json` / `split.json` | 完整冻结目录与父家族分组 | 保持分区与 CV |
| `audit_plan.json` / `source_hashes.json` | 核验范围、源码差异、原始文件 SHA-256 | 溯源 |
| `quality.json` / `audit_report.md` | 核验结果与费用差异说明 | 质量报告 |
| `manifest.json` | 文件哈希、特征契约与完成封印 | 加载入口 |

无效会话的所有前缀均被排除，即使前面某些工具调用正常、甚至环境最终状态已完成任务。
工具错误和业务失败本身不会导致排除。标签通过环境重放重新计算，不能用终止原因代替。

冻结协议未设置累计 Token、Session 时间和单次输出上限；本版本审计只接受这一自动采集协议。
每次模型响应由原始 SSE 重建，累计 token/cache/reasoning 用量及 CNY 费用逐步核对。
工具响应、状态、错误标记、写入、故障、最终状态及验证结果通过本地环境重放核对。
可决策前缀要求工具步数和模型调用数均未达到上限。

终止请求即使报协议错误，也可能有完整 SSE usage。满足完整结束、finish reason、合法 usage、
无错误载荷等条件时，费用单独标记为 `recovered`；重复 usage 只取最后一次，不重复求和。
恢复费用不修改原始 result、不改变有效性；未知费用保留为空，不能填零。所有费用均为单价估算，
并非供应商账单。

批次 ID 在中断前后可能重叠，实际归属通过 condition_id 和账本 started_at 所在时间区间确定。
早期批次没有显式 workers 字段时使用冻结协议值，同时标记 `workers_source=protocol`。
各批次/恢复记录中影响任务、环境、判定和请求的源码必须与当前重放源码一致；
数据分析模块变更单独记录，不放宽采集运行器的冻结检查。

## 训练分区 EDA

已完成的分析入口为 `.venv/bin/python -m agentsentinel.expanded_eda`，使用上述封印加载器，
只对训练分区做失败诊断、固定时点特征统计及批次检查。
结果见[训练EDA与建模依据](EXPANDED_TRAINING_EDA.md)；当前产物为
`artifacts/expanded_eda_v2`；被v2取代的初次EDA已清除，复现使用新的输出目录。

诊断表包含从私有任务答案及最终状态重放得到的字段，不能替代 `prefixes.csv` 作为模型输入。
特别是必要写入剩余数、违规类型、顺序错误和完成时点，均只能用于事后解释与评价。
验证器不独立验证必要读取；无需写入任务的写入/状态完成时点记为0。

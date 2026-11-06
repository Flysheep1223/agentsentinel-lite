# 仓库结构与清理记录

整理日期：2026-11-06。仓库以6,000次采集研究为当前工作范围，
入口为[最终报告](RESEARCH_REPORT_6000_FINAL.md)和[合作者指南](COLLABORATOR_GUIDE.md)。
全部文档的用途、阶段与模型入口见[文档导航](README.md)。

## 保留的研究依赖

| 目录                                       | 保留原因                                                     |
| ------------------------------------------ | ------------------------------------------------------------ |
| `data/expanded_20261031_6000_v3_auto`    | 唯一正式分析批次，包含原始日志、协议、设计、父设计副本及账本 |
| `artifacts/expanded_dataset_v1`          | 全量审计、有效任务与前缀表、原始文件哈希                     |
| `artifacts/expanded_eda_v2`              | 当前训练EDA，主实验冻结时的依据                              |
| `artifacts/expanded_model_comparison_v1` | 预先冻结的LR/KNN/规则主实验及消融                            |
| `artifacts/stepwise_lr_v1`               | 本批次逐步预测探索，也是v2的上游依赖                         |
| `artifacts/stepwise_lr_v2`               | 最终任务结果预测，早期LR探索的22列对照                       |
| `artifacts/early_lr_v1`                  | 冻结的48列LR及开发验证                                       |
| `artifacts/early_lr_test_v1`             | 既有测试集复评及结果封印                                     |

依赖从下游到上游为：

```text
early_lr_test_v1
└── early_lr_v1
    └── stepwise_lr_v2
        └── stepwise_lr_v1
            └── expanded_model_comparison_v1
                ├── expanded_dataset_v1
                │   └── expanded_20261031_6000_v3_auto
                └── expanded_eda_v2
```

这里的 `stepwise_lr_v1` 等目录均来自当前6,000条数据；
版本号较早不代表无用。下游校验会核对上游产物、源码快照、协议与依赖文件，
因此这些目录与各阶段 `*_PROTOCOL.md` 保持原样。

`src/agentsentinel/` 与 `tests/` 保留完整。一些旧采集、在线运行和报告模块
仍属于当前实验冻结时的源码集合，直接删掉会破坏现有模型的完整性复核。
它们不再作为当前使用入口，旧数据删除后也不保证旧入口可独立运行。
后续若精简代码，应作为新的版本迁移处理，不能改写现有封印来绕过检查。

`.env`、`.venv`、`.git`、`pyproject.toml`、`uv.lock`、
`papers/` 和原始提案保留。参考文献及提案仍有研究价值。

## 清理范围

以下10个目录与当前复核链无依赖，已从工作区清除：

| 路径                                       | 内容                    | 文件数 |
| ------------------------------------------ | ----------------------- | -----: |
| `data/authorized_20261025_30plus600_v1`  | 旧600条正式采集及pilot  |   2538 |
| `data/demo_initial`                      | 初始脚本演示数据        |   2419 |
| `data/demo_verified`                     | 脚本验证演示数据        |   3067 |
| `data/expanded_20261031_6000_v2`         | 已暂停的早期试采        |     96 |
| `sample_sessions_20261025_211146_823696` | 早期独立样例轨迹        |     24 |
| `artifacts/eda_v1`                       | 旧批次EDA               |     29 |
| `artifacts/model_comparison_v1`          | 旧批次模型比较          |     99 |
| `artifacts/online_80_plan_v1`            | 旧80次在线对照及报告    |    340 |
| `artifacts/report_final_v1`              | 旧报告配套产物          |      4 |
| `artifacts/expanded_eda_v1`              | 已由当前v2取代的初次EDA |     35 |

目录内共8651个文件、506127389字节内容，文件分配块合计约502MiB。
另清除4份过期文档：`IMPLEMENTATION_STATUS.md`、`RESEARCH_REPORT_DRAFT.md`、
`RESEARCH_REPORT_FINAL.md`、`RESEARCH_REVIEW.md`。
共移除8655个文件；这些计数不包含目录项，也不代表Git历史压缩后的大小。

README已改为当前研究入口，完成记录已整理，保留文档中指向旧报告的链接已更新。
封印产物和源码中的历史文字保持原样，以免改变已有证据指纹。

这是工作区清理：已跟踪文件表现为待提交删除；本次不暂存、不提交、不推送，
也不重写Git历史。历史提交中的旧数据仍占Git对象空间，工作区清理不会移除它们。
清除的未跟踪演示和试采文件没有Git恢复保证。

## 当前命令

在仓库根目录使用 `.venv`，以下复核无需API：

```bash
.venv/bin/python -m agentsentinel.expanded_dataset verify \
  --dataset artifacts/expanded_dataset_v1 --verify-raw
.venv/bin/python -m agentsentinel.expanded_comparison verify \
  --output artifacts/expanded_model_comparison_v1
.venv/bin/python -m agentsentinel.early_test verify \
  --output artifacts/early_lr_test_v1 --check-sources
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

依赖旧模型文件的在线脚本烟测在旧产物缺失时按测试原有逻辑跳过，
不通过删除测试或放宽校验掩盖依赖问题。当前数据和模型的封印复核仍须全部通过。
预测使用方式见[README](../README.md)与[早期LR说明](EARLY_LR_USAGE.md)。

## 数据清理时的复核记录

- 6,000条采集、24,007个源文件的全量哈希核验通过，完整性问题为0。
- 主实验封印复核通过，指纹保持 `0f922269eb60c1e02803a20b35fb815aacbba2a3dd70f5c8774d387867cd34ed`。
- 后续LR与既有测试复评的依赖及源文件校验通过，复评指纹保持 `91095ad0f7073b18de949c160b4ea2bd83b97fffacbe636cc157d945306b5a79`。
- `ruff check src tests`、`git diff --check`通过，23份入口及文档的139个本地链接全部有效。
- 全量回归为409项通过、1项跳过（191.07秒）；跳过项为依赖已清除旧模型文件的在线脚本烟测，沿用测试原有条件。

## 后续文档整理

全仓库检查覆盖原有36份Markdown与6份论文PDF，包括封印内报告和源码快照中的协议。
新增文档导航后为37份Markdown与6份PDF。
本次合并重复的审计段落，纠正已清除EDA的保留说明，将复现命令改用未占用目录，
并标明原始提案、历史v1、22列v2、48列探索及既有测试复评的用途。
阶段报告中的待办已更新为完成记录或未来研究方向，历史测试计数保留所属阶段。

5份冻结协议、产物内7份报告及5份协议快照保持字节不变；6份参考论文保持原版本。
现存结果说明与完整报告用途不同，继续保留，未为减少文件数删除可复核证据。
本次只整理文档；上方409项通过、1项跳过是数据清理时的代码回归记录。

本次文档复核通过：37份Markdown的221个本地链接有效，7处复现输出目录未被占用；
主实验及后续测试复评的封印检查通过，`git diff --check`通过。

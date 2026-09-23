# RobotOps AI

> 机器人运营数据分析智能 Agent

**已完成阶段：Phase 1「机器人运营数据分析基础模块」+ Phase 2「DeepSeek 运营数据分析 Agent」+ Phase 3「运营问题案例 RAG 知识库」+ Phase 4「LangGraph 多 Agent 工作流」+ Phase 5「Streamlit 可视化平台」+ Phase 6「运营问题整改闭环」**

- Phase 1（本地分析，已稳定）：把机器人运营数据（Excel / CSV）读进来、清洗干净、算出核心指标、识别异常并输出报告。
- Phase 2（大模型解读，新增）：先用 Phase 1 的 Pandas 模块把数据变成**结构化分析结果**，再交给 DeepSeek 生成
  「运营概览 / 关键发现 / 异常发现 / 可能原因 / 优化建议」的结构化结论。**原始 Excel 不会直接发送给大模型**。
- Phase 3（知识库检索，新增）：把 17 个模拟运营案例建成 ChromaDB 本地知识库，
  对「异常描述」做 Top-K 检索，把「**当前项目数据 + 历史相似案例**」一起交给 DeepSeek，
  并要求模型严格区分当前事实与历史参考。
- Phase 4（多 Agent 工作流，新增）：用 LangGraph 把分析、诊断、RAG、建议、报告拆成 5 个单一职责 Agent，
  并按「有无异常」自动分支：无异常直接出报告，有异常走完整诊断链路。
- Phase 5（可视化界面，新增）：Streamlit 页面「RobotOps AI 机器人运营智能分析平台」，
  支持 Excel/CSV 上传、数据预览、一键分析、运营概览、异常项目、AI 分析、RAG 案例与报告下载。
- Phase 6（整改闭环，新增）：异常可一键生成运营问题（ISSUE-YYYYMM-NNN），
  支持状态流转、处理结果、备注、整改后重新分析、**程序自动计算改善幅度**与问题关闭统计，
  问题数据保存在本地 SQLite（`data/issues.db`）。
- 仍然**没有** Docker 与外部数据库——本阶段只使用 Python 标准库 `sqlite3`。

---

## 1. Phase 1 能力范围

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| Excel / CSV 数据读取 | ✅ 已实现 | 支持 `.xlsx / .xlsm / .csv`，CSV 自动识别 UTF-8 / GBK 编码 |
| 机器人运营数据模型 | ✅ 已实现 | `robotops/models.py`：字段字典 + 记录模型 + 指标/异常模型 |
| 数据清洗 | ✅ 已实现 | 类型转换、业务边界校验、去重、缺失值处理，并输出问题清单 |
| 核心指标计算 | ✅ 已实现 | 项目数量、机器人数量、平均运行时长、故障率、平均满意度、总运营成本、平均节降率 |
| 异常识别 | ✅ 已实现 | 故障率 > 5%、满意度 < 85、运行率 < 80%、节降率 < 10% |
| 结果输出 | ✅ 已实现 | 控制台报告 + JSON / CSV / Excel / Markdown |
| DeepSeek API 接入 | ✅ Phase 2 已实现 | API Key 通过 .env 配置，模型名集中管理 |
| 机器人运营数据分析 Agent | ✅ Phase 2 已实现 | 结构化载荷 → DeepSeek → 结构化结论 |
| 结构化输出与异常处理 | ✅ Phase 2 已实现 | 5 分区 JSON、超时/重试/空返回/网络错误处理 |
| 运营问题案例知识库 | ✅ Phase 3 已实现 | `knowledge/` 下 17 个模拟案例，覆盖 12 类问题 |
| 文档加载与切分 | ✅ Phase 3 已实现 | Markdown 元数据解析 + 小节切块（含重叠） |
| 向量数据库（ChromaDB） | ✅ Phase 3 已实现 | 本地持久化、初始化/写入/检索/状态 |
| RAG 检索（Top-K 案例） | ✅ Phase 3 已实现 | 相关性门控 + 分类优先 + 相似度阈值 |
| Agent 融合当前数据与历史案例 | ✅ Phase 3 已实现 | 提示词强制区分四类内容 |
| 多 Agent 协作（5 个 Agent） | ✅ Phase 4 已实现 | 分析 / 诊断 / RAG / 建议 / 报告 |
| LangGraph 工作流编排 | ✅ Phase 4 已实现 | 条件分支：无异常直接出报告 |
| 工作流日志与错误处理 | ✅ Phase 4 已实现 | `[Agent] started/completed`、失败降级不抛裸 traceback |
| Streamlit 可视化平台 | ✅ Phase 5 已实现 | 上传 / 预览 / 一键分析 / 结果展示 / 报告下载 |
| 运营问题管理 | ✅ Phase 6 已实现 | 问题编号、异常指标、AI 诊断、RAG 案例、AI 建议 |
| 问题整改闭环 | ✅ Phase 6 已实现 | 待处理 → 处理中 → 待验证 → 已完成 → 已关闭 |
| 整改效果验证 | ✅ Phase 6 已实现 | 程序对比整改前后指标并自动计算改善幅度 |
| 问题统计 | ✅ Phase 6 已实现 | 待处理/处理中/已完成数量、关闭率、平均处理周期 |
| 本地持久化 | ✅ Phase 6 已实现 | SQLite（标准库 sqlite3，`data/issues.db`） |
| Docker 部署 | ❌ 未实现 | 后续阶段 |

## 2. 环境要求

- 操作系统：Windows 10 / 11（已在 Windows 11 + VS Code 验证）
- Python：3.10 及以上（开发环境为 Python 3.12.1）
- 依赖库：`pandas`、`numpy`、`openpyxl`（Phase 1 数据读写）、`chromadb`（Phase 3 向量库）；
  Phase 2 的 HTTP 调用与 .env 读取使用 Python 标准库实现，**没有引入额外依赖**

## 3. 目录结构

```text
RobotOps AI/
├── run_analysis.py                # 命令行入口：python run_analysis.py
├── main.py                        # Phase 2 入口：python main.py（Pandas 分析 + DeepSeek Agent）
├── run_workflow.py                # Phase 4 入口：python run_workflow.py（多 Agent 工作流）
├── requirements.txt               # 运行依赖
├── README.md                      # 本文档
├── .gitignore
├── .env.example                   # 环境变量模板（复制为 .env 后填写 API Key）
├── data/
│   ├── raw/
│   │   └── robot_operation_demo.xlsx   # 演示数据（Phase 1 测试用）
│   ├── chroma/                    # Phase 3 本地向量数据库（可重建，已 gitignore）
│   └── processed/                 # 预留目录（暂未使用）
├── knowledge/                     # Phase 3 运营问题案例知识库（17 个模拟案例）
│   ├── README.md                  # 知识库说明与案例格式
│   ├── equipment_fault/           # 设备故障类案例（4 个）
│   ├── maintenance/               # 维修与售后类案例（4 个）
│   ├── satisfaction/              # 用户满意度类案例（4 个）
│   └── operation/                 # 运营类案例（5 个）
├── output/                        # 运行结果输出目录（自动生成）
├── .streamlit/
│   └── config.toml                # Phase 5：Streamlit 主题与上传大小等配置
├── frontend/                      # Phase 5：可视化界面
│   ├── app.py                     # Streamlit 入口（侧边栏切换「分析工作台 / 运营问题」）
│   ├── views/analysis.py          # 分析工作台视图（上传/预览/分析/展示/下载/创建问题）
│   ├── views/issues.py            # 运营问题视图（列表/详情/处理/整改验证/关闭）
│   └── ui_helpers.py              # 展示数据整理与工作流调用桥接（纯函数，可单测）
│   └── issues_helpers.py          # 运营问题展示辅助（统计卡片/详情表/验证结果）
├── robotops/                      # 核心代码包
│   ├── __init__.py                # 包导出与版本信息
│   ├── __main__.py                # 支持 python -m robotops
│   ├── cli.py                     # 命令行参数解析与入口逻辑
│   ├── config.py                  # 路径 / 列名 / 校验边界 / 异常阈值（唯一配置源）
│   ├── console_io.py              # Windows 控制台 UTF-8 编码处理
│   ├── exceptions.py              # 自定义异常类型
│   ├── models.py                  # 机器人运营数据模型、指标模型、异常模型
│   ├── data_loader.py             # 数据读取与表头规范化
│   ├── data_cleaner.py            # 数据清洗
│   ├── metrics.py                 # 核心指标计算
│   ├── anomaly.py                 # 异常识别
│   ├── report.py                  # 控制台报告 + 结果文件导出
│   ├── pipeline.py                # 流程编排：读取 → 清洗 → 指标 → 异常 → 输出
│   ├── llm/                       # Phase 2：LLM 模块
│   │   ├── config.py              # 模型名/地址/超时/重试的唯一配置来源
│   │   ├── env_loader.py          # .env 读取（内置解析器，可选 python-dotenv）
│   │   ├── client.py              # DeepSeek 客户端（超时、重试、错误映射）
│   │   ├── prompts.py             # Agent 系统提示词与输出格式约束
│   │   ├── schema.py              # 结构化输出解析、校验与渲染
│   │   ├── errors.py              # LLM 错误类型（含排查建议）
│   │   └── logger.py              # 基础日志（密钥脱敏）
│   ├── rag/                       # Phase 3：RAG 知识库模块
│   │   ├── config.py              # 知识库路径 / 切分 / 检索 / Embedding 参数
│   │   ├── document_loader.py     # Markdown 案例加载、元数据解析、文本切分
│   │   ├── embedding.py           # 本地哈希 Embedding（无需下载模型）
│   │   ├── vector_store.py        # ChromaDB 向量库（初始化/写入/检索/状态）
│   │   ├── query_builder.py       # 异常识别结果 → 检索用异常描述
│   │   ├── retriever.py           # 相关性门控 + 分类优先 + Top-K 聚合
│   │   └── errors.py              # RAG 错误类型
│   └── agent/                     # Phase 2：机器人运营数据分析 Agent
│       ├── payload.py             # Phase 1 结果 → Agent 输入载荷（不含原始明细）
│       └── robot_ops_agent.py     # Agent 主体：载荷 + RAG 案例 → DeepSeek → 结构化结果
│   └── issues/                    # Phase 6：运营问题整改闭环
│       ├── models.py              # 问题模型、状态机、改善幅度计算
│       ├── store.py               # SQLite 持久化（标准库 sqlite3）
│       └── service.py             # 创建/流转/处理/验证/关闭/统计
├── scripts/
│   ├── generate_demo_data.py      # 生成演示数据 robot_operation_demo.xlsx
│   ├── build_knowledge_base.py    # Phase 3 知识库管理（rebuild/sync/search/queries/demo）
│   └── phase2_smoke_test.py       # Phase 2 离线自检（不消耗 API、不需要 Key）
├── app/                           # Phase 4：LangGraph 多 Agent 工作流层
│   ├── agents/                    # 5 个单一职责 Agent
│   │   ├── base.py                # Agent 基类（日志、异常捕获、降级跳过）
│   │   ├── data_analysis.py       # ① Data Analysis Agent（程序计算，不调用大模型）
│   │   ├── diagnosis.py           # ② Diagnosis Agent（大模型：异常原因推测）
│   │   ├── rag_agent.py           # ③ RAG Agent（复用 Phase 3 检索历史案例）
│   │   ├── recommendation.py      # ④ Recommendation Agent（大模型：优化建议）
│   │   └── report.py              # ⑤ Report Agent（大模型归纳 + 程序模板兜底）
│   └── graph/                     # LangGraph 编排层
│       ├── state.py               # 统一 State（含需求要求的 7 个字段）
│       ├── workflow.py            # 图构建、条件分支、运行与导出
│       ├── logging_utils.py       # [Agent] started/completed 日志
│       └── prompts.py             # 3 个 LLM Agent 的提示词
└── tests/                         # 单元测试与端到端测试（标准库 unittest）
    ├── _helpers.py                # 测试公共构造数据
    ├── test_data_loader.py
    ├── test_data_cleaner.py
    ├── test_metrics.py
    ├── test_anomaly.py
    ├── test_pipeline.py
    ├── test_env_loader.py         # Phase 2：.env 读取
    ├── test_llm_config.py         # Phase 2：LLM 配置与模型名集中管理
    ├── test_deepseek_client.py    # Phase 2：超时/重试/错误映射/日志脱敏
    ├── test_ai_schema.py          # Phase 2：结构化输出解析与校验
    ├── test_agent_payload.py      # Phase 2：Agent 载荷（不传原始 Excel）
    ├── test_agent.py              # Phase 2：Agent 数据流与 Prompt 约束
    ├── test_main_cli.py           # Phase 2：命令行与退出码
    ├── test_rag_document_loader.py # Phase 3：案例加载、元数据与切分
    ├── test_rag_embedding.py       # Phase 3：本地 Embedding 方案
    ├── test_rag_retriever.py       # Phase 3：向量库与 Top-K 检索
    ├── test_rag_agent.py           # Phase 3：Agent 区分当前数据与历史案例
    ├── test_workflow.py            # Phase 4：工作流路由 / State / 日志 / 错误处理
    ├── test_workflow_cli.py        # Phase 4：工作流命令行与退出码
    ├── test_frontend_helpers.py    # Phase 5：前端展示辅助函数
    ├── test_frontend_app.py        # Phase 5：Streamlit 应用（AppTest 无头端到端）
    ├── test_issues.py              # Phase 6：问题创建/流转/处理/验证/关闭/统计
    └── test_issues_page.py         # Phase 6：运营问题页面与创建入口
```

## 4. 安装依赖

在项目根目录（`D:\Vibe Coding\RobotOps AI`）打开终端执行：

```powershell
# 1) 创建虚拟环境（可选但推荐）
python -m venv .venv

# 2) 激活虚拟环境（PowerShell）
.\.venv\Scripts\Activate.ps1
#   如提示执行策略受限，可改用：.\.venv\Scripts\activate.bat

# 3) 安装依赖
python -m pip install -r requirements.txt
```

VS Code 提示：按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `.venv` 中的 Python，
这样集成终端与编辑器使用同一个解释器。

## 5. 快速开始

```powershell
# 第 1 步：生成演示数据（输出到 data/raw/robot_operation_demo.xlsx）
python scripts\generate_demo_data.py

# 第 2 步：运行分析（默认读取演示数据，结果输出到 output/）
python run_analysis.py
```

运行后会看到「数据清洗结果 → 核心指标 → 项目维度汇总 → 异常识别 → 已导出文件」五段报告。

使用自己的数据：

```powershell
python run_analysis.py --data "D:\我的数据\机器人运营.xlsx"
python run_analysis.py --data "D:\我的数据\机器人运营.csv" --encoding gbk
```

## 6. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `-d, --data` | 数据文件路径（`.xlsx / .xlsm / .csv`），默认 `data/raw/robot_operation_demo.xlsx` |
| `-o, --output` | 结果输出目录，默认 `output/` |
| `--sheet` | 指定 Excel 工作表名称 |
| `--encoding` | 指定 CSV 编码（如 `gbk`），不指定则自动识别 |
| `--list-sheets` | 只列出 Excel 的工作表名称 |
| `--no-export` | 只打印报告，不导出文件 |
| `--no-excel` | 导出结果但不生成 Excel 汇总工作簿 |
| `-q, --quiet` | 隐藏过程日志，只输出最终报告 |
| `--fault-rate-upper` | 故障率上限阈值（%），默认 5 |
| `--satisfaction-lower` | 满意度下限阈值（分），默认 85 |
| `--uptime-rate-lower` | 运行率下限阈值（%），默认 80 |
| `--saving-rate-lower` | 节降率下限阈值（%），默认 10 |
| `--fault-rate-scope` | 故障率判定口径：`robot_period`（默认）/ `record` |

## 7. 数据模型（字段说明）

演示数据文件 `data/raw/robot_operation_demo.xlsx` 含 4 个工作表：
`运营数据`、`字段说明`、`项目与机器人`、`数据说明`。

| 字段 | 类型 | 单位 | 是否必需 | 说明 |
| --- | --- | --- | --- | --- |
| 日期 | 日期 | — | 必需 | 每台机器人每天一条记录 |
| 项目名称 | 文本 | — | 必需 | 指标「项目数量」按其去重统计 |
| 机器人ID | 文本 | — | 必需 | 指标「机器人数量」按其去重统计 |
| 机器人类型 | 文本 | — | 必需 | 清洁 / 巡检 / 安防巡逻 / 配送 / 消杀 / AGV搬运 |
| 运行时长 | 数值 | 小时 | 必需 | 当日实际运行时长 |
| 计划运行时长 | 数值 | 小时 | 可选 | 用于计算运行率；缺失时按机器人类型默认值兜底（默认 24 小时） |
| 故障次数 | 整数 | 次 | 必需 | 当日故障次数 |
| 巡检次数 | 整数 | 次 | 必需 | 当日巡检/作业任务次数，作为故障率分母 |
| 维修次数 | 整数 | 次 | 必需 | 当日维修次数 |
| 用户满意度 | 数值 | 分 | 必需 | 有效区间 0-100 |
| 运营成本 | 数值 | 元 | 必需 | 当日运营成本 |
| 节降率 | 数值 | % | 必需 | 成本节降比例 |

派生字段（由清洗阶段自动计算，原始数据无需提供）：

- `运行率` = 运行时长 ÷ 计划运行时长 × 100%
- `故障率` = 故障次数 ÷ 巡检次数 × 100%

表头兼容性：读取时会自动把常见别名映射为标准列名，例如
`项目 → 项目名称`、`机器人编号 → 机器人ID`、`满意度 → 用户满意度`、`成本 → 运营成本`、`节能率 → 节降率` 等。

## 8. 数据清洗规则

清洗流程固定、可复现，每一步发现的问题都会写入 `output/robot_ops_report_cleaning_issues.csv`：

1. **表头规范**：去空格 / 全角空格 / BOM，按别名映射标准列名，校验必需列是否齐全；
2. **结构清理**：删除完全为空的行；删除与数据模型无关的全空列；
3. **文本清洗**：关键文本字段去除首尾及重复空白；项目名称、机器人ID 为空的记录剔除；
4. **日期解析**：支持日期文本与 Excel 序列号；无法解析的记录剔除；
5. **类型转换**：数值字段统一转 float，支持带千分位逗号、百分号、全角符号的文本；
6. **业务边界校验**：
   - 运行时长 < 0 → 置空；
   - 单日运行时长 > 24 小时 → 记录警告但保留（便于人工复核口径）；
   - 计数类字段（故障/巡检/维修次数）为负 → 置空；
   - 用户满意度超出 0-100 → 置空且不计入平均值；
   - 运营成本为负 → 置空且不计入合计；
   - 节降率超出 -100% ~ 100% → 置空；
   - 计划运行时长缺失或 ≤ 0 → 按机器人类型默认值补齐（保证运行率可算）；
7. **缺失值处理**：故障次数 / 巡检次数 / 维修次数缺失按 0 处理；满意度、运行时长、运营成本、节降率缺失时保留记录并跳过计算；
8. **去重**：先删除完全重复的行，再按「日期 + 机器人ID」去重并保留最后一条（同一机器人同一天只保留一次上报）；
9. **派生列计算**：运行率、故障率；分母为 0 时保留空值；
10. **排序**：按「日期 + 机器人ID」升序输出。

## 9. 指标口径

| 指标 | 单位 | 计算口径 |
| --- | --- | --- |
| 项目数量 | 个 | 对「项目名称」去重计数 |
| 机器人数量 | 台 | 对「机器人ID」去重计数 |
| 平均运行时长 | 小时 | 清洗后记录的「运行时长」算术平均值（跳过空值） |
| 故障率 | % | 总故障次数 ÷ 总巡检次数 × 100%（累计口径） |
| 平均满意度 | 分 | 有效「用户满意度」记录的算术平均值 |
| 总运营成本 | 元 | 有效「运营成本」记录的求和 |
| 平均节降率 | % | 有效「节降率」记录的算术平均值 |

另外附带 3 项补充统计（不属于验收指标，仅用于说明规模与运行健康度）：
平均运行率、记录数、总故障次数。项目维度汇总复用同一套口径，与总体指标一致。

## 10. 异常识别规则

| 异常类型 | 判定条件 | 判定粒度 |
| --- | --- | --- |
| 故障率偏高 | 故障率 > 5% | 机器人 + 分析周期累计 |
| 满意度偏低 | 用户满意度 < 85 分 | 逐日记录 |
| 运行率偏低 | 运行率 < 80% | 逐日记录 |
| 节降率偏低 | 节降率 < 10% | 逐日记录 |

**严重程度分级**：偏差幅度 = |实际值 − 阈值| ÷ |阈值|；
偏差 ≥ 50% 记为「高」，≥ 20% 记为「中」，其余记为「低」。

**为什么故障率按周期累计判定？**
故障率是比率型指标，如果按单日判定，分母很小（例如 1 次故障 ÷ 8 次巡检 = 12.5%），
5% 阈值会把几乎所有机器人、几乎所有日期都判成异常，异常清单失去定位价值
（实测：逐条判定会命中 88 条「故障率偏高」，占全部异常的 57%，涉及 19/23 台机器人）。
因此默认采用「机器人 + 分析周期」累计口径（累计故障次数 ÷ 累计巡检次数），
与总体指标口径保持一致，异常清单也能直接定位到具体机器人。

如需改回逐条判定，可使用命令行参数：

```powershell
python run_analysis.py --fault-rate-scope record
```

判定指标为空值的记录（例如巡检次数为 0 导致故障率无法计算）不会命中异常，
但会在清洗问题清单中体现，避免被静默忽略。

## 11. 输出文件说明

运行 `python run_analysis.py` 后，`output/` 目录会生成以下文件：

| 文件 | 内容 |
| --- | --- |
| `robot_ops_report_metrics.csv` | 核心指标表（指标 / 数值 / 单位 / 类型 / 计算口径） |
| `robot_ops_report_project_summary.csv` | 项目维度汇总 |
| `robot_ops_report_anomalies.csv` | 异常明细（逐条记录） |
| `robot_ops_report_cleaning_issues.csv` | 数据清洗问题清单 |
| `robot_ops_report_metrics.json` | 指标 + 清洗统计 + 阈值（供后续阶段或其他程序调用） |
| `robot_ops_report.xlsx` | Excel 汇总工作簿（9 个工作表，见下） |
| `robot_ops_report.md` | Markdown 版完整分析报告 |

Excel 工作簿包含 9 个工作表：核心指标、指标口径、项目汇总、异常规则、异常汇总、异常项目汇总、异常明细、清洗统计、清洗问题。
所有 CSV 使用 `utf-8-sig` 编码，Windows 下双击可直接用 Excel 打开且不乱码。

## 12. 如何验证运行结果

> 说明：下面各阶段章节中提到的用例数（36 / 109 / 147 / 164 / 189）是该阶段完成时的历史数量；
> **当前（Phase 6 完成后）全量回归共 221 个用例**，统一用最后一条命令运行。

### 方式一：对照控制台的已知答案

演示数据由固定随机种子（`20260801`）生成，因此结果可复现。执行
`python scripts\generate_demo_data.py` 与 `python run_analysis.py` 后，应看到：

| 校验项 | 预期值 |
| --- | --- |
| 原始记录数 | 697（23 台机器人 × 30 天 + 7 条脏数据） |
| 剔除记录数 | 2（1 条完全重复 + 1 条重复上报） |
| 清洗后记录数 | 695 |
| 日期范围 | 2026-08-01 至 2026-08-31 |
| 项目数量 / 机器人数量 | 6 / 23 |
| 平均运行时长 | ≈ 11.39 小时 |
| 故障率 | ≈ 2.06% |
| 平均满意度 | ≈ 90.91 分 |
| 总运营成本 | ≈ 1,111,739.60 元 |
| 平均节降率 | ≈ 16.73% |
| 异常命中记录数 | 77 条（节降率偏低 32、满意度偏低 27、运行率偏低 15、故障率偏高 3） |

> 说明：脏数据位于数据末尾（前 2 条与正常记录同日期同机器人、后 5 条日期为 2026-08-31），
> 具体注入内容见演示数据「数据说明」工作表。
> 若修改了 `--seed / --days / --start-date`，数值会随数据变化。

另外可以核对这几处逻辑是否生效：

- 「满意度越界」记录的 120 分被置空，未拉高平均满意度；
- 「运营成本为负」的 -300 元被置空，未冲减总运营成本；
- 「项目名称」带空格的记录被归并回原项目，项目数量仍为 6；
- 7 类脏数据都能在清洗问题清单中找到对应条目。

### 方式二：检查输出文件

```powershell
dir output
```

应看到 7 个结果文件；用 Excel 打开 `output/robot_ops_report.xlsx`，
检查「异常明细」工作表的「判定粒度」列：故障率为「机器人周期累计」，其余为「逐日记录」。

### 方式三：运行测试

```powershell
python -m unittest discover -s tests -t . -v
```

共 36 个测试用例，覆盖数据读取（含编码、别名、异常提示）、数据清洗（去重、越界、缺失值、派生列）、
指标计算（精确数值校验）、异常识别（四条规则、严重程度、空值跳过）与端到端流程（生成数据 → 分析 → 导出 → 命令行退出码）。
预期结果：`Ran 36 tests ... OK`。

## 13. 常见问题

**Q1：控制台中文乱码？**
程序启动时会自动把 Windows 控制台代码页切换为 UTF-8 并重配置标准输出编码。
若仍有乱码，请在 VS Code 终端执行 `chcp 65001` 后重试。

**Q2：报错「文件被占用或无写入权限」？**
通常是 Excel 正打开着 `output/robot_ops_report.xlsx` 或数据文件。关闭 Excel 后重试。

**Q3：提示不支持 `.xls`？**
Phase 1 只支持 `.xlsx / .xlsm / .csv`。请在 Excel 中「另存为」`.xlsx` 后再分析。

**Q4：提示「数据缺少必需列」？**
报错信息会列出缺失列与实际列名。请对照第 7 节的字段表补齐列名，
或使用已支持的别名（如 `成本`、`满意度`）。

**Q5：CSV 打开乱码 / 读取失败？**
程序会依次尝试 `utf-8-sig / gbk / utf-8 / utf-16`。若仍失败，可用 `--encoding gbk` 显式指定。

**Q6：为什么我的数据里很多记录被判为「满意度偏低」？**
这是阈值本身的结果（默认 < 85 分）。可放宽阈值验证，例如：
`python run_analysis.py --satisfaction-lower 80`。

## 14. Phase 2：DeepSeek 运营数据分析 Agent

### 14.1 数据流（原始 Excel 不会直接发给大模型）

```text
Excel / CSV
  → Phase 1 Pandas 分析（读取 → 清洗 → 指标计算 → 异常识别）
  → 结构化载荷 output/ai_input_payload.json
  → DeepSeek API（chat/completions，JSON 输出模式）
  → 结构化分析结果 output/ai_analysis.json / .md
```

载荷中只包含**结构化结果**（核心指标、项目汇总、异常项目、异常机器人、清洗统计、口径说明），
不含逐条明细，既控制请求体积，也避免大模型重新计算指标。

### 14.2 配置 DeepSeek API（通过 .env）

```powershell
# 1) 复制模板
copy .env.example .env

# 2) 编辑 .env，填入真实密钥
#    DEEPSEEK_API_KEY=sk-你的真实密钥
```

支持的变量（默认值统一定义在 `robotops/llm/config.py`，代码中没有任何硬编码密钥或模型名）：

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | ✅ 必填 | 无 | DeepSeek API Key，只从环境变量读取 |
| `DEEPSEEK_BASE_URL` | 可选 | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | 可选 | `deepseek-chat` | 模型名称 |
| `DEEPSEEK_TIMEOUT` | 可选 | `60` | 单次请求超时（秒） |
| `DEEPSEEK_MAX_RETRIES` | 可选 | `2` | 失败自动重试次数（指数退避） |
| `DEEPSEEK_TEMPERATURE` | 可选 | `0.2` | 采样温度 |
| `DEEPSEEK_MAX_TOKENS` | 可选 | `4096` | 单次最大输出长度 |
| `DEEPSEEK_LOG_LEVEL` | 可选 | `INFO` | 日志级别（日志写入 output/deepseek_llm.log） |
| `ROBOTOPS_ENV_FILE` | 可选 | 项目根目录 `.env` | 自定义 .env 位置 |

安全约定：

- `.env` 已写入 `.gitignore`（连同 `.env.*`，仅保留 `.env.example`），不会被提交；
- 日志中的密钥会自动脱敏（形如 `sk-abcd****wxyz`）；
- 进程环境变量优先于 `.env`，可用于 CI/服务器注入密钥。

### 14.3 启动

```powershell
# 完整流程：Pandas 分析 + DeepSeek 解读（需要 .env 中的 API Key）
python main.py

# 只打印将发送给模型的结构化载荷，不调用 API（无需 Key、不产生费用）
python main.py --dry-run

# 查看实际发送的 Prompt（system + user）
python main.py --dry-run --show-prompt

# 指定数据文件 / 阈值 / 输出目录
python main.py --data "D:\我的数据\运营数据.xlsx" --satisfaction-lower 88 --output output

# 以 JSON 输出 AI 结果（便于二次处理）
python main.py --json

# Phase 1 入口保持不变
python run_analysis.py
```

退出码约定：`0` 成功；`2` 输入/配置问题（含缺少 API Key、数据列缺失）；`3` AI 调用失败（网络/超时/接口错误/返回格式异常）。

### 14.4 输出文件

| 文件 | 内容 |
| --- | --- |
| `output/ai_input_payload.json` | 实际发送给 DeepSeek 的结构化载荷（可审计） |
| `output/ai_analysis.json` | AI 结构化分析结果（overview / key_findings / abnormal_projects / possible_reasons / recommendations + meta） |
| `output/ai_analysis.md` | AI 分析结果的 Markdown 版本 |
| `output/deepseek_llm.log` | LLM 调用日志（请求、耗时、Token、重试、错误） |
| `output/deepseek_raw_response.txt` | 仅在返回内容无法解析为 JSON 时生成，便于排查 |
| `output/robot_ops_report.*` | Phase 1 结果文件（默认同时导出，可用 --no-export 关闭） |

### 14.5 Agent 输出结构

```json
{
  "overview": "2~4 句整体结论",
  "key_findings": [{"finding": "结论", "evidence": "引用的数值证据", "metric": "指标名"}],
  "abnormal_projects": [{"project": "项目名", "abnormal_type": "异常类型", "metric_value": "异常值", "threshold": "阈值", "severity": "高/中/低", "evidence": "证据"}],
  "possible_reasons": [{"reason": "推测原因", "confidence": "高/中/低", "based_on": "判断依据", "data_gap": "缺失数据或“当前数据不足以判断”"}],
  "recommendations": [{"action": "建议动作", "priority": "高/中/低", "target": "对象", "expected_effect": "预期效果", "verification": "验证方式"}]
}
```

Agent 的系统提示词（`robotops/llm/prompts.py`，版本 `phase2-agent-v1`）强制约束：

1. 只能基于提供的结构化结果分析；
2. 不得虚构数据；
3. 不得修改或重新计算程序已算好的核心指标；
4. 必须区分数据事实、异常发现、可能原因（推测）、优化建议；
5. 数据不足时必须写明「当前数据不足以判断」并指出缺少哪些字段；
6. 不允许把推测写成确定事实（禁止「是因为」「导致」「证明」式表述）。

程序侧还会做输出校验：缺少字段、类型不符或出现额外字段时，会在结果中记录 `warnings` 并在报告末尾提示。

### 14.6 如何测试 Phase 2

```powershell
# 1) 离线自检：用模拟响应跑通全链路（不需要 Key、不消耗 API 额度）
python scripts\phase2_smoke_test.py
# 预期：自检结果 11/11 项通过

# 2) 干跑：确认发送给模型的数据是否符合预期
python main.py --dry-run --show-prompt

# 3) 完整单元测试（Phase 1 + Phase 2 共 109 个用例）
python -m unittest discover -s tests -t . -v

# 4) 真实 API 联调（需要 .env 中已配置 Key）
python main.py --data data/raw/robot_operation_demo.xlsx
```

### 14.7 完整测试示例（从零到出结果）

```powershell
cd "D:\Vibe Coding\RobotOps AI"
.\.venv\Scripts\Activate.ps1                      # 可选：使用虚拟环境
python -m pip install -r requirements.txt          # 安装依赖（pandas / numpy / openpyxl）
copy .env.example .env                             # 复制环境变量模板
notepad .env                                       # 把 DEEPSEEK_API_KEY 改为真实密钥后保存

python scripts\generate_demo_data.py               # 生成演示数据（697 条）
python scripts\phase2_smoke_test.py                # 离线自检 11/11（不调用 API）
python main.py --dry-run --quiet                   # 确认结构化载荷
python main.py --quiet                             # 真实调用 DeepSeek 并输出结论
```

第 5 步成功后应看到：

- 控制台依次输出「运营概览 / 关键发现 / 异常发现 / 可能原因（推测）/ 优化建议」；
- `output/` 目录新增 `ai_input_payload.json`、`ai_analysis.json`、`ai_analysis.md`；
- `output/deepseek_llm.log` 中记录本次调用的模型、耗时与 Token 用量。

### 14.8 API 调用失败排查

| 现象 / 提示 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `[配置错误] 未检测到环境变量 DEEPSEEK_API_KEY` | 没有 .env 或变量名写错 | 复制 `.env.example` 为 `.env` 并填写 `DEEPSEEK_API_KEY`；确认文件在项目根目录 |
| `401 认证失败` | Key 错误、过期，或值里带了引号/空格 | 重新生成 Key；确认 `.env` 中没有多余字符；`--dry-run` 可先避开 API |
| `402 账户余额不足` | 账户余额为 0 | 登录 DeepSeek 平台充值 |
| `429 触发限流` | 请求过于频繁 | 程序会自动指数退避重试；可调大 `DEEPSEEK_MAX_RETRIES` 或稍后再试 |
| `请求 DeepSeek 超时` | 网络慢或输出过长 | 调大 `DEEPSEEK_TIMEOUT`（如 120），或减小 `DEEPSEEK_MAX_TOKENS` |
| `无法连接 DeepSeek` / DNS 失败 | 断网、代理、公司防火墙 | 浏览器访问 https://api.deepseek.com 验证；必要时配置 `HTTP_PROXY` / `HTTPS_PROXY` |
| `请求被拒绝（HTTP 400/404）` | 模型名或地址不对 | 检查 `DEEPSEEK_MODEL`（如 `deepseek-chat`）与 `DEEPSEEK_BASE_URL` |
| `返回内容为空` | 模型返回被截断 | 调大 `DEEPSEEK_MAX_TOKENS`，或根据提示减小载荷 |
| `返回内容不是合法 JSON` | 模型未按格式输出 | 原始返回已保存到 `output/deepseek_raw_response.txt`，重试一次通常即可恢复 |
| 只想确认程序没问题、不想花额度 | 未配置 Key 或不想调用 | 使用 `python scripts\phase2_smoke_test.py` 与 `python main.py --dry-run` |

排查顺序建议：先 `--dry-run` 确认输入载荷正常 → 再 `--quiet` 真实调用 → 失败时查看
`output/deepseek_llm.log`（含重试与状态码）与 `output/deepseek_raw_response.txt`。

## 15. Phase 3：机器人运营问题案例 RAG 知识库

### 15.1 RAG 架构说明

```text
Excel / CSV
  → Phase 1：Pandas 分析（读取 → 清洗 → 指标计算 → 异常识别）
  → 生成异常描述（query_builder.py：把异常指标写成运营问题）
  → RAG 检索（ChromaDB 向量库 + 本地 Embedding，Top-K 历史案例）
  → 组装提示词：当前项目数据 + 历史相似案例
  → DeepSeek 生成：当前事实 / 历史参考 / 推测原因 / 建议措施
  → output/ai_analysis.json + ai_analysis.md（含历史案例章节）
```

模块职责（全部位于 `robotops/rag/`，**未改动 Phase 1 / Phase 2 的任何既有逻辑**）：

| 模块 | 职责 |
| --- | --- |
| `config.py` | 知识库目录、ChromaDB 目录、切分参数、Top-K、相似度阈值、Embedding 方案（唯一配置源，支持 .env 覆盖） |
| `document_loader.py` | 读取 `knowledge/**/*.md`，解析元数据（front matter）与 5 个必填小节，按小节切块并保留元数据 |
| `embedding.py` | 本地哈希 Embedding（字符 2-3 gram + 英文词 + 领域词，L2 归一化，cosine 空间） |
| `vector_store.py` | ChromaDB 本地持久化向量库：初始化、重建、增量写入、删除、相似度检索、索引指纹校验 |
| `query_builder.py` | 把 Phase 1 的异常明细转成检索用「异常描述」（含异常类型、实际值、阈值、严重程度、项目层面指标） |
| `retriever.py` | 相关性门控 → 分类优先检索 → 相似度阈值过滤 → 按案例聚合 Top-K |

与 Phase 2 的关系：**Phase 2 的默认行为完全不变**（不传知识库即纯数据分析）；
只有显式启用 RAG 时，提示词才切换为 `phase3-rag-v1`。
禁用方式：`python main.py --no-rag`。

### 15.2 Embedding 方案说明（为什么不用 DeepSeek）

**结论：DeepSeek 当前不提供 embedding 接口，因此本方案使用本地 Embedding。**

DeepSeek 开放平台提供的是 Chat Completions（对话补全）接口，
没有 `POST /embeddings` 这类文本向量化接口，无法用 API Key 生成向量；
因此 Phase 3 采用**本地哈希向量**方案（`robotops/rag/embedding.py`）：

| 项目 | 说明 |
| --- | --- |
| 方案名称 | `robotops-local-hashing-v1` |
| 维度 | 1024（可配置） |
| 特征 | 字符 2-gram / 3-gram（中文主特征）+ 英文/数字单词 + 领域业务词加权 |
| 相似度 | 余弦相似度（ChromaDB `cosine` 空间，相似度 = 1 − 距离） |
| 依赖 | 仅 NumPy，**不下载任何预训练模型**，无网络也能跑 |
| 确定性 | 同一文本永远得到同一向量，索引可重建、结果可复现 |
| 与 API Key 的关系 | 不需要任何 API Key；密钥仍然只通过 .env 提供给 DeepSeek 对话接口 |

为什么不用 `sentence-transformers` / `text-embedding-*`：
前者需要下载数百 MB 的模型与 torch 依赖，后者需要额外的向量化服务或 API；
本项目的检索对象是**篇幅短、术语密集**的中文案例，字符 n-gram + 领域词加权已经能稳定区分
「故障 / 运行率 / 满意度 / 节降率 / 闲置 / 调度」等主题（见 15.7 的自检结果）。

如需替换为其他 Embedding：修改 `robotops/rag/config.py` 的 `EMBEDDING_BACKEND` 并在
`embedding.py` 的 `build_embedding_function` 中扩展实现，然后执行 `rebuild` 重建索引
（索引指纹变化时程序会提示需要重建）。

### 15.3 知识库说明

```text
knowledge/
├── README.md              # 知识库说明与案例格式
├── equipment_fault/       # 设备故障类：重复故障、维修频繁、维护周期过长、电池老化
├── maintenance/           # 维修与售后类：售后响应慢、备件不足、返修、维保排期
├── satisfaction/          # 满意度类：清洁效果、投诉响应、噪音、巡检不到位
└── operation/             # 运营类：运行率下降、设备闲置、调度异常、成本过高、节降率下降
```

- 案例数量：**17 个**，覆盖需求要求的 12 类问题
  （机器人重复故障、运行率下降、巡检不到位、满意度下降、清洁效果下降、设备维修频繁、
  售后响应慢、运营成本过高、节降率下降、设备闲置、任务调度异常、设备维护周期过长）；
- **重要声明**：所有案例均为**模拟业务知识（虚构内容）**，不代表任何真实企业数据，
  每个案例的元数据都带 `data_nature: 模拟案例（虚构）`，检索结果会把该标记一并传给模型；
- 每个案例包含：**问题、现象、可能原因、处理措施、处理结果**（另含「经验总结」），
  以及元数据：`case_id / title / case_type / category / severity_hint / applicable_anomaly /
  robot_types / keywords / data_nature`；
- 切分方式：按 Markdown 小节（`## 问题` 等）切块，超长小节二次切分（默认 480 字符、重叠 120 字符），
  每个文本块都保留案例编号、标题、类型、分类、命中片段等元数据，可溯源到具体案例；
- 索引规模：17 个案例 → **102 个文本块**（本地 ChromaDB，`data/chroma/`，已 gitignore，可随时重建）。

### 15.4 如何添加案例文档

1. 在对应分类目录下新建 `.md` 文件（建议文件名与 `case_id` 一致，例如 `CASE-OPS-006.md`）；
2. 按 `knowledge/README.md` 的模板写元数据与 5 个必填小节：

```markdown
---
case_id: CASE-OPS-006
title: 案例标题
case_type: 机器人运行率下降
category: operation
keywords: [运行率下降, 待机, 排班]
data_nature: 模拟案例（虚构）
---

# CASE-OPS-006 案例标题

## 问题
## 现象
## 可能原因
## 处理措施
## 处理结果
```

3. 让新案例进入向量库（二选一）：

```powershell
python scripts\build_knowledge_base.py sync      # 增量写入新案例（推荐）
python scripts\build_knowledge_base.py rebuild   # 全量重建（改过 Embedding/切分参数时用）
```

4. 校验：`python scripts\build_knowledge_base.py list` 能看到新案例；
   `python scripts\build_knowledge_base.py search "你的问题"` 能检索到它。

### 15.5 如何构建 / 重建知识库

```powershell
python scripts\build_knowledge_base.py rebuild    # 清空并重建索引（首次初始化用这个）
python scripts\build_knowledge_base.py sync       # 增量同步（新增/删除案例）
python scripts\build_knowledge_base.py status     # 查看集合、案例数、块数、Embedding、是否需要重建
python scripts\build_knowledge_base.py list       # 列出全部案例

# 也可以直接用主入口的开关
python main.py --rebuild-knowledge                # 重建索引后退出
python main.py --rag-status                       # 查看索引状态
```

可调参数（`.env`，默认值见 `robotops/rag/config.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ROBOTOPS_KNOWLEDGE_DIR` | `knowledge` | 知识库目录 |
| `ROBOTOPS_CHROMA_DIR` | `data/chroma` | 向量库持久化目录 |
| `ROBOTOPS_RAG_TOP_K` | `5` | 每次查询召回块数 |
| `ROBOTOPS_RAG_MAX_CASES` | `6` | 最多注入提示词的案例数 |
| `ROBOTOPS_RAG_MIN_SIMILARITY` | `0.35` | 相似度阈值（过滤不相关命中） |
| `ROBOTOPS_EMBEDDING_DIM` | `1024` | 向量维度（改动后需 rebuild） |
| `ROBOTOPS_RAG_AUTO_BUILD` | `true` | 索引缺失时自动构建 |

### 15.6 在 Agent 中使用（Phase 2 → Phase 3 的新流程）

```powershell
python main.py                       # 默认启用 RAG：分析 → 抽异常描述 → 检索案例 → DeepSeek
python main.py --no-rag              # 关闭知识库，退化为纯 Phase 2 流程
python main.py --dry-run             # 不调用 API，打印载荷 + 检索到的历史案例（可先验证 RAG）
python main.py --rag-top-k 3 --rag-min-similarity 0.4
python main.py --search "B小区机器人故障率明显升高，同时维修次数增加。"
```

提示词新增的强制规则（`robotops/llm/prompts.py`，版本 `phase3-rag-v1`）：

1. 结论必须**优先依据当前项目数据**，历史案例只能作为参考；
2. 历史案例是**模拟案例（虚构）**，严禁把其中的数值/项目名/设备编号/时间/成本当作当前项目事实；
3. 引用案例必须写明案例编号（如 `[CASE-FAULT-001]`），并说明相似与不相似之处；
4. 必须区分四类内容：**当前项目事实 / 历史相似案例 / 推测原因 / 建议措施**；
5. 没有足够相关案例时必须明确写「**未检索到足够相关的历史案例**」；
6. 案例处理结果不得写成对当前项目的确定性承诺。

结构化输出的新增字段：`possible_reasons[].historical_reference`、`recommendations[].reference_case`
（无对应案例时填 `"无"`）。报告中的历史案例独立成章节「历史相似案例（RAG 检索）」，
并标注「模拟案例，仅供参考」，与当前项目数据分区展示。

### 15.7 如何测试 RAG

**方式一：离线自检（不需要 API Key、不消耗额度）**

```powershell
python scripts\build_knowledge_base.py demo
```

实测结果（10/10 通过）：

| 类型 | 问题 | 结果 |
| --- | --- | --- |
| 相关 | B小区机器人故障率明显升高，同时维修次数增加。 | [CASE-FAULT-001] 相关度 0.472 |
| 相关 | 机器人运行率下降，设备长时间停在充电位。 | [CASE-OPS-001] 相关度 0.498 |
| 相关 | 用户满意度下降，投诉说地面清洁不干净。 | [CASE-SAT-001] 相关度 0.536 |
| 相关 | 运营成本偏高，节降率下降。 | [CASE-OPS-004] 相关度 0.588 |
| 相关 | 设备闲置，任务分配不均。 | [CASE-OPS-002] 相关度 0.533 |
| 相关 | 巡检不到位，出现漏检点位。 | [CASE-SAT-004] 相关度 0.514 |
| 无关 | 今天天气怎么样？ / 帮我写一首诗 | 未命中（符合预期） |
| 无关 | 机器人多少钱一台 / Python 怎么读取 Excel 文件 | 未命中（符合预期） |

**方式二：单独测试检索模块**

```powershell
python scripts\build_knowledge_base.py search "B小区机器人故障率明显升高，同时维修次数增加。"
python scripts\build_knowledge_base.py queries    # 用演示数据的异常描述走一遍真实检索流程
python main.py --search "机器人运行率下降"
```

`queries` 会展示 Agent 的真实流程：Phase 1 分析 → 生成 4 条异常描述 → 检索并输出 Top-K 案例
（实测命中 6 个案例，故障率/运行率/满意度/节降率各自匹配到对应案例）。

**方式三：验证 Agent 是否区分「当前数据」与「历史案例」**

```powershell
python main.py --dry-run --show-prompt     # 查看提示词：当前项目数据区块 + 历史案例区块
python -m unittest tests.test_rag_agent -v
```

**方式四：完整单元测试（Phase 1 + 2 + 3 共 147 个用例）**

```powershell
python -m unittest discover -s tests -t . -v
```

其中 RAG 相关用例覆盖：

- **正常问题能检索到相关案例**：6 条业务问题均命中预期案例（`test_rag_retriever.py`）；
- **无关问题不返回错误案例**：4 条无关问题全部返回空结果，且给出「未检索到足够相关的历史案例」说明；
- **Agent 正确区分当前数据与历史案例**：提示词中当前数据区块与历史案例区块各自独立，
  当前指标来自 `core_metrics`，历史内容以 `[CASE-xxx]` 编号出现，且载荷中不混入案例
  （`test_rag_agent.py`）；
- 另外覆盖：案例元数据与 12 类问题是否齐全、长文本切分、Embedding 确定性与归一化、
  索引自动构建、参数变化触发「需要重建」、增量写入与按案例删除、无相关案例时的提示。

### 15.8 RAG 常见问题排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `未安装 ChromaDB` | 依赖未安装 | `python -m pip install -r requirements.txt`；或加 `--no-rag` 跳过 |
| `知识库目录不存在` / `没有案例文件` | 目录或文档缺失 | 确认 `knowledge/` 下有分类目录与 `.md` 案例 |
| `案例缺少 case_id / case_type / 必需小节` | 文档格式不符合模板 | 对照 `knowledge/README.md` 补齐元数据与 5 个必填小节 |
| `案例编号重复` | 两个文件用了同一 `case_id` | 修改其中一个 |
| 提示「需要重建」（stale） | 改了 Embedding 维度或切分参数 | `python scripts\build_knowledge_base.py rebuild` |
| 检索结果为空 | 问题与运营无关；或阈值过高；或案例不足 | 先用 `search` 验证；必要时调低 `ROBOTOPS_RAG_MIN_SIMILARITY`（如 0.30） |
| 命中案例不相关 | 阈值过低或案例质量不足 | 调高阈值、补充更贴近的案例、丰富案例 `keywords` |
| `main.py` 启动时提示「已跳过 RAG」 | 知识库不可用，已自动降级为 Phase 2 | 按提示修复；不影响分析主流程 |

### 15.9 Phase 3 新增输出文件

| 文件 | 内容 |
| --- | --- |
| `output/ai_rag_retrieval.json` | RAG 检索明细：检索查询、跳过原因、命中案例、相似度、命中片段 |
| `output/ai_analysis.json` | 其中 `meta.retrieved_cases` 记录本次参考的历史案例 |
| `output/ai_analysis.md` | 新增「历史相似案例（RAG 检索）」章节 |

## 16. Phase 4：LangGraph 多 Agent 工作流

### 16.1 Multi-Agent 架构图

```text
                     ┌──────────────────────────────┐
   Excel / CSV  ───► │ ① Data Analysis Agent        │  Phase 1：读取 → 清洗 → 指标 → 异常
                     │   （程序计算，不调用大模型）   │
                     └───────────────┬──────────────┘
                                     ▼
                     ┌──────────────────────────────┐
                     │  Abnormal Check（程序判定）   │  故障率 > 5% / 满意度 < 85
                     │  按阈值判断，LLM 不参与       │  运行率 < 80% / 节降率 < 10%
                     └───────┬───────────────┬──────┘
                     无异常  │               │  有异常
                             ▼               ▼
                             │   ┌──────────────────────────────┐
                             │   │ ② Diagnosis Agent            │  大模型：异常原因推测
                             │   │   （只解释原因，不给建议）    │
                             │   └───────────────┬──────────────┘
                             │                   ▼
                             │   ┌──────────────────────────────┐
                             │   │ ③ RAG Agent                  │  Phase 3：检索历史案例
                             │   │   （保留案例来源与标题）      │
                             │   └───────────────┬──────────────┘
                             │                   ▼
                             │   ┌──────────────────────────────┐
                             │   │ ④ Recommendation Agent       │  大模型：当前数据+诊断+案例
                             │   └───────────────┬──────────────┘
                             └───────────┬───────┘
                                         ▼
                     ┌──────────────────────────────┐
                     │ ⑤ Report Agent               │  大模型归纳（失败时程序模板兜底）
                     └───────────────┬──────────────┘
                                     ▼
                                    END

输出：output/workflow_report.md（报告）
      output/workflow_state.json（完整 State）
      output/workflow_steps.json（节点轨迹 + 错误）
      output/workflow.log（工作流日志）
```

### 16.2 Agent 职责（单一职责，不重复劳动）

| # | Agent | 是否用大模型 | 输入 | 输出 | 职责边界 |
| --- | --- | --- | --- | --- | --- |
| ① | Data Analysis Agent | ❌ 程序 | 数据文件路径 + 阈值 | `metrics` / `abnormal_projects` / `anomaly_summary` / `abnormal_robots` / `raw_data_summary` | 只做 Phase 1 分析与汇总，不做任何主观判断 |
| ② | Diagnosis Agent | ✅ | 当前指标 + 程序判定的异常 | `diagnosis`（summary + possible_reasons，含依据与缺失数据） | 只解释原因，**不给建议、不写报告、不重新判定异常** |
| ③ | RAG Agent | ❌ 程序 | Phase 1 分析结果（生成异常描述） | `retrieved_cases`（案例编号 / 标题 / 类型 / 来源文件 / 相关度） | 只做检索与整理，不解读案例 |
| ④ | Recommendation Agent | ✅ | 当前数据 + 诊断 + 历史案例 | `recommendations`（动作 / 优先级 / 验证方式 / 参考案例） | 只给建议，不写报告，不修改指标 |
| ⑤ | Report Agent | ✅（可降级） | 全部工作流产出 | `final_report`（Markdown 报告） | 只做汇总写作，不新增事实 |

复用关系：①调用 Phase 1；②④⑤复用 Phase 2 的 `DeepSeekClient`（超时/重试/错误映射）与日志；
③复用 Phase 3 的 `KnowledgeBaseRetriever`。**没有重复实现任何既有能力**。

### 16.3 LangGraph 工作流

节点与边（`app/graph/workflow.py`）：

| 节点 | 说明 | 出边 |
| --- | --- | --- |
| `data_analysis` | ① 数据分析 | 条件边：成功 → `abnormal_check`；失败 → `report` |
| `abnormal_check` | 程序判定异常（阈值来自配置） | 条件边：有异常 → `diagnosis`；无异常 → `report` |
| `diagnosis` | ② 诊断 | `rag` |
| `rag` | ③ 历史案例检索 | `recommendation` |
| `recommendation` | ④ 优化建议（前序 LLM 失败时自动跳过） | `report` |
| `report` | ⑤ 报告汇总 | `END` |

```python
from app.graph.workflow import RobotOpsWorkflow

state = RobotOpsWorkflow().run(data_path="data/raw/robot_operation_demo.xlsx")
print(state["final_report"])
```

### 16.4 State 设计

统一 State 定义在 `app/graph/state.py`（`WorkflowState`），需求要求的 7 个字段全部包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `raw_data_summary` | dict | 数据源、工作表、分析窗口、记录数、清洗统计、阈值 |
| `metrics` | dict | `values`（核心指标与补充统计）+ `units`（单位） |
| `abnormal_projects` | list | 异常项目汇总（来自程序判定） |
| `diagnosis` | dict | 诊断结论（推测原因 / 置信度 / 依据 / 缺失数据 / 来源） |
| `retrieved_cases` | list | 历史案例（含 `case_id`、`title`、`case_type`、`source_file`、`similarity`、`case_summary`） |
| `recommendations` | dict | 优化建议（动作 / 优先级 / 验证方式 / 参考案例 / 来源） |
| `final_report` | str | 最终 Markdown 报告 |

辅助字段：`anomaly_summary`、`abnormal_robots`、`data_quality`、`retrieval`、`has_anomalies`、
`anomaly_count`、`abnormal_check_report`、`workflow_status`、`steps`（节点轨迹）、`errors`、
`failed_stages`、`skipped_stages`、`report_meta` 等。

> 注意：LangGraph 只会保留 State 中**声明过的字段**，新增字段需同步补充到 `WorkflowState`。

### 16.5 异常分支

1. **异常判定完全由程序完成**：`abnormal_check` 节点读取 Phase 1 的异常识别结果（阈值来自
   `robotops/config.py`），大模型不参与"是否异常"的判断，也不能修改阈值；
2. **无异常**：`abnormal_check → report`，不执行诊断 / RAG / 建议（测试 1 验证）；
3. **有异常**：走完整链路 `diagnosis → rag → recommendation → report`（测试 2 验证）；
4. **没有检索到案例**：工作流继续执行，并在报告与建议提示中明确写「未检索到足够相关的历史案例」（测试 3 验证）；
5. **大模型调用失败**：记录结构化错误（含错误类型、消息、排查建议），后续 LLM Agent 自动跳过，
   报告改用程序模板生成并在报告中标注降级原因（测试 4 验证）；
6. **知识库不可用**：记录 RAG 错误并降级为不使用历史案例，工作流仍产出报告；
7. **数据分析失败**：直接进入报告节点，报告写明失败原因，`workflow_status = failed`。

### 16.6 运行方式

```powershell
python run_workflow.py                      # 完整工作流（需要 .env 中的 DeepSeek Key）
python run_workflow.py --no-llm             # 离线模板模式：不调用大模型，零消耗
python run_workflow.py --no-rag             # 关闭知识库检索
python run_workflow.py --json               # 以 JSON 输出完整 State
python run_workflow.py --data "D:\数据\运营.xlsx" --rag-top-k 3
```

退出码：`0` 成功；`2` 输入/配置问题（数据、知识库、缺少 API Key）；`3` 大模型调用失败（报告已降级生成）。

日志示例（控制台与 `output/workflow.log`）：

```text
[Workflow] started - 数据源=robot_operation_demo.xlsx 知识库=启用 大模型=启用
[Data Analysis Agent] started
[Data Analysis Agent] completed - 项目 6 个 / 机器人 23 台 / 异常 77 条
[Abnormal Check] 程序判定命中异常 77 条（阈值：…），进入诊断流程
[Diagnosis Agent] started
[Diagnosis Agent] completed - 输出 4 条推测原因
[RAG Agent] started
[RAG Agent] retrieved 6 cases
[RAG Agent] completed - retrieved 6 cases
[Recommendation Agent] started
[Recommendation Agent] completed - 输出 4 条优化建议
[Report Agent] started
[Report Agent] completed - 生成报告 3551 字符（llm）
[Workflow] finished
```

### 16.7 测试方法

```powershell
# 0) 离线跑通（不消耗 API，最快验证整条链路）
python run_workflow.py --no-llm --quiet

# 1) 工作流用例（含需求要求的 4 个场景）
python -m unittest tests.test_workflow -v

# 2) 命令行用例（退出码、导出文件、降级提示）
python -m unittest tests.test_workflow_cli -v

# 3) 全量回归（Phase 1~4，共 164 个用例）
python -m unittest discover -s tests -t . -v
```

需求要求的 4 个场景与对应用例：

| 场景 | 用例 | 验证点 |
| --- | --- | --- |
| 测试1 正常数据 → 不触发诊断 | `test_01_normal_data_skips_diagnosis_flow`、`test_01b_...` | 节点轨迹只有 `data_analysis / abnormal_check / report`，无异常时不调用诊断与建议 |
| 测试2 异常数据 → 完整流程 | `test_02_abnormal_data_triggers_full_flow` | 5 个 Agent 全部执行，诊断/案例/建议均非空，LLM 调用 3 次 |
| 测试3 RAG 无案例 → 仍可完成 | `test_03_rag_without_cases_still_completes` | `retrieved_cases` 为空但状态为 `completed`，并把「未检索到足够相关的历史案例」传给建议 Agent |
| 测试4 DeepSeek 失败 → 明确错误 | `test_08_llm_failure_gives_clear_error`、`test_cli_llm_failure_exits_with_llm_error` | 错误记录含「认证失败」与排查建议、后续 LLM 阶段跳过、报告降级生成、CLI 退出码 3 |

另外还覆盖：阈值由程序决定（放宽阈值后不再触发诊断）、State 字段契约与序列化、
日志行格式、Agent 单一职责、数据文件缺失的明确报错、知识库不可用降级。

### 16.8 错误处理与排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `[配置错误] 未检测到环境变量 DEEPSEEK_API_KEY` | 未配置 Key | 复制 `.env.example` 为 `.env` 并填写；或先用 `--no-llm` 离线跑通 |
| `[Diagnosis Agent] failed - LLM 调用失败：认证失败（HTTP 401）` | Key 无效/过期 | 按提示检查 `DEEPSEEK_API_KEY`；工作流已降级出报告 |
| `[Recommendation Agent] skipped` | 前序大模型调用失败 | 属预期降级行为，修复 Key 后重跑 |
| `[RAG Agent] failed - 业务处理失败：知识库目录不存在` | 知识库缺失 | 执行 `python scripts\build_knowledge_base.py rebuild`，或用 `--no-rag` |
| `[Data Analysis Agent] failed - 业务处理失败：未找到数据文件` | 数据路径错误 | 检查 `--data` 参数 |
| 工作流状态 `completed_with_errors` | 某个阶段降级 | 查看 `output/workflow_steps.json` 的 `errors` 与报告末尾的「工作流异常与降级说明」 |

## 17. Phase 5：Streamlit 可视化平台

页面名称：**RobotOps AI 机器人运营智能分析平台**

### 17.1 界面职责与复用关系

前端只做四件事：**用户操作、文件上传、展示结果、调用已有工作流**。
指标计算、异常判定、RAG 检索、多 Agent 编排全部复用 Phase 1~4 的既有代码，
本阶段**没有新增任何 Agent**，也没有修改已有核心逻辑。

| 前端文件 | 职责 |
| --- | --- |
| `frontend/app.py` | Streamlit 页面：上传、预览、一键分析、结果分区展示、报告下载 |
| `frontend/ui_helpers.py` | 展示数据整理（字段信息 / 指标卡片 / 趋势 / 案例卡片 / 错误整理）、调用 Phase 4 工作流 |
| `.streamlit/config.toml` | 主题（浅色企业风）、上传大小上限、禁用使用统计上报 |

### 17.2 安装依赖

```powershell
cd "D:\Vibe Coding\RobotOps AI"
python -m pip install -r requirements.txt      # 含 streamlit>=1.57
# 如需虚拟环境：
# python -m venv .venv && .\.venv\Scripts\Activate.ps1 && python -m pip install -r requirements.txt
```

### 17.3 配置 .env

页面与命令行共用同一套配置，密钥仍然只放在 `.env`（**不要**写进代码）：

```powershell
copy .env.example .env
notepad .env        # 填写 DEEPSEEK_API_KEY=sk-你的密钥
```

- 未配置 Key 时页面会给出提示，可关闭左侧「启用 AI 分析（DeepSeek）」改用离线模板结论；
- 知识库（`knowledge/` 与 `data/chroma/`）与 Phase 3 完全一致，无需重复配置。

### 17.4 启动方式

```powershell
streamlit run frontend/app.py
# 默认地址：http://localhost:8501
```

常见参数（可选）：`--server.port 8502`、`--server.headless=true`。

### 17.5 页面功能

| 区域 | 内容 |
| --- | --- |
| 侧边栏「分析设置」 | 启用 AI 分析、参考历史案例（RAG）、检索案例数量、异常判定阈值（4 项）、API Key 状态 |
| 一、上传运营数据 | 支持 `.xlsx` / `.csv`；可一键「使用项目演示数据试跑」 |
| 数据预览 | 前 10 行数据、数据总量（行/列）、工作表与编码、字段信息（类型/非空/缺失/唯一值/是否必需） |
| 二、开始分析 | 「开始分析」按钮调用 Phase 4 多 Agent 工作流（分析 → 诊断 → RAG → 建议 → 报告） |
| 三、运营概览 | 项目数量、机器人数量、平均运行时长、故障率、平均满意度、总运营成本、平均节降率（含按日趋势 sparkline） |
| 四、异常项目 | 项目名称、异常指标、严重程度、推测原因与判断依据，以及按日的**变化趋势**折线图 |
| 五、AI 分析 | 5 个标签页：数据事实 / 异常发现 / 可能原因（推测）/ 历史相似案例 / 优化建议 |
| 六、RAG 历史参考案例 | 案例名称、案例类型、相关度、来源文件、相似原因、处理措施，并明确标注「历史参考案例，不代表当前项目实际情况」 |
| 七、最终运营报告 | Report Agent 输出以 Markdown 结构化展示，并提供**下载 Markdown 报告**按钮 |
| 工作流执行轨迹与日志 | 各 Agent 的执行状态、耗时与 `[Agent] started/completed` 日志 |

### 17.6 示例操作流程

```text
1) 启动：streamlit run frontend/app.py
2) 左侧确认「启用 AI 分析」为开（已配置 .env 中的 DEEPSEEK_API_KEY）
3) 点击「使用项目演示数据试跑」（或上传自己的 .xlsx / .csv）
4) 查看数据预览：前 10 行、数据总量、字段信息
5) 点击「开始分析」，等待 5 个 Agent 依次执行完成
6) 依次查看：运营概览 → 异常项目 → AI 分析 → RAG 历史参考案例 → 最终运营报告
7) 点击「下载 Markdown 报告」，得到 robotops_operation_report.md
```

想零消耗试跑时：把「启用 AI 分析」关掉（离线模板模式），其余步骤一致。

### 17.7 错误提示与处理

| 场景 | 页面表现 |
| --- | --- |
| 未上传文件 | 提示「请先上传运营数据文件（.xlsx / .csv）」，且「开始分析」按钮禁用 |
| 文件格式错误（如改名的损坏 Excel） | 顶部红框提示「文件读取失败：…」，并保留具体原因 |
| 上传了不支持的类型 | 文件选择器限制为 `.xlsx/.csv`；后端保存时再次校验并提示 |
| 数据字段缺失 | 红框列出缺失字段，并展开「必需字段清单」表格 |
| 未配置 API Key | 侧边栏与主区同时提示配置 `.env`，并说明可关闭 AI 分析离线运行 |
| Agent 执行失败（如 401 / 超时 / 知识库异常） | 顶部黄色提示列出失败的 Agent、原因与处理建议，报告以降级内容继续展示 |
| 分析过程未预期错误 | 顶部红框显示可读原因 + 「排查建议」折叠区，不展示 traceback |

### 17.8 前端测试

```powershell
# 前端纯逻辑（上传/预览/字段信息/指标卡片/趋势/案例卡片/错误整理/下载内容）
python -m unittest tests.test_frontend_helpers -v

# Streamlit 应用无头端到端（AppTest，离线模板模式，不消耗 API）
python -m unittest tests.test_frontend_app -v

# 全量回归（Phase 1~5，共 189 个用例）
python -m unittest discover -s tests -t . -v
```

AppTest 用例覆盖：页面正常渲染（标题/上传控件/提示）、未上传时按钮禁用、
关闭 AI 后使用演示数据跑通完整流程（预览 → 分析 → 五个结果区域 → 下载按钮），
并校验工作流 State 与程序计算的核心指标（`项目数量 = 6`）。

### 17.9 设计说明

- **简洁企业后台风格**：仅使用 Streamlit 原生组件 + `.streamlit/config.toml` 主题（浅色、低饱和、单一主色），
  未注入自定义 CSS，避免"过度设计"；
- **稳定优先**：上传解析结果缓存（`st.cache_data`）、分析结果保存在 `st.session_state`，
  避免重复解析与重复调用；指标卡片使用 `st.container(horizontal=True)` 自适应换行；
- **不重复实现业务逻辑**：前端所有数值直接来自工作流 State（由程序计算），
  AI 结论、历史案例与报告分别来自对应的 Agent，前端不做二次计算。

## 18. Phase 6：运营问题整改闭环

### 18.1 业务闭环（RobotOps AI 的核心业务流程）

```text
① 数据分析（Phase 1 Pandas：清洗 + 指标 + 异常识别）
        ↓
② 异常发现（程序按阈值判定，不用大模型判断）
        ↓
③ 问题诊断（Phase 4 Diagnosis Agent：可能原因，标注推测）
        ↓
④ 历史案例检索（Phase 3 RAG：Top-K 相似案例）
        ↓
⑤ 运营建议（Phase 4 Recommendation Agent：可执行措施 + 验证方式）
        ↓
⑥ 创建运营问题（ISSUE-YYYYMM-NNN，负责人 / 优先级 / 状态=待处理）
        ↓
⑦ 售后/运维处理（处理中 → 填写处理结果与备注 → 待验证）
        ↓
⑧ 重新上传数据（整改后数据 → 再次运行工作流）
        ↓
⑨ 整改效果验证（程序对比整改前后指标，自动计算改善幅度）
        ↓
⑩ 问题关闭（已完成 → 已关闭，统计关闭率与平均处理周期）
```

### 18.2 问题数据模型与状态机

问题编号：`ISSUE-YYYYMM-NNN`（按月自增，例如 `ISSUE-202609-001`）。

| 字段 | 说明 |
| --- | --- |
| 项目 / 问题 / 问题类型 | 项目名称、问题标题（如「机器人故障率异常升高」）、异常类型 |
| 异常指标 | 指标名、当前值（项目平均）、阈值、判定条件、命中记录、严重程度、**最差设备及其取值** |
| 当前数据 | 项目维度快照（机器人数量、运行时长、运行率、故障率、满意度、成本、节降率） |
| AI 诊断结果 | 诊断 Agent 输出的推测原因、置信度、判断依据、缺失数据 |
| RAG 历史案例 | 案例编号、标题、类型、相关度、来源文件（标注为模拟案例） |
| AI 建议 | 建议动作、优先级、对象、预期效果、验证方式、参考案例 |
| 负责人 / 状态 / 创建时间 / 更新时间 / 处理结果 | 任务跟踪信息 |
| 整改验证 | 整改前后指标、变化方向、改善幅度、结论（程序计算） |

状态流转（`robotops/issues/models.py::ALLOWED_TRANSITIONS`）：

```text
待处理 ──► 处理中 ──► 待验证 ──► 已完成 ──► 已关闭
  │            │  ▲        │           │        │
  │            │  └────────┘（验证不通过打回处理中）  │
  └────────────┴──► 已关闭（误报可直接关闭）        │
                              └───────────────────┘（已关闭可重新打开 → 处理中）
```

非法流转会被拒绝并给出提示（例如「不允许从『待验证』直接变更为『已关闭』」，并列出允许的下一状态）。

### 18.3 问题创建（可在界面一键完成）

- 入口一（推荐）：在「分析工作台」完成分析后，点击 **创建运营问题** → 选择项目、填写负责人 → 批量生成问题；
- 入口二（脚本/服务）：`IssueService.create_from_state(state, owner=..., projects=[...])`；
- 每个异常项目生成 1 个问题，优先级由严重程度自动判定（存在高危记录 → 高）；
- 创建时会把该项目的异常指标、AI 诊断、RAG 案例、AI 建议与数据来源一并写入问题详情。

### 18.4 问题处理

在「运营问题」页面可以：

1. 修改状态（只能选择合法的下一状态）；
2. 修改负责人；
3. 填写处理结果；
4. 添加备注（每次状态变更、处理结果、整改验证都会自动生成处理记录）。

### 18.5 整改效果验证（关键：全部由程序计算）

当问题状态为 **待验证** 时，页面允许重新上传整改后的运营数据，系统会：

1. 对整改后数据重新运行分析（离线模板模式，只用程序指标，不消耗 API）；
2. 按项目对比整改前后的同一指标；
3. 计算改善幅度并生成结论。

```text
改善幅度 = |整改前 - 整改后| ÷ |整改前|
```

| 指标 | 改善方向 | 示例 |
| --- | --- | --- |
| 故障率 | 越低越好 | 8.2% → 3.7%：下降约 **54.9%**，结论「当前数据表现出明显改善」 |
| 满意度 | 越高越好 | 81 分 → 92 分：上升约 13.6% |
| 运行率 | 越高越好 | 78% → 93%：上升约 19.2% |
| 节降率 | 越高越好 | 8% → 15%：上升约 87.5% |

结论规则：改善幅度 ≥ 10% → 「明显改善」；0 < 改善幅度 < 10% → 「有轻微改善，建议继续观察」；
未见改善 → 「未体现改善，建议继续处理并再次验证」。验证通过时问题自动进入 **已完成**，
并可继续关闭。验证结果与说明会写入问题记录，且明确标注
「以上结果由程序按整改前后的实际数据计算，未使用大模型推断」。

### 18.6 问题统计

页面顶部展示：**总问题数、待处理、处理中、已完成、问题关闭率、平均处理周期（天）、未关闭问题数**。

- 问题关闭率 = 已关闭问题数 ÷ 总问题数；
- 平均处理周期 = 已关闭问题的（关闭时间 − 创建时间）平均值；无已关闭问题时显示「—」。

### 18.7 SQLite 存储与数据安全

| 项目 | 说明 |
| --- | --- |
| 存储方式 | Python 标准库 `sqlite3`，单文件 `data/issues.db`（WAL 模式） |
| 表结构 | `issues`（问题主表）+ `issue_notes`（处理记录）+ `meta`（结构版本） |
| 自定义位置 | 环境变量 `ROBOTOPS_ISSUES_DB`（便于测试与多环境部署） |
| 是否入库 | **不保存任何 API Key**：密钥只放在 `.env`，问题表只存业务字段（有专门的单元测试校验库文件中不含 `sk-`） |
| Git | `.env` 与 `data/issues.db*` 均已加入 `.gitignore`，不会被提交 |

### 18.8 前端「运营问题」页面

入口：`streamlit run frontend/app.py` → 侧边栏切换到 **运营问题**（「分析工作台」为默认模块）。

页面内容：问题统计 → 问题列表（问题编号 / 项目 / 问题类型 / 优先级 / 负责人 / 状态 / 创建时间，
支持按状态与项目筛选）→ 选中问题查看**完整详情** → 问题处理表单 → 整改效果验证（待验证状态）→
关闭问题。切换模块采用「单入口 + 侧边栏切换 + 直接执行视图脚本」的实现，
避免多页路由在无头测试环境中重复执行页面导致的控件重复问题（详见本文件第 17 章的界面说明）。

### 18.9 测试与运行

```powershell
# Phase 6 核心逻辑（创建/流转/处理/验证/关闭/统计/持久化/数据安全）
python -m unittest tests.test_issues -v

# Phase 6 页面（问题列表/详情/状态修改/验证入口/创建入口）
python -m unittest tests.test_issues_page -v

# 全量回归（Phase 1~6，共 221 个用例）
python -m unittest discover -s tests -t . -v
```

需求要求的 6 个场景与用例对应关系：

| 场景 | 用例 |
| --- | --- |
| 1 创建问题 | `test_create_issue_from_state`、`test_create_issue_entry_writes_issue` |
| 2 修改状态 | `test_status_transitions`、`test_illegal_transition_is_rejected`、`test_close_and_reopen` |
| 3 添加处理结果 | `test_add_note_and_resolution` |
| 4 整改后重新分析 | `test_verify_improvement_flow`、`test_end_to_end_closed_loop_with_real_data` |
| 5 自动计算改善幅度 | `test_compute_metric_change_matches_requirement_example`（8.2% → 3.7% = 54.9%） |
| 6 完成问题关闭 | `test_close_issue_and_stats`、`test_stats_with_mixed_statuses` |

### 18.10 端到端实测示例（真实演示数据）

```text
问题编号：ISSUE-202609-001   项目：华北-天津港智能巡检项目
问题：机器人故障率异常升高    优先级：高    负责人：售后运维    状态：待处理
异常指标：故障率 2.91%（> 5%，最差设备 TJ-INS-05 10.96%）
AI 诊断：1 条推测原因；RAG 案例：CASE-FAULT-001；AI 建议：3 条
处理：处理中（已派单）→ 填写处理结果 → 待验证
整改后重新分析 → 验证：故障率 2.91% → 0%，下降约 100%，明显改善 → 自动「已完成」
关闭：已关闭（关闭时间 2026-09-24 02:04，处理记录 6 条）
统计：总问题 1 / 已关闭 1 / 问题关闭率 100% / 平均处理周期 0.0 天
```

## 19. 后续阶段（本文档不实现）

按需求，以下内容留待后续阶段：Docker 部署、外部数据库（PostgreSQL 等）、定时调度与告警、
移动端或多租户支持。
Phase 1 的指标与异常结果已固化为 JSON / CSV / Excel，Phase 2 的输入载荷固化为
`output/ai_input_payload.json`，Phase 3 的检索明细固化为 `output/ai_rag_retrieval.json`，
Phase 4 的工作流 State 与轨迹固化为 `output/workflow_state.json` / `workflow_steps.json`，
Phase 6 的运营问题固化为 `data/issues.db`（SQLite），后续阶段可直接消费，无需重新设计数据口径。

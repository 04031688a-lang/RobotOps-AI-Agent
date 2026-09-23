# RobotOps AI · 机器人运营数据分析智能平台

> 一个端到端的机器人运营数据分析项目：从 Excel / CSV 数据清洗、指标计算与异常识别，
> 到 DeepSeek 大模型诊断、RAG 历史案例检索、LangGraph 多 Agent 工作流，
> 最后落到「运营问题整改闭环」与可视化操作界面。

**核心设计原则**

1. **数值只由程序计算**：所有指标、异常判定、整改前后改善幅度均由 Pandas / Python 计算，大模型只做解读，不允许改数。
2. **原始数据不直接交给大模型**：先转成结构化结果（指标 + 异常 + 历史案例），再发送给 DeepSeek，既省 token 也便于审计。
3. **每一步都可追溯**：清洗问题清单、异常明细、RAG 检索记录、工作流轨迹、问题处理记录全部落盘。

> 详细的分阶段开发手册见仓库内的 `shouce.md`（可选阅读，按开发阶段记录每一步的取舍与验证方式）。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心业务流程](#2-核心业务流程)
- [3. 功能特性](#3-功能特性)
- [4. 项目截图](#4-项目截图)
- [5. 目录结构](#5-目录结构)
- [6. 环境要求](#6-环境要求)
- [7. 从零安装](#7-从零安装)
- [8. 配置 .env](#8-配置-env)
- [9. 操作手册](#9-操作手册)
- [10. 数据模型与字段说明](#10-数据模型与字段说明)
- [11. 数据清洗规则](#11-数据清洗规则)
- [12. 指标口径](#12-指标口径)
- [13. 异常识别规则](#13-异常识别规则)
- [14. 配置项清单](#14-配置项清单)
- [15. 输出文件说明](#15-输出文件说明)
- [16. 测试与验证](#16-测试与验证)
- [17. 安全与隐私](#17-安全与隐私)
- [18. 常见问题](#18-常见问题)
- [19. 后续规划](#19-后续规划)
- [20. 许可与免责声明](#20-许可与免责声明)

---

## 1. 项目简介

RobotOps AI 面向**机器人运营团队**（清洁 / 巡检 / 安防巡逻 / 配送 / 消杀 / AGV 搬运等场景），
把散落在 Excel 报表里的运营数据变成可执行的运营决策：

```text
上传运营数据 → Pandas 清洗与指标计算 → 程序判定异常
   → DeepSeek 分析原因 → RAG 检索历史相似案例 → 生成优化建议
   → 创建运营问题 → 售后/运维处理 → 重新上传整改后数据
   → 程序对比整改效果 → 关闭问题
```

适合作为：

- 机器人运营 / 售后服务团队的数据分析工具；
- 学习「数据工程 + RAG + LangGraph 多 Agent + Streamlit」完整链路的参考项目；
- 二次开发的底座（指标口径、异常规则、知识库、Agent 提示词都集中在配置与独立模块中）。

> 仓库内附带的演示数据与知识库案例均为**模拟数据**，不含任何真实企业、真实设备或个人信息。

## 2. 核心业务流程

```text
① 数据分析（Pandas：清洗 → 指标 → 异常识别）
        ↓
② 异常发现（程序按固定阈值判定，大模型不参与）
        ↓
③ 问题诊断（Diagnosis Agent：可能原因，标注「推测」与缺失数据）
        ↓
④ 历史案例检索（RAG：从知识库检索 Top-K 相似案例）
        ↓
⑤ 运营建议（Recommendation Agent：可执行措施 + 验证方式 + 参考案例）
        ↓
⑥ 创建运营问题（ISSUE-YYYYMM-001，负责人 / 优先级 / 状态=待处理）
        ↓
⑦ 售后 / 运维处理（处理中 → 填写处理结果与备注 → 待验证）
        ↓
⑧ 重新上传整改后数据（再次运行分析）
        ↓
⑨ 整改效果验证（程序对比整改前后指标，自动计算改善幅度）
        ↓
⑩ 问题关闭（已完成 → 已关闭，统计关闭率与平均处理周期）
```

## 3. 功能特性

### 3.1 六个阶段的完整能力

| 阶段 | 能力 | 关键实现 |
| --- | --- | --- |
| Phase 1 数据分析 | Excel / CSV 读取、数据清洗、7 项核心指标、4 类异常识别、报告导出 | `robotops/data_loader.py`、`data_cleaner.py`、`metrics.py`、`anomaly.py` |
| Phase 2 AI 分析 | 把结构化分析结果交给 DeepSeek，输出「概览 / 关键发现 / 异常发现 / 可能原因 / 优化建议」 | `robotops/llm/`、`robotops/agent/` |
| Phase 3 RAG 知识库 | 17 个运营案例建成本地向量库（ChromaDB），按异常描述检索 Top-K 相似案例 | `robotops/rag/`、`knowledge/` |
| Phase 4 多 Agent 工作流 | LangGraph 编排 5 个单一职责 Agent，按「有无异常」自动分支 | `app/agents/`、`app/graph/` |
| Phase 5 可视化界面 | Streamlit 平台：上传、预览、一键分析、结果展示、报告下载 | `frontend/` |
| Phase 6 整改闭环 | 运营问题管理、状态流转、处理记录、整改效果验证、问题统计 | `robotops/issues/` |

### 3.2 技术栈

| 类别 | 选型 |
| --- | --- |
| 语言 / 运行时 | Python 3.10+（开发验证环境：Python 3.12，Windows 11 + VS Code） |
| 数据处理 | pandas、numpy、openpyxl |
| 大模型 | DeepSeek Chat Completions（OpenAI 兼容接口），HTTP 使用标准库 `urllib` 实现，超时 / 重试 / 错误映射自带 |
| 向量库 | ChromaDB（本地持久化） |
| Embedding | 本地哈希向量（无需下载模型，详见 9.7 节） |
| 工作流编排 | LangGraph |
| 界面 | Streamlit |
| 存储 | SQLite（标准库 `sqlite3`，仅用于运营问题） |
| 测试 | 标准库 `unittest`（221 个用例，无需额外依赖） |

## 4. 项目截图

截图统一放在 `docs/images/`（当前为占位目录）。按下面命名放入自己的截图后，取消本节末尾的注释即可显示。

| 建议文件名 | 建议截图内容 |
| --- | --- |
| `analysis-overview.png` | 分析工作台：运营概览 7 项指标 + 按日趋势 |
| `analysis-ai.png` | AI 分析标签页：可能原因 / 优化建议 |
| `rag-cases.png` | RAG 历史参考案例（含「历史参考案例，不代表当前项目实际情况」标注） |
| `issues-board.png` | 运营问题：统计卡片与问题列表 |
| `issue-verification.png` | 整改效果验证：整改前后对比与改善幅度 |

```markdown
<!-- 图片放好后取消注释即可显示
![运营概览](docs/images/analysis-overview.png)
![运营问题](docs/images/issues-board.png)
-->
```

## 5. 目录结构

```text
RobotOps-AI/
├── run_analysis.py            # Phase 1 入口：数据分析 + 异常识别 + 报告
├── main.py                    # Phase 2/3 入口：数据分析 + DeepSeek（可含 RAG）
├── run_workflow.py            # Phase 4 入口：LangGraph 多 Agent 工作流
├── requirements.txt
├── README.md                  # 本文档（公开项目说明）
├── shouce.md                  # 分阶段开发手册（可选阅读）
├── LICENSE                    # MIT License
├── .env.example               # 环境变量模板（复制为 .env 后填写）
├── .gitignore
├── robotops/                  # 核心业务包
│   ├── config.py              # 路径 / 列名 / 校验边界 / 异常阈值（唯一配置源）
│   ├── models.py              # 数据模型：字段字典、记录模型、指标模型
│   ├── data_loader.py         # 读取 Excel / CSV，表头规范化与别名映射
│   ├── data_cleaner.py        # 清洗 10 步流程 + 问题清单
│   ├── metrics.py             # 7 项核心指标 + 项目维度汇总
│   ├── anomaly.py             # 4 类异常规则 + 严重程度分级
│   ├── report.py              # 报告与结果文件导出（CSV / Excel / JSON / Markdown）
│   ├── pipeline.py            # 流程编排：读取 → 清洗 → 指标 → 异常 → 输出
│   ├── llm/                   # Phase 2：DeepSeek 客户端、配置、提示词、结构化输出、日志
│   ├── agent/                 # Phase 2：单 Agent（载荷构造 → DeepSeek → 结构化结果）
│   ├── rag/                   # Phase 3：文档加载、切分、Embedding、向量库、检索
│   └── issues/                # Phase 6：运营问题模型、SQLite 存储、闭环服务
├── app/                       # Phase 4：LangGraph 工作流层
│   ├── agents/                # 5 个 Agent：分析 / 诊断 / RAG / 建议 / 报告
│   └── graph/                 # State、图编排、日志、提示词
├── frontend/                  # Phase 5/6：Streamlit 界面
│   ├── app.py                 # 入口（侧边栏切换模块）
│   ├── views/analysis.py      # 分析工作台
│   ├── views/issues.py        # 运营问题
│   ├── ui_helpers.py          # 展示数据整理 + 调用工作流
│   └── issues_helpers.py      # 运营问题展示辅助
├── scripts/                   # 辅助脚本
│   ├── generate_demo_data.py  # 生成演示数据
│   ├── build_knowledge_base.py# 知识库管理（重建 / 增量 / 检索 / 自检）
│   ├── phase2_smoke_test.py   # Phase 2 离线自检（不消耗 API）
│   └── check_deepseek_api.py  # DeepSeek 连通性检查（真实调用，消耗极少 token）
├── knowledge/                 # RAG 知识库（17 个模拟案例，4 个分类）
├── docs/images/               # 项目截图目录
├── data/
│   ├── raw/                   # 原始数据（含演示数据 robot_operation_demo.xlsx）
│   ├── chroma/                # ChromaDB 本地向量库（自动生成，已忽略）
│   └── issues.db              # 运营问题 SQLite（自动生成，已忽略）
├── output/                    # 运行产物（报告 / 日志 / 检索记录，已忽略）
├── .streamlit/config.toml     # 界面主题与上传大小配置
└── tests/                     # 221 个自动化测试
```

## 6. 环境要求

- 操作系统：Windows 10/11、macOS、Linux 均可（开发与验证环境为 Windows 11）
- Python：3.10 及以上（推荐 3.12）
- 内存：建议 ≥ 4 GB；磁盘：安装依赖约需 400 MB～1 GB
- 网络：安装依赖与调用 DeepSeek 时需要联网；**除 DeepSeek 对话接口外，程序不会向外发送任何数据**

## 7. 从零安装

```bash
# 1) 获取代码
git clone https://github.com/<your-name>/RobotOps-AI.git
cd RobotOps-AI

# 2) 创建并激活虚拟环境
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3) 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) 配置环境变量（见下一节）
# Windows
copy .env.example .env
# macOS / Linux
# cp .env.example .env
# 然后用编辑器打开 .env，填入你的 DeepSeek API Key

# 5) 自检：确认安装成功（不需要 API Key）
python scripts/phase2_smoke_test.py         # 预期：自检结果 11/11 项通过
python scripts/build_knowledge_base.py demo # 预期：自检结果 10/10 项通过
```

`requirements.txt` 内容：

```text
pandas>=2.2.0
numpy>=1.26.0
openpyxl>=3.1.0
chromadb>=1.0.0
langgraph>=1.0.0
streamlit>=1.57.0
```

> 说明：项目**不需要** `python-dotenv`（未安装时使用内置 `.env` 解析器，安装了会自动优先使用），
> 也没有使用 `requests` / `openai` SDK——HTTP 调用由标准库实现，依赖越少越好装。

## 8. 配置 .env

把 `.env.example` 复制为 `.env`，按需修改（**只有 API Key 是必填项**）：

```ini
# ===== DeepSeek 大模型（Phase 2/4）=====
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx   # 必填，在 DeepSeek 控制台创建
# DEEPSEEK_MODEL=deepseek-chat                 # 默认 deepseek-chat
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# DEEPSEEK_TIMEOUT=60                          # 单次请求超时（秒）
# DEEPSEEK_MAX_RETRIES=2                       # 限流 / 超时 / 服务端错误自动退避重试
# DEEPSEEK_TEMPERATURE=0.2
# DEEPSEEK_MAX_TOKENS=4096
# DEEPSEEK_LOG_LEVEL=INFO                      # 日志写入 output/deepseek_llm.log

# ===== RAG 知识库（Phase 3）=====
# ROBOTOPS_KNOWLEDGE_DIR=knowledge             # 知识库目录
# ROBOTOPS_CHROMA_DIR=data/chroma              # 向量库持久化目录
# ROBOTOPS_RAG_TOP_K=5                         # 每次检索召回块数
# ROBOTOPS_RAG_MAX_CASES=6                     # 最多注入提示词的案例数
# ROBOTOPS_RAG_MIN_SIMILARITY=0.35             # 相似度阈值（低于该值视为不相关）
# ROBOTOPS_EMBEDDING_DIM=1024                  # 本地向量维度（改动后需 rebuild）

# ===== 运营问题（Phase 6）=====
# ROBOTOPS_ISSUES_DB=data/issues.db            # 问题库位置（SQLite）
```

**安全约定（务必遵守）**

- `.env` 已在 `.gitignore` 中，**永远不要提交**；仓库里只保留不含密钥的 `.env.example`。
- 代码中不存在任何硬编码密钥；日志中的密钥会自动脱敏（形如 `sk-abcd****wxyz`）。
- 运营问题库只保存业务字段（项目、指标、处理记录），**不保存任何密钥**。
- 提交前可自检：

```bash
git status --short | grep "\.env"          # 期望：无输出
git check-ignore -v .env                   # 期望：显示 .gitignore 中的匹配规则
```

## 9. 操作手册

> 命令均在项目根目录执行；Windows 用反斜杠（`scripts\xxx.py`），macOS / Linux 用正斜杠（`scripts/xxx.py`）。

### 9.1 第一步：准备数据

两种方式二选一：

**A. 使用演示数据（推荐先跑通）**

```bash
python scripts/generate_demo_data.py
```

生成 `data/raw/robot_operation_demo.xlsx`（4 个工作表：运营数据、字段说明、项目与机器人、数据说明）：

- 6 个项目、23 台机器人、30 天 = 690 条正常记录，另加 **7 条故意注入的脏数据**（重复上报、满意度越界、负数成本、字段缺失等）；
- 用于验证整个系统：清洗后应为 **695 条**（剔除 2 条重复）。

**B. 使用你自己的数据**

把 Excel / CSV 放到任意位置，运行时用 `--data` 指定路径即可。字段要求见第 10 节，
列名支持常见别名（如 `项目`、`机器人编号`、`满意度`、`成本`），缺失必需列时会给出明确的缺列提示。

### 9.2 第二步：Phase 1 —— 数据分析（无需 API Key）

```bash
python run_analysis.py                                    # 使用演示数据
python run_analysis.py --data data/raw/your_file.xlsx     # 使用自己的数据
python run_analysis.py --help                             # 查看全部参数
```

运行后会输出 5 段报告，并在 `output/` 生成结果文件：

1. 数据清洗结果（原始 / 剔除 / 清洗后记录数、问题条目、日期范围、缺失统计）
2. 核心指标（项目数量、机器人数量、平均运行时长、故障率、平均满意度、总运营成本、平均节降率）
3. 项目维度汇总
4. 异常识别（阈值、命中条数、按类型汇总、异常明细）
5. 已导出文件清单

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-d, --data` | 数据文件路径（`.xlsx` / `.xlsm` / `.csv`） |
| `-o, --output` | 结果输出目录（默认 `output/`） |
| `--sheet` | 指定 Excel 工作表名 |
| `--encoding` | 指定 CSV 编码（如 `gbk`），默认自动识别 |
| `--no-export` | 只打印报告，不写文件 |
| `--satisfaction-lower` 等 | 覆盖异常阈值（故障率 / 满意度 / 运行率 / 节降率） |
| `--fault-rate-scope` | 故障率判定口径：`robot_period`（默认，按机器人累计）/ `record`（逐日判定） |

（`python -m robotops` 与 `python run_analysis.py` 等价。）

### 9.3 第三步：Phase 2/3 —— AI 分析（DeepSeek，可选 RAG）

```bash
python main.py                            # 完整流程：Pandas 分析 → DeepSeek 解读（含 RAG 案例）
python main.py --dry-run                  # 只打印将发送给模型的结构化载荷，不调用 API（零消耗）
python main.py --dry-run --show-prompt    # 查看完整 Prompt
python main.py --no-rag                   # 关闭历史案例检索
python main.py --json                     # 以 JSON 输出 AI 分析结果
```

数据流：`Excel → Pandas 结构化结果 → 结构化载荷 → DeepSeek → 结构化结论（JSON）`，
原始明细**不会**发送给大模型。

输出：`output/ai_input_payload.json`（发送内容备份）、`ai_analysis.json` / `ai_analysis.md`（AI 结论）、
`ai_rag_retrieval.json`（RAG 检索明细）。

> 想验证 API 是否接通：`python scripts/check_deepseek_api.py`（真实调用一次，约消耗 65 token）。

### 9.4 第四步：Phase 4 —— LangGraph 多 Agent 工作流

```bash
python run_workflow.py                 # 完整工作流（需要 API Key）
python run_workflow.py --no-llm        # 离线模板模式：不调用大模型、零消耗
python run_workflow.py --no-rag        # 不检索历史案例
python run_workflow.py --json          # 以 JSON 输出完整 State
```

工作流节点与分支：

```text
START → data_analysis → abnormal_check
                            ├─ 无异常 → report → END
                            └─ 有异常 → diagnosis → rag → recommendation → report → END
```

控制台会打印每个 Agent 的执行日志（同时写入 `output/workflow.log`）：

```text
[Workflow] started - 数据源=robot_operation_demo.xlsx 知识库=启用 大模型=启用
[Data Analysis Agent] started
[Data Analysis Agent] completed - 项目 6 个 / 机器人 23 台 / 异常 77 条
[Abnormal Check] 程序判定命中异常 77 条（阈值：…），进入诊断流程
[Diagnosis Agent] completed - 输出 4 条推测原因
[RAG Agent] retrieved 6 cases
[Recommendation Agent] completed - 输出 4 条优化建议
[Report Agent] completed - 生成报告 3551 字符
[Workflow] finished
```

退出码：`0` 成功；`2` 输入 / 配置问题（数据、知识库、缺少 API Key）；`3` 大模型调用失败（报告会自动降级生成）。

输出：`output/workflow_report.md`（报告）、`workflow_state.json`（完整状态）、`workflow_steps.json`（节点轨迹与错误）。

### 9.5 第五步：Phase 5 —— 启动可视化界面

```bash
streamlit run frontend/app.py
# 浏览器打开 http://localhost:8501
```

界面为「单入口 + 侧边栏模块切换」，两个模块：

| 模块 | 作用 |
| --- | --- |
| 分析工作台 | 上传数据 → 预览 → 一键分析 → 查看指标 / 异常 / AI 分析 / RAG 案例 / 报告 → 创建运营问题 |
| 运营问题 | 问题统计与列表 → 查看详情 → 处理 → 整改效果验证 → 关闭问题 |

侧边栏「分析设置」可以：开关 AI 分析（关闭则用离线模板结论、零消耗）、开关历史案例检索、
设置检索案例数量、调整 4 项异常阈值。

### 9.6 第六步：完整整改闭环操作（界面分步）

1. **上传数据**：在「分析工作台」上传 `.xlsx` / `.csv`（或点「使用项目演示数据试跑」）；
2. **数据预览**：确认前 10 行、数据总量、字段信息（类型 / 非空 / 缺失 / 唯一值 / 是否必需）；
3. **开始分析**：点击「开始分析」，等待 5 个 Agent 执行完成；
4. **查看结果**：运营概览（7 项指标 + 按日趋势）→ 异常项目（含推测原因与趋势图）→
   AI 分析（数据事实 / 异常发现 / 可能原因 / 历史相似案例 / 优化建议）→ RAG 案例 → 最终报告；
5. **创建运营问题**：在「八、运营问题」区域选择项目、填写负责人，点击「创建运营问题」，
   生成 `ISSUE-YYYYMM-NNN`；
6. **处理问题**：切到「运营问题」模块 → 选中问题 → 在「四、问题处理」中修改状态（待处理 → 处理中）、
   填写处理结果与备注；
7. **进入待验证**：把状态改为「待验证」（此时页面会出现整改验证区）；
8. **上传整改后数据**：上传整改后的运营数据，点击「验证整改效果」；
9. **查看验证结论**：系统按实际数据计算改善幅度，例如「指标较整改前下降约 54.9%，当前数据表现出明显改善」，
   验证通过会自动把问题标记为「已完成」；
10. **关闭问题**：在「六、关闭问题」中点击关闭，状态变为「已关闭」，统计中同步更新关闭率与处理周期。

### 9.7 知识库维护（RAG）

知识库位于 `knowledge/`，当前含 17 个**模拟**案例（4 个分类：`equipment_fault`、`maintenance`、
`satisfaction`、`operation`），覆盖 12 类典型运营问题（重复故障、运行率下降、巡检不到位、
满意度下降、清洁效果下降、维修频繁、售后响应慢、成本过高、节降率下降、设备闲置、调度异常、维护周期过长）。

每个案例 = 一个 Markdown 文件，由「元数据 + 5 个必填小节」组成：

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

管理命令：

```bash
python scripts/build_knowledge_base.py rebuild   # 重建索引（首次或改了切分 / Embedding 参数后）
python scripts/build_knowledge_base.py sync      # 增量同步（新增案例）
python scripts/build_knowledge_base.py status    # 索引状态
python scripts/build_knowledge_base.py list      # 案例清单
python scripts/build_knowledge_base.py search "B小区机器人故障率明显升高，同时维修次数增加。"
python scripts/build_knowledge_base.py queries   # 用演示数据的异常描述自动生成查询并检索
python scripts/build_knowledge_base.py demo      # 离线自检：相关问题能命中 / 无关问题不误命中
```

**关于 Embedding 的说明**：DeepSeek 开放平台只提供对话补全接口、**没有 embedding 接口**，
因此本项目采用本地哈希向量（字符 2-3 gram + 英文词 + 领域词加权，1024 维，余弦空间）：
无需下载模型、离线可用、结果可复现；检索时还会做「领域词门控 + 相似度阈值过滤」，
确保无关问题不会硬匹配到案例。更换 Embedding 方案后需执行一次 `rebuild`。

### 9.8 三个典型使用场景（速查）

```bash
# 场景 1：只想看指标和异常，不花钱
python run_analysis.py

# 场景 2：想让大模型给出诊断和建议
python main.py --quiet

# 场景 3：要走完整闭环（含整改验证）
streamlit run frontend/app.py
```

## 10. 数据模型与字段说明

每行 = 一台机器人 + 一天。字段要求：

| 字段 | 类型 | 单位 | 是否必需 | 说明 |
| --- | --- | --- | --- | --- |
| 日期 | 日期 | — | 必需 | 每台机器人每天一条记录 |
| 项目名称 | 文本 | — | 必需 | 「项目数量」按其去重统计 |
| 机器人ID | 文本 | — | 必需 | 「机器人数量」按其去重统计 |
| 机器人类型 | 文本 | — | 必需 | 清洁 / 巡检 / 安防巡逻 / 配送 / 消杀 / AGV搬运 |
| 运行时长 | 数值 | 小时 | 必需 | 当日实际运行时长 |
| 计划运行时长 | 数值 | 小时 | 可选 | 用于计算运行率，缺失时按机器人类型默认值兜底 |
| 故障次数 | 整数 | 次 | 必需 | 当日故障次数 |
| 巡检次数 | 整数 | 次 | 必需 | 当日巡检 / 作业次数，作为故障率分母 |
| 维修次数 | 整数 | 次 | 必需 | 当日维修次数 |
| 用户满意度 | 数值 | 分 | 必需 | 有效区间 0-100 |
| 运营成本 | 数值 | 元 | 必需 | 当日运营成本 |
| 节降率 | 数值 | % | 必需 | 成本节降比例 |

派生字段（程序计算，原始数据无需提供）：`运行率 = 运行时长 ÷ 计划运行时长 × 100%`、
`故障率 = 故障次数 ÷ 巡检次数 × 100%`。

## 11. 数据清洗规则

清洗流程固定、可复现，每一步的问题都会写入 `output/robot_ops_report_cleaning_issues.csv`：

1. **表头规范**：去空格 / 全角空格 / BOM，按别名映射标准列名，校验必需列；
2. **结构清理**：删除全空行；仅删除非模型字段的全空列（标准字段整列为空时给出错误提示）；
3. **文本清洗**：文本字段去首尾与重复空白；项目名称、机器人ID 为空的记录剔除；
4. **日期解析**：支持日期文本与 Excel 序列号，无法解析的记录剔除；
5. **类型转换**：支持带千分位、百分号、全角符号的文本；
6. **业务边界**：负运行时长 / 负计数 / 负成本置空；满意度超出 0-100 置空；节降率超出 -100%~100% 置空；
   单日运行时长超过 24 小时仅告警不删除；
7. **缺失值处理**：故障 / 巡检 / 维修次数缺失按 0 计；满意度、运行时长、成本、节降率缺失则保留记录并跳过计算；
8. **去重**：先删完全重复行，再按「日期 + 机器人ID」去重（保留最后一条上报）；
9. **派生列**：计算运行率与故障率（分母为 0 时留空）；
10. **排序**：按日期 + 机器人ID 升序输出。

## 12. 指标口径

| 指标 | 单位 | 口径 |
| --- | --- | --- |
| 项目数量 | 个 | 对「项目名称」去重计数 |
| 机器人数量 | 台 | 对「机器人ID」去重计数 |
| 平均运行时长 | 小时 | 清洗后记录「运行时长」的平均值（跳过空值） |
| 故障率 | % | 总故障次数 ÷ 总巡检次数 × 100% |
| 平均满意度 | 分 | 有效「用户满意度」记录的平均值 |
| 总运营成本 | 元 | 有效「运营成本」记录求和 |
| 平均节降率 | % | 有效「节降率」记录的平均值 |

补充统计（非验收指标）：平均运行率、记录数、总故障次数。项目维度汇总复用同一套口径。

## 13. 异常识别规则

| 异常类型 | 判定条件 | 判定粒度 |
| --- | --- | --- |
| 故障率偏高 | 故障率 > 5% | 机器人 + 分析周期累计 |
| 满意度偏低 | 用户满意度 < 85 分 | 逐日记录 |
| 运行率偏低 | 运行率 < 80% | 逐日记录 |
| 节降率偏低 | 节降率 < 10% | 逐日记录 |

- **严重程度**：偏差幅度 = |实际值 − 阈值| ÷ |阈值|，≥50% 为「高」，≥20% 为「中」，其余为「低」。
- **为什么故障率按周期累计**：故障率是比率型指标，逐日判定的分母太小（1 次故障 ÷ 8 次巡检 = 12.5%），
  会让 5% 阈值失去区分度；因此默认按「机器人 + 分析周期」累计判定，其余三条规则按每日记录判定。
  也可用 `--fault-rate-scope record` 切回逐日判定。
- **异常判定完全由程序完成**：阈值集中在 `robotops/config.py`，大模型不参与「是否异常」的判断，也不能修改阈值。

## 14. 配置项清单

所有可配置项都可以写在 `.env` 中（默认值集中在代码内）：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无（必填） | DeepSeek 密钥，仅从环境变量读取 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 接口地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `DEEPSEEK_TIMEOUT` | `60` | 单次请求超时（秒） |
| `DEEPSEEK_MAX_RETRIES` | `2` | 失败自动重试次数（指数退避） |
| `DEEPSEEK_TEMPERATURE` | `0.2` | 采样温度 |
| `DEEPSEEK_MAX_TOKENS` | `4096` | 单次最大输出长度 |
| `DEEPSEEK_LOG_LEVEL` | `INFO` | 日志级别 |
| `ROBOTOPS_ENV_FILE` | 项目根 `.env` | 自定义 .env 位置 |
| `ROBOTOPS_KNOWLEDGE_DIR` | `knowledge` | 知识库目录 |
| `ROBOTOPS_CHROMA_DIR` | `data/chroma` | 向量库目录 |
| `ROBOTOPS_RAG_TOP_K` | `5` | 检索召回块数 |
| `ROBOTOPS_RAG_MAX_CASES` | `6` | 注入提示词的最大案例数 |
| `ROBOTOPS_RAG_MIN_SIMILARITY` | `0.35` | 相似度阈值 |
| `ROBOTOPS_EMBEDDING_BACKEND` | `local_hashing` | Embedding 方案 |
| `ROBOTOPS_EMBEDDING_DIM` | `1024` | 向量维度 |
| `ROBOTOPS_RAG_AUTO_BUILD` | `true` | 索引缺失时自动构建 |
| `ROBOTOPS_ISSUES_DB` | `data/issues.db` | 运营问题 SQLite 路径 |

## 15. 输出文件说明

| 文件 | 来源 | 内容 |
| --- | --- | --- |
| `output/robot_ops_report_metrics.csv` | Phase 1 | 核心指标表 |
| `output/robot_ops_report_project_summary.csv` | Phase 1 | 项目维度汇总 |
| `output/robot_ops_report_anomalies.csv` | Phase 1 | 异常明细 |
| `output/robot_ops_report_cleaning_issues.csv` | Phase 1 | 数据清洗问题清单 |
| `output/robot_ops_report_metrics.json` | Phase 1 | 指标 + 清洗统计 + 阈值 |
| `output/robot_ops_report.xlsx` | Phase 1 | Excel 汇总（9 个工作表） |
| `output/robot_ops_report.md` | Phase 1 | Markdown 报告 |
| `output/ai_input_payload.json` | Phase 2 | 实际发送给大模型的结构化载荷 |
| `output/ai_analysis.json` / `.md` | Phase 2/3 | AI 结构化结论（含历史案例章节） |
| `output/ai_rag_retrieval.json` | Phase 3 | RAG 检索明细（查询、命中案例、相似度） |
| `output/deepseek_llm.log` | Phase 2 | LLM 调用日志（密钥脱敏） |
| `output/workflow_report.md` | Phase 4 | 多 Agent 工作流最终报告 |
| `output/workflow_state.json` | Phase 4 | 完整工作流 State |
| `output/workflow_steps.json` | Phase 4 | 节点轨迹与错误 |
| `output/workflow.log` | Phase 4 | Agent 执行日志 |
| `data/chroma/` | Phase 3 | 本地向量库（可删除重建） |
| `data/issues.db` | Phase 6 | 运营问题 SQLite（不含任何密钥） |

`output/*`、`data/chroma/`、`data/issues.db*`、`.env` 均在 `.gitignore` 中，不会进入版本库。

## 16. 测试与验证

```bash
# 全量回归（221 个用例，无需 API Key，不产生费用）
python -m unittest discover -s tests -t . -v

# 按阶段运行
python -m unittest tests.test_data_loader tests.test_data_cleaner tests.test_metrics tests.test_anomaly tests.test_pipeline -v
python -m unittest tests.test_env_loader tests.test_llm_config tests.test_deepseek_client tests.test_ai_schema tests.test_agent_payload tests.test_agent tests.test_main_cli -v
python -m unittest tests.test_rag_document_loader tests.test_rag_embedding tests.test_rag_retriever tests.test_rag_agent -v
python -m unittest tests.test_workflow tests.test_workflow_cli -v
python -m unittest tests.test_frontend_helpers tests.test_frontend_app tests.test_issues tests.test_issues_page -v

# 各阶段离线自检
python scripts/phase2_smoke_test.py            # 预期 11/11
python scripts/build_knowledge_base.py demo    # 预期 10/10
python run_workflow.py --no-llm --quiet        # 离线跑通多 Agent 工作流
```

**演示数据的已知答案**（固定随机种子，可用来核对运行结果）：

| 校验项 | 预期值 |
| --- | --- |
| 原始 / 剔除 / 清洗后记录数 | 697 / 2 / 695 |
| 项目数量 / 机器人数量 | 6 / 23 |
| 平均运行时长 / 故障率 / 平均满意度 | 11.39 小时 / 2.06% / 90.91 分 |
| 总运营成本 / 平均节降率 | 1,111,739.60 元 / 16.73% |
| 异常命中记录数 | 77 条（节降率 32 / 满意度 27 / 运行率 15 / 故障率 3） |

测试还覆盖数据安全：问题库中不含密钥、日志不写完整密钥、`.env` 不出现在待提交列表。

## 17. 安全与隐私

| 项目 | 说明 |
| --- | --- |
| 密钥管理 | 只放在 `.env`（已 gitignore）；仓库不包含任何真实密钥，`.env.example` 仅含占位符 |
| 日志 | 大模型调用日志对密钥脱敏；不打印请求头中的 Authorization |
| 数据外发 | 只有调用 DeepSeek 对话接口时会发送**结构化分析结果**（指标、异常、案例摘要），不发送原始明细文件、不发送 `.env` |
| 本地存储 | 向量库与问题库都在本地（`data/`），不使用任何云端存储 |
| 演示数据 | `data/raw/robot_operation_demo.xlsx` 与 `knowledge/` 全部为模拟数据，不含真实企业信息 |
| 使用真实数据时 | 请自行确认数据脱敏与合规，尤其是包含客户、人员或成本明细时 |

## 18. 常见问题

**Q1：`ModuleNotFoundError: No module named 'pandas'`**
当前解释器没有安装依赖。确认虚拟环境是否激活（`python -c "import sys; print(sys.executable)"`），
然后执行 `python -m pip install -r requirements.txt`。若编辑器选中了一个空的 `.venv`，重新选择解释器或删除该目录。

**Q2：控制台中文乱码**
程序启动时会自动把 Windows 控制台切到 UTF-8；若仍乱码，在终端执行 `chcp 65001` 后重试。

**Q3：提示「数据缺少必需列」**
报错会列出缺失列与实际列名，对照第 10 节补齐即可；常见别名（`项目`、`满意度`、`成本`）已自动映射。

**Q4：`.xls` 报错不支持**
请先在 Excel 中另存为 `.xlsx`（旧格式需要额外依赖，本项目未引入）。

**Q5：写入结果时报「文件被占用」**
关闭正在打开该 Excel 的窗口后重试。

**Q6：DeepSeek 调用失败**

| 提示 | 处理 |
| --- | --- |
| `未检测到环境变量 DEEPSEEK_API_KEY` | 复制 `.env.example` 为 `.env` 并填写 Key |
| `401 认证失败` | Key 无效 / 过期，或值里带了引号与空格 |
| `402 余额不足` | 到 DeepSeek 平台充值 |
| `429 触发限流` | 程序会自动退避重试，也可调大 `DEEPSEEK_MAX_RETRIES` |
| `请求超时` | 调大 `DEEPSEEK_TIMEOUT`（如 120） |
| `无法连接 DeepSeek` | 检查网络 / 代理 / 防火墙，确认能访问 `https://api.deepseek.com` |

不想花钱调试时，用 `--no-llm`（离线模板）或 `--dry-run`（只打印不发请求）。

**Q7：RAG 检索不到案例**
先用 `python scripts/build_knowledge_base.py search "你的问题"` 验证；
若确实不相关，可调低 `ROBOTOPS_RAG_MIN_SIMILARITY`（如 0.30）或补充更贴近的案例。

**Q8：改了 Embedding 维度 / 切分参数后检索异常**
索引会检测到参数变化并提示重建：`python scripts/build_knowledge_base.py rebuild`。

**Q9：Streamlit 端口被占用**
`streamlit run frontend/app.py --server.port 8502`。

**Q10：安装依赖较慢、体积较大**
ChromaDB 会带来 onnxruntime、kubernetes 等依赖，属于向量库自身要求；本项目未额外引入深度学习框架。

## 19. 后续规划

当前版本已覆盖「数据分析 → AI 诊断 → RAG → 多 Agent → 可视化 → 整改闭环」全链路，后续可选方向：

- 多用户与权限（当前为单机工具，无登录体系）
- 移动端 / 多租户支持
- 定时调度与异常告警推送
- 更多 Embedding 方案（如接入本地语义模型）
- 与工单系统对接

## 20. 许可与免责声明

- 本项目使用 **MIT License**，详见仓库根目录的 [`LICENSE`](LICENSE)。
  > 如果你是仓库作者，记得先把 `LICENSE` 中的 `<Your Name or GitHub Username>` 换成你的署名。
- 仓库内的**演示数据、知识库案例均为模拟内容**，不代表任何真实企业、真实设备或真实运营结果；
  大模型输出的诊断与建议仅供运营参考，不构成任何形式的承诺或专业意见。
- 使用本工具处理真实业务数据时，请自行确保数据合规与脱敏，并妥善保管 API Key。


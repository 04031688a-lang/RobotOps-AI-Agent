# RobotOps AI · 机器人运营智能分析平台

> 上传运营数据 → 自动算指标、查异常 → AI 诊断 + 历史案例 → 整改建议 → 工单闭环验证。
> 无需 API Key 即可跑通离线分析；接入 DeepSeek 后获得可解释的诊断与报告。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#环境要求)
[![Tests](https://img.shields.io/badge/Tests-417%20passed-success)](#测试)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

面向机器人 / 智能设备运营场景的数据分析与问题管理工具。把 Excel/CSV 运营明细自动变成可执行的运营建议和可跟踪的整改任务，定位为**个人 / 小团队可离线运行的本地工具**。

解决的痛点：异常发现滞后、经验随人员流失、报表里的异常无人跟进。

---

## 核心功能

- **数据质量检查**：重复 / 缺值 / 越界 / 类型错误统一清洗，输出问题清单
- **6 项核心指标** + 趋势分析 + 项目横向对比（全部 Pandas 计算）
- **4 条异常规则**：故障率 / 满意度 / 运行率 / 节降率，含高 / 中 / 低分级
- **多 Agent 工作流**（LangGraph）：Data Analysis → Diagnosis → RAG → Recommendation → Report
- **本地 RAG**：自研哈希 Embedding，不下载模型，17 篇历史案例可追溯
- **整改闭环**：异常 → 工单 → 重新上传数据 → 程序计算改善幅度 → 验证关闭（SQLite）
- **全链路离线降级**：`--no-llm` 不消耗任何 LLM 额度

---

## 系统架构

```text
用户上传 Excel/CSV
      ↓
数据质量检查 ──→ Pandas 指标/趋势/对比 ──→ 异常检测（程序判定）
      ↓                                            ↓
      └─────────── LangGraph 工作流 ───────────────┘
            Data Analysis → Abnormal Check
                 ├─ 无异常 ────────────────→ Report
                 └─ 有异常 → Diagnosis(LLM) → RAG → Recommendation(LLM) → Report
      ↓
运营问题（SQLite）→ 整改 → 重传数据 → 效果验证（程序计算）→ 关闭
```

**关键边界**：Python 负责计算与判定（指标、异常、改善幅度），LLM 只做自然语言解释与建议，RAG 提供历史案例参考，LangGraph 负责流程编排。大模型不参与「是否异常」的判断。

---

## RAG 架构

```text
异常清单（程序产出）
   ↓ 按异常类型构造检索词
本地哈希 Embedding（BLAKE2b + 字符 n-gram + 领域词，1024 维，确定性）
   ↓
ChromaDB 余弦检索 → 相似度阈值过滤（≥0.35）→ Top-K 聚合
   ↓
返回案例编号 / 标题 / 来源 / 相似度，供诊断与建议引用
```

> DeepSeek 无 embedding 接口，故采用本地方案：零依赖、离线可用、结果可复现。

---

## 数据模型（E-R 图）

运营问题库为 SQLite（WAL 模式，默认 `data/issues.db`），3 张表，结构化字段存 JSON。

```text
issues（问题主表）
  PK issue_id  TEXT  编号 ISSUE-YYYYMM-NNN
     project / title / anomaly_type / priority / owner / status
     created_at / updated_at / closed_at
     metrics / current_data / diagnosis / cases / recommendations / verification  (JSON)
     resolution / source_file
        │ 1
        │
        │ N
issue_notes（处理记录）
  PK id  INTEGER 自增
  FK issue_id → issues.issue_id (ON DELETE CASCADE)
     created_at / author / kind / text

meta（元信息）  key(PK) / value

索引：idx_issues_status / idx_issues_project / idx_notes_issue
```

工单状态：待处理 → 处理中 → 待验证 → 已完成 → 已关闭（支持回退与重开）。

---

## 技术栈

| 层 | 选型 |
| --- | --- |
| 数据处理 | pandas / numpy / openpyxl |
| 向量库 | ChromaDB + 自研哈希 Embedding |
| 工作流 | LangGraph |
| LLM | DeepSeek（OpenAI 兼容，标准库 urllib，指数退避重试） |
| 前端 | Streamlit |
| 持久化 | SQLite（标准库，WAL） |
| 测试 | pytest（417 passed / 169 subtests） |

---

## 项目结构

```text
robotops/    核心库：数据加载/清洗/指标/异常/趋势/对比/RAG/Agent/工单
app/         Phase 4 LangGraph 工作流（agents + graph）
frontend/    Streamlit 界面（分析工作台 + 运营问题）
knowledge/   17 篇历史案例（设备故障/维保/运营/满意度）
tests/       37 个测试文件
scripts/     演示数据生成等
data/raw/robot_operation_demo.xlsx   内置 Demo 数据
```

---

## 快速开始

```bash
git clone <your-repo-url> robotops-ai
cd robotops-ai
python -m venv .venv && .venv\Scripts\activate      # Windows
python -m pip install -r requirements.txt

# 1) 离线分析（无需 Key）—— 数据质量 → 指标 → 异常 → 报告
python run_analysis.py

# 2) 多 Agent 工作流（离线）—— 数据分析 → RAG → Agent → 报告
python run_workflow.py --no-llm

# 3) 启用 AI（需配置 Key）
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
python run_workflow.py

# 4) 可视化界面
streamlit run frontend/app.py
```

**环境要求**：Python 3.10+（已在 3.12 验证）。完整配置项见 [.env.example](.env.example)。

---

## Demo 数据

内置 `data/raw/robot_operation_demo.xlsx`：6 项目 / 23 机器人 / 30 天，刻意覆盖全部 4 类异常：

| 异常类型 | 演示对象 |
| --- | --- |
| 故障率异常 | 天津港巡检 TJ-INS-05、深圳安防 SZ-SEC-03 |
| 满意度下降 | 苏州清洁 SZ-CLN-04 |
| 运行率下降 | 深圳安防 SZ-SEC-02 |
| 成本 / 节降率下降 | 武汉消杀项目整体 |

另含 7 条「脏数据」用于验证清洗逻辑。

---

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

结果：`417 passed, 169 subtests passed`。所有阶段均有离线 / 干跑路径，CI 不消耗 LLM 额度。

---

## 项目难点与关键设计

1. **不用 LLM 算数据**：数值计算与异常判定归程序，LLM 只做解释，杜绝幻觉数字；原始 Excel 不离开本机。
2. **故障率的判定粒度**：比率型指标单日样本过小，故默认按「机器人 + 周期」累计判定。
3. **RAG 必要性**：通用大模型缺企业内部经验，本地案例让建议有出处、可追溯。
4. **容错**：单 Agent 失败不阻断，错误结构化记录；LLM 失败后报告由程序模板兜底产出。
5. **整改验证客观化**：改善幅度 = |改后−改前|/|改前|，全程程序计算，不依赖大模型判断效果。

---

## 局限

哈希 Embedding 语义能力弱于预训练模型；案例库仅 17 篇；仅适配 DeepSeek；单用户本地工具，无多租户/权限/并发；未做生产部署；异常检测为阈值型。详见各阶段代码，README 不描述未实现功能。

## 许可证

[MIT](LICENSE)

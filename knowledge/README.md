# RobotOps AI 机器人运营问题案例知识库（RAG）

> **重要声明**：本目录下的全部案例均为**模拟业务知识（虚构内容）**，用于演示 RAG 检索与
> 大模型分析流程，**不代表任何真实企业、真实项目或真实设备的数据**。
> 每个案例的元数据中都带有 `data_nature: 模拟案例（虚构）` 标记，检索结果会把这个标记一并传给大模型。

## 目录结构

```text
knowledge/
├── equipment_fault/    # 设备故障类：重复故障、维修频繁、维护周期过长、部件老化
├── maintenance/        # 维修与售后类：响应慢、备件不足、返修、维保排期
├── satisfaction/       # 用户满意度类：清洁效果、投诉响应、噪音
└── operation/          # 运营类：运行率下降、设备闲置、调度异常、成本与节降率、巡检不到位
```

## 案例文件格式

每个案例是一个 Markdown 文件，由「元数据（front matter）+ 正文小节」组成：

```markdown
---
case_id: CASE-FAULT-001
title: 清洁机器人重复故障（同一部件反复损坏）
case_type: 机器人重复故障
category: equipment_fault
severity_hint: 高
applicable_anomaly: 故障率偏高
robot_types: [清洁机器人]
keywords: [重复故障, 故障率升高, 维修次数增加]
data_nature: 模拟案例（虚构）
---

# CASE-FAULT-001 清洁机器人重复故障（同一部件反复损坏）

## 问题
## 现象
## 可能原因
## 处理措施
## 处理结果
## 经验总结
```

必填小节：**问题、现象、可能原因、处理措施、处理结果**；`经验总结` 可选。

## 如何新增案例

1. 在对应分类目录下新建 `.md` 文件（建议文件名与 `case_id` 一致，例如 `CASE-OPS-006.md`）；
2. 复制上面的格式，填写元数据与 5 个必填小节；
3. `case_id` 在同一分类内唯一即可（重建索引时会校验重复）；
4. 执行重建索引命令，让新案例进入向量库：

```powershell
python scripts\build_knowledge_base.py rebuild
```

## 索引与向量库

- 向量库：ChromaDB 本地持久化，默认位置 `data/chroma/`（已在 `.gitignore` 中忽略，可随时删除重建）；
- 集合名与 Embedding 方案记录在集合元数据中，若更换 Embedding 方案需执行 `rebuild`；
- 文本切分：按 Markdown 小节切块（超长小节会二次切分），每个文本块都保留
  `case_id / title / case_type / category / section / keywords` 等元数据，便于结果溯源。


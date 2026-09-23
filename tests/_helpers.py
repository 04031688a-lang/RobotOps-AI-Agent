"""测试公共工具。"""

from __future__ import annotations

import json
import atexit
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from robotops import config as robotops_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 保证直接运行测试文件时也能导入 robotops
    sys.path.insert(0, str(PROJECT_ROOT))

from robotops import config  # noqa: E402
from robotops.llm.client import HttpRequest, HttpResponse  # noqa: E402


def base_operation_frame() -> pd.DataFrame:
    """构造一份最小的干净运营数据（中文列名）。"""

    return pd.DataFrame(
        [
            {
                config.COL_DATE: "2026-08-01",
                config.COL_PROJECT: "项目甲",
                config.COL_ROBOT_ID: "R-01",
                config.COL_ROBOT_TYPE: "清洁机器人",
                config.COL_RUNNING_HOURS: 8.0,
                config.COL_PLANNED_HOURS: 10.0,
                config.COL_FAULT_COUNT: 1,
                config.COL_INSPECTION_COUNT: 10,
                config.COL_REPAIR_COUNT: 0,
                config.COL_SATISFACTION: 90.0,
                config.COL_COST: 100.0,
                config.COL_SAVING_RATE: 20.0,
            },
            {
                config.COL_DATE: "2026-08-01",
                config.COL_PROJECT: "项目甲",
                config.COL_ROBOT_ID: "R-02",
                config.COL_ROBOT_TYPE: "巡检机器人",
                config.COL_RUNNING_HOURS: 6.0,
                config.COL_PLANNED_HOURS: 10.0,
                config.COL_FAULT_COUNT: 2,
                config.COL_INSPECTION_COUNT: 10,
                config.COL_REPAIR_COUNT: 1,
                config.COL_SATISFACTION: 80.0,
                config.COL_COST: 200.0,
                config.COL_SAVING_RATE: 10.0,
            },
            {
                config.COL_DATE: "2026-08-02",
                config.COL_PROJECT: "项目乙",
                config.COL_ROBOT_ID: "R-03",
                config.COL_ROBOT_TYPE: "配送机器人",
                config.COL_RUNNING_HOURS: 10.0,
                config.COL_PLANNED_HOURS: 10.0,
                config.COL_FAULT_COUNT: 0,
                config.COL_INSPECTION_COUNT: 10,
                config.COL_REPAIR_COUNT: 0,
                config.COL_SATISFACTION: 95.0,
                config.COL_COST: 500.0,
                config.COL_SAVING_RATE: 30.0,
            },
        ]
    )


def write_excel(frame: pd.DataFrame, path: Path, sheet_name: str = "运营数据") -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


# ---------------------------------------------------------------------------
# Phase 2 测试工具：假传输层与示例返回
# ---------------------------------------------------------------------------
class FakeTransport:
    """可编排的假 HTTP 传输层，用于在无网络、无 API Key 的情况下测试调用逻辑。

    - 传入 ``HttpResponse`` 时返回该响应；
    - 传入 ``Exception`` 时抛出该异常（用于模拟超时 / 网络错误）；
    - 响应数量少于调用次数时，重复使用最后一个响应。
    """

    def __init__(self, *responses: HttpResponse | Exception) -> None:
        if not responses:
            raise ValueError("FakeTransport 至少需要一个响应")
        self.responses: list[HttpResponse | Exception] = list(responses)
        self.requests: list[HttpRequest] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        item = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last_request_json(self) -> dict[str, Any]:
        payload = json.loads(self.requests[-1].body.decode("utf-8"))
        return payload


def deepseek_response(
    content: str | None,
    *,
    model: str = "deepseek-chat",
    usage: dict[str, int] | None = None,
    finish_reason: str | None = "stop",
    request_id: str = "req-test",
) -> HttpResponse:
    """构造一个 OpenAI/DeepSeek 兼容的成功响应。"""

    body = {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    return HttpResponse(status=200, body=json.dumps(body, ensure_ascii=False), headers={"x-request-id": request_id})


def deepseek_error_response(status: int, message: str = "接口错误") -> HttpResponse:
    """构造一个错误响应（DeepSeek 错误体形如 {"error": {"message": ...}}）。"""

    body = {"error": {"message": message, "type": "test_error", "code": status}}
    return HttpResponse(status=status, body=json.dumps(body, ensure_ascii=False), headers={})


def sample_ai_payload() -> dict[str, Any]:
    """一份符合要求的 Agent 输出示例。"""

    return {
        "overview": "本期共 6 个项目、23 台机器人，平均运行率 92.4%、平均满意度 90.91 分。",
        "key_findings": [
            {"finding": "故障集中在少数设备", "evidence": "累计故障率 10.96% vs 同项目 0.88%", "metric": "故障率"}
        ],
        "abnormal_projects": [
            {
                "project": "华北-天津港智能巡检项目",
                "abnormal_type": "故障率偏高",
                "metric_value": "10.96%",
                "threshold": "5%",
                "severity": "高",
                "evidence": "累计 41 次故障 / 374 次巡检",
            }
        ],
        "possible_reasons": [
            {
                "reason": "推测：单台设备部件劣化",
                "confidence": "中",
                "based_on": "该台维修 8 次且单位运行小时成本高于同项目其他台",
                "data_gap": "当前数据不足以判断，缺少故障类型与部件数据",
            }
        ],
        "recommendations": [
            {
                "action": "对高故障机器人做单机专项复盘",
                "priority": "高",
                "target": "TJ-INS-05",
                "expected_effect": "故障率向项目均值收敛",
                "verification": "补齐故障类型字段后重新计算累计故障率",
            }
        ],
    }


def sample_ai_json(*, fenced: bool = False) -> str:
    text = json.dumps(sample_ai_payload(), ensure_ascii=False, indent=2)
    return f"```json\n{text}\n```" if fenced else text


# ---------------------------------------------------------------------------
# Phase 3 测试工具
# ---------------------------------------------------------------------------
def release_chroma_client(retriever_or_store: Any) -> None:
    """释放 ChromaDB 持有的 sqlite 句柄，保证临时目录可被删除（Windows 必需）。"""

    try:
        from chromadb.api.shared_system_client import SharedSystemClient  # type: ignore

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    store = getattr(retriever_or_store, "store", retriever_or_store)
    if store is not None:
        try:
            store._client = None
        except Exception:
            pass
    gc.collect()


def make_temp_chroma_dir(prefix: str = "robotops_test_chroma_") -> Path:
    """在 output/ 下创建临时向量库目录（该目录已被 .gitignore 忽略）。

    说明：ChromaDB 在 Windows 上会持有 sqlite 文件句柄，导致刚创建的目录无法立即删除，
    因此这里顺带清理若干分钟前的残留目录，并注册进程退出时的兜底清理。
    """

    purge_temp_chroma_dirs(min_age_seconds=300.0, prefix=prefix)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=PROJECT_ROOT / "output"))


def cleanup_dir(path: Path) -> None:
    """尽力清理临时目录（Windows 上文件可能仍被占用，忽略失败）。"""

    shutil.rmtree(path, ignore_errors=True)


def release_local_chroma(retriever_or_store: Any) -> None:
    """只释放当前对象的客户端引用（不影响其他测试的 Chroma 系统缓存）。"""

    store = getattr(retriever_or_store, "store", retriever_or_store)
    if store is not None:
        try:
            store._client = None
        except Exception:
            pass
    gc.collect()


def purge_temp_chroma_dirs(*, min_age_seconds: float = 0.0, prefix: str = "robotops_test_") -> int:
    """删除测试用临时目录（默认不限年龄，覆盖向量库与工作流测试目录）。"""

    root = PROJECT_ROOT / "output"
    if not root.exists():
        return 0
    now = time.time()
    removed = 0
    for path in root.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        try:
            if min_age_seconds > 0 and (now - path.stat().st_mtime) < min_age_seconds:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    return removed


atexit.register(purge_temp_chroma_dirs)  # 进程退出时句柄已释放，可彻底清理


# ---------------------------------------------------------------------------
# Phase 4 测试工具
# ---------------------------------------------------------------------------
def make_temp_dir(prefix: str = "robotops_test_") -> Path:
    """在 output/ 下创建临时目录（已被 .gitignore 忽略）。"""

    purge_temp_chroma_dirs(min_age_seconds=300.0, prefix=prefix)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=PROJECT_ROOT / "output"))


def build_no_anomaly_frame(days: int = 5) -> pd.DataFrame:
    """构造一份**不含任何异常**的运营数据：

    运行率 100%（>=80%）、满意度 95 分（>=85）、节降率 20%（>=10）、
    故障率 0%（<=5%），用于验证「正常数据不触发诊断流程」。
    """

    rows = []
    for offset in range(days):
        rows.append(
            {
                robotops_config.COL_DATE: f"2026-09-{offset + 1:02d}",
                robotops_config.COL_PROJECT: "测试-正常项目",
                robotops_config.COL_ROBOT_ID: "TEST-ROBOT-01",
                robotops_config.COL_ROBOT_TYPE: "清洁机器人",
                robotops_config.COL_RUNNING_HOURS: 8.0,
                robotops_config.COL_PLANNED_HOURS: 8.0,
                robotops_config.COL_FAULT_COUNT: 0,
                robotops_config.COL_INSPECTION_COUNT: 10,
                robotops_config.COL_REPAIR_COUNT: 0,
                robotops_config.COL_SATISFACTION: 95.0,
                robotops_config.COL_COST: 800.0,
                robotops_config.COL_SAVING_RATE: 20.0,
            }
        )
    return pd.DataFrame(rows)


def write_no_anomaly_data(path: Path, *, days: int = 5) -> Path:
    """把「无异常」数据集写入 Excel。"""

    return write_excel(build_no_anomaly_frame(days=days), path)


def diagnosis_ai_json() -> str:
    return json.dumps(
        {
            "summary": "推测：异常集中在少数设备，可能与部件劣化或作业强度有关。",
            "possible_reasons": [
                {
                    "reason": "推测：高故障设备可能存在部件劣化或作业强度偏高",
                    "confidence": "中",
                    "based_on": "该设备累计故障率高于同项目其他设备，且维修次数同步偏高",
                    "data_gap": "当前数据不足以判断，缺少故障类型与部件明细",
                }
            ],
        },
        ensure_ascii=False,
    )


def recommendation_ai_json() -> str:
    return json.dumps(
        {
            "summary": "建议先定位高故障设备的根因，再优化运行率与成本结构。",
            "recommendations": [
                {
                    "action": "对该设备做单机专项复盘",
                    "priority": "高",
                    "target": "高故障机器人",
                    "expected_effect": "定位重复故障根因",
                    "verification": "补充故障类型字段后重新计算累计故障率",
                    "reference_case": "CASE-FAULT-001",
                }
            ],
        },
        ensure_ascii=False,
    )


def report_markdown() -> str:
    return (
        "# RobotOps AI 机器人运营分析报告（多 Agent 工作流）\n\n"
        "## 一、运营概览\n\n当前项目整体运行情况（来自程序指标）。\n\n"
        "## 二、关键指标\n\n| 指标 | 数值 | 单位 |\n| --- | --- | --- |\n| 项目数量 | 6 | 个 |\n\n"
        "## 三、异常发现\n\n异常由程序按阈值判定。\n\n"
        "## 四、可能原因（推测）\n\n推测：可能与部件劣化有关。\n\n"
        "## 五、历史相似案例（仅供参考）\n\n模拟案例，仅供参考。\n\n"
        "## 六、优化建议\n\n建议做单机专项复盘。\n\n"
        "## 七、数据与口径说明\n\n指标由 Phase 1 Pandas 模块计算。\n"
    )


def workflow_llm_responses() -> tuple[str, str, str]:
    """按工作流调用顺序返回（诊断 / 建议 / 报告）三个模型响应。"""

    return diagnosis_ai_json(), recommendation_ai_json(), report_markdown()

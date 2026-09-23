"""RobotOps AI —— Phase 2 离线自检脚本。

作用：用**模拟的 DeepSeek 返回**跑通 Phase 2 全链路（Pandas 分析 → 结构化载荷 →
Agent 解析 → 结果渲染/导出），并逐项验证异常处理是否符合预期。

特点：**不需要 API Key，不产生任何网络请求与费用**，适合在配置真实密钥之前先确认代码可跑。

用法：
    python scripts/phase2_smoke_test.py
    python scripts/phase2_smoke_test.py --data data/raw/robot_operation_demo.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotops.agent.robot_ops_agent import RobotOpsAnalysisAgent  # noqa: E402
from robotops import config as project_config  # noqa: E402
from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.llm.errors import (  # noqa: E402
    LLMAuthError,
    LLMEmptyResponseError,
    LLMError,
    LLMResponseFormatError,
    LLMTimeoutError,
)
from robotops.llm.logger import setup_llm_logger  # noqa: E402
from robotops.pipeline import run_analysis  # noqa: E402

try:  # 复用测试工具里的假传输层与示例返回
    from tests._helpers import deepseek_error_response, deepseek_response, sample_ai_json
except ImportError:  # pragma: no cover - 缺少 tests 包时的兜底
    deepseek_error_response = deepseek_response = sample_ai_json = None  # type: ignore


def _build_client(transport, *, max_retries: int = 0) -> DeepSeekClient:
    config = LLMConfig(api_key="sk-smoke-test", max_retries=max_retries)
    return DeepSeekClient(
        config=config,
        transport=transport,
        logger=setup_llm_logger(level="WARNING", console=False),
        sleep=lambda seconds: None,
    )


class _ScriptedTransport:
    """按脚本依次返回响应或抛出异常。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        item = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "[通过]" if condition else "[失败]"
    print(f"  {status} {name}" + (f" —— {detail}" if detail else ""))
    return condition


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="Phase 2 离线自检（不调用真实 API）")
    parser.add_argument("-d", "--data", dest="data_path", default=None, help="数据文件路径")
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Excel 工作表名称")
    args = parser.parse_args(argv if argv is not None else [])

    if sample_ai_json is None:
        print("无法导入 tests._helpers，请在项目根目录运行本脚本。")
        return 2

    print("=" * 78)
    print("RobotOps AI Phase 2 离线自检（不消耗 API、不需要 API Key）")
    print("=" * 78)

    print("\n[1] 运行 Phase 1 Pandas 分析")
    result = run_analysis(args.data_path, sheet_name=args.sheet_name, export=False, verbose=False)
    print(
        f"      记录 {result.metrics.record_count} 条 / 项目 {result.metrics.project_count} 个 / "
        f"机器人 {result.metrics.robot_count} 台 / 异常 {result.anomaly_count} 条"
    )

    passed = 0
    total = 0

    print("\n[2] 结构化载荷（发送给模型的内容）")
    agent = RobotOpsAnalysisAgent(
        config=LLMConfig(api_key="sk-smoke-test"), logger=setup_llm_logger(level="WARNING", console=False)
    )
    payload = agent.build_payload(result)
    core = payload["core_metrics"]
    total += 1
    passed += _check(
        "载荷包含 core_metrics 且数值与 Phase 1 一致",
        core["project_count"] == result.metrics.project_count
        and core["robot_count"] == result.metrics.robot_count,
        f"project_count={core['project_count']} robot_count={core['robot_count']}",
    )
    total += 1
    passed += _check("载荷不含原始明细（仅结构化结果）", "records" not in payload)

    print("\n[3] 正常返回：完整链路与结果导出")
    transport = _ScriptedTransport(deepseek_response(sample_ai_json()))
    fake_agent = RobotOpsAnalysisAgent(
        config=LLMConfig(api_key="sk-smoke-test", max_retries=0),
        client=_build_client(transport),
        logger=setup_llm_logger(level="WARNING", console=False),
    )
    run = fake_agent.analyze(result)
    analysis = run.analysis
    total += 1
    passed += _check(
        "解析出 5 个结构化分区",
        bool(analysis)
        and bool(analysis.overview)
        and len(analysis.abnormal_projects) >= 1
        and len(analysis.recommendations) >= 1,
    )
    total += 1
    passed += _check("模型返回未出现校验告警", not analysis.warnings, str(analysis.warnings))

    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "ai_analysis.json"
        md_path = Path(tmp) / "ai_analysis.md"
        json_path.write_text(
            json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path.write_text(analysis.to_markdown(), encoding="utf-8")
        total += 1
        passed += _check("结果可导出为 JSON 与 Markdown", json_path.stat().st_size > 0 and md_path.stat().st_size > 0)

    print("\n[4] 异常场景处理")
    # 失败场景会生成 output/deepseek_raw_response.txt，这里重定向到临时目录，避免污染项目输出
    _temp_output = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    _output_patch = mock.patch.object(
        project_config, "OUTPUT_DIR", Path(_temp_output.name)
    )
    _output_patch.start()
    scenarios = [
        ("401 密钥无效 → LLMAuthError", LLMAuthError, _ScriptedTransport(deepseek_error_response(401, "Authentication Fails")), 0),
        ("429 限流后自动重试成功", None, _ScriptedTransport(deepseek_error_response(429, "rate limit"), deepseek_response(sample_ai_json())), 1),
        ("请求超时 → LLMTimeoutError", LLMTimeoutError, _ScriptedTransport(LLMTimeoutError("请求 DeepSeek 超时")), 0),
        ("返回为空 → LLMEmptyResponseError", LLMEmptyResponseError, _ScriptedTransport(deepseek_response("")), 0),
        ("返回非 JSON → LLMResponseFormatError", LLMResponseFormatError, _ScriptedTransport(deepseek_response("这不是 JSON")), 0),
    ]
    for name, expected_error, scenario_transport, max_retries in scenarios:
        total += 1
        scripted_agent = RobotOpsAnalysisAgent(
            config=LLMConfig(api_key="sk-smoke-test", max_retries=max_retries),
            client=_build_client(scenario_transport, max_retries=max_retries),
            logger=setup_llm_logger(level="WARNING", console=False),
        )
        try:
            scenario_run = scripted_agent.analyze(result)
            ok = expected_error is None and scenario_run.attempts == max_retries + 1
            detail = f"attempts={scenario_run.attempts}"
        except LLMError as error:
            ok = expected_error is not None and isinstance(error, expected_error)
            detail = f"{type(error).__name__}"
        passed += _check(name, ok, detail)

    _output_patch.stop()
    logging.shutdown()  # 关闭日志文件句柄，便于清理临时目录
    _temp_output.cleanup()

    print("\n[5] 缺少 API Key 的提示")
    total += 1
    try:
        LLMConfig(api_key=None).validate(require_api_key=True)
        passed += _check("缺少 API Key 时抛出明确异常", False)
    except LLMError as error:
        passed += _check(
            "缺少 API Key 时抛出明确异常",
            "DEEPSEEK_API_KEY" in str(error),
            str(error)[:60],
        )

    print("\n" + "=" * 78)
    print(f"自检结果：{passed}/{total} 项通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

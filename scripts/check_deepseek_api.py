r"""RobotOps AI —— Phase 2 DeepSeek API 连通性检查。

用途：一条命令回答「DeepSeek API 到底接上没有」。

检查内容：
  1. .env / 环境变量是否被正确读取（API Key 脱敏显示）；
  2. 配置是否合法（地址、模型、超时、重试）；
  3. 发送一次**真实的极小请求**，确认网络与密钥可用（消耗极少 token，约几十个）；
  4. 用 JSON 输出模式再请求一次，确认 Agent 依赖的结构化返回可用。

用法：
    python scripts/check_deepseek_api.py
    python scripts/check_deepseek_api.py --timeout 120
    python scripts/check_deepseek_api.py --skip-json-check     # 只做基础连通性检查
    python scripts/check_deepseek_api.py --env-file D:/config/robotops.env

退出码：0 接通；2 配置问题（如缺少 API Key）；3 调用失败（网络/密钥/限流等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.llm.client import DeepSeekClient  # noqa: E402
from robotops.llm.config import LLMConfig  # noqa: E402
from robotops.llm.env_loader import find_env_file  # noqa: E402
from robotops.llm.errors import LLMError, MissingAPIKeyError  # noqa: E402
from robotops.llm.logger import setup_llm_logger  # noqa: E402

PLAIN_PROMPT = "请只回复四个字：连接正常"
JSON_PROMPT = '请只返回一个 JSON 对象，不要任何其他文字：{"connected": true}'


def _log_file_path(logger: object) -> str:
    """取出日志文件路径（找不到时返回 "-"）。"""

    for handler in getattr(logger, "handlers", []) or []:
        filename = getattr(handler, "baseFilename", None)
        if filename:
            return str(filename)
    return "-"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_deepseek_api.py",
        description="检查 DeepSeek API 是否已接通（会发送真实请求，消耗极少 token）",
    )
    parser.add_argument("--env-file", dest="env_file", default=None, help="指定 .env 路径")
    parser.add_argument("--timeout", type=float, default=None, help="覆盖超时秒数，例如 120")
    parser.add_argument("--max-tokens", type=int, default=32, help="本次检查的最大输出长度，默认 %(default)s")
    parser.add_argument("--skip-json-check", action="store_true", help="跳过 JSON 输出模式检查")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 %(default)s")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    width = 78
    print("=" * width)
    print("RobotOps AI Phase 2 —— DeepSeek API 连通性检查")
    print("=" * width)

    # ---- 1. .env 与配置 ------------------------------------------------
    print("\n[1/4] 检查 .env 与配置")
    env_file = find_env_file(args.env_file)
    print(f"      .env 文件  ：{env_file if env_file else '未找到（将只使用系统环境变量）'}")

    config = LLMConfig.from_env(env_file=args.env_file)
    if args.timeout is not None:
        config = replace(config, timeout=float(args.timeout))
    public = config.to_public_dict()
    print(f"      API Key    ：{public['api_key']}")
    print(f"      模型       ：{public['model']}")
    print(f"      接口地址   ：{config.chat_url}")
    print(f"      超时/重试  ：{public['timeout_seconds']}s / {public['max_retries']} 次")

    try:
        config.validate(require_api_key=True)
    except MissingAPIKeyError as error:
        print(f"\n[未接通] {error.describe()}")
        return 2
    except LLMError as error:
        print(f"\n[未接通] {error.describe()}")
        return 2

    if "请替换" in (config.api_key or "") or "your" in (config.api_key or "").lower():
        print(
            "\n[注意] 当前 API Key 看起来仍是 .env.example 里的占位符，"
            "请填入 DeepSeek 控制台创建的真实密钥（形如 sk-xxxxxxxx）。"
        )
        return 2

    logger = setup_llm_logger(level=args.log_level)
    client = DeepSeekClient(config=config, logger=logger)

    # ---- 2. 基础连通性（真实请求）--------------------------------------
    print("\n[2/4] 发送真实请求（基础对话，消耗极少 token）")
    try:
        result = client.chat(
            [{"role": "user", "content": PLAIN_PROMPT}],
            response_format=None,  # 基础检查不使用 JSON 模式
            max_tokens=args.max_tokens,
        )
    except LLMError as error:
        print(f"\n[未接通] {error.describe()}")
        print(f"      详细日志：{_log_file_path(logger)}")
        return 3

    print(f"      HTTP 状态  ：200（成功）")
    print(f"      耗时       ：{result.elapsed_seconds:.2f}s（第 {result.attempts} 次尝试）")
    print(f"      返回模型   ：{result.model}")
    print(f"      返回内容   ：{result.content.strip()[:80]}")
    print(f"      Token 用量 ：{result.usage or '（接口未返回）'}")

    # ---- 3. JSON 输出模式（Agent 依赖）--------------------------------
    json_ok = True
    if args.skip_json_check:
        print("\n[3/4] 跳过 JSON 输出模式检查（--skip-json-check）")
    else:
        print("\n[3/4] 检查 JSON 输出模式（Agent 依赖该模式生成结构化结论）")
        try:
            json_result = client.chat(
                [{"role": "user", "content": JSON_PROMPT}], max_tokens=args.max_tokens
            )
            content = json_result.content.strip()
            try:
                parsed = json.loads(content)
                json_ok = parsed.get("connected") is True
                print(f"      返回内容   ：{content[:80]}")
                print(f"      解析结果   ：{'通过' if json_ok else '内容不是预期的 JSON，但可解析'}")
            except json.JSONDecodeError:
                json_ok = False
                print(f"      返回内容   ：{content[:80]}")
                print("      解析结果   ：注意 —— 返回内容不是合法 JSON")
                print("                  Agent 仍可兼容 ```json 代码块与前后说明文字，但建议检查模型配置")
        except LLMError as error:
            json_ok = False
            print(f"      检查失败   ：{error.describe()}")

    # ---- 4. 结论 -------------------------------------------------------
    print("\n[4/4] 检查结论")
    print("-" * width)
    print("      API 已接通：密钥有效、网络可达，Phase 2 具备完整调用能力")
    if not json_ok and not args.skip_json_check:
        print("      提示：JSON 输出模式异常，Agent 解析可能降级（可使用兼容解析，但结果稳定性下降）")
    print(f"      下一步：python main.py --quiet        # 运行完整的运营数据分析 Agent")
    print(f"      日志文件：{_log_file_path(logger)}")
    print("=" * width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

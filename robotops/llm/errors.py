"""RobotOps AI —— Phase 2 LLM 模块异常定义。

所有异常都继承 Phase 1 的 ``RobotOpsError``，因此命令行入口可以统一捕获。
每个异常都带 ``hint``（排查建议）与可选的 ``status_code``（HTTP 状态码），
便于在控制台给出可执行的错误提示，而不是抛出一堆堆栈。
"""

from __future__ import annotations

from ..exceptions import RobotOpsError


class LLMError(RobotOpsError):
    """LLM 相关错误的基础类型。"""

    default_hint: str = ""

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        status_code: int | None = None,
        raw: str | None = None,
    ) -> None:
        super().__init__(message)
        self.hint = hint or self.default_hint
        self.status_code = status_code
        self.raw = raw

    def describe(self) -> str:
        """返回“错误信息 + 排查建议”的完整文本。"""

        text = str(self)
        if self.hint:
            text = f"{text}\n排查建议：{self.hint}"
        return text


class MissingAPIKeyError(LLMError):
    """未配置 API Key。"""

    default_hint = (
        "请在项目根目录创建 .env 文件并写入 DEEPSEEK_API_KEY=你的密钥"
        "（可复制 .env.example 作为模板），或先执行 python main.py --dry-run 查看输入载荷而不调用 API。"
    )


class LLMConfigurationError(LLMError):
    """配置不合法（超时、重试次数、模型名等）。"""

    default_hint = "请检查 robotops/llm/config.py 的默认值或 .env 中的 DEEPSEEK_* 变量。"


class LLMAuthError(LLMError):
    """401 / 403：密钥无效或无权限。"""

    default_hint = (
        "确认 DEEPSEEK_API_KEY 填写正确且未过期（注意不要有多余空格或引号），"
        "并确认该密钥所属账号已开通对应模型权限。"
    )


class LLMInsufficientBalanceError(LLMError):
    """402：账户余额不足。"""

    default_hint = "登录 DeepSeek 开放平台查看账户余额并充值后重试。"


class LLMRateLimitError(LLMError):
    """429：触发限流。"""

    default_hint = "稍后重试，或提高 .env 中的 DEEPSEEK_MAX_RETRIES（当前会自动指数退避重试）。"


class LLMRequestError(LLMError):
    """400 / 404 / 422：请求参数或模型名不正确。"""

    default_hint = (
        "检查 .env 中的 DEEPSEEK_MODEL 是否为平台支持的模型名，"
        "以及 DEEPSEEK_BASE_URL 是否为 https://api.deepseek.com。"
    )


class LLMServerError(LLMError):
    """5xx：服务端错误。"""

    default_hint = "属于 DeepSeek 服务端问题，通常自动重试即可恢复；如持续失败请稍后再试。"


class LLMTimeoutError(LLMError):
    """请求超时。"""

    default_hint = (
        "本机网络较慢或模型响应较长，可调大 .env 中的 DEEPSEEK_TIMEOUT（单位秒），"
        "或缩小 DEEPSEEK_MAX_TOKENS / 传入更精简的分析数据。"
    )


class LLMNetworkError(LLMError):
    """网络不可达、DNS 解析失败、代理或证书问题。"""

    default_hint = (
        "确认本机可以访问 https://api.deepseek.com（浏览器或 curl 测试）；"
        "若在公司网络下，请检查代理设置 HTTP_PROXY / HTTPS_PROXY。"
    )


class LLMEmptyResponseError(LLMError):
    """接口返回 200，但内容为空。"""

    default_hint = "可能是模型返回被截断，可减小 DEEPSEEK_MAX_TOKENS 或换用其他模型后重试。"


class LLMResponseFormatError(LLMError):
    """返回内容不是合法的结构化 JSON，或缺少必需字段。"""

    default_hint = (
        "程序已把原始返回保存到 output/deepseek_raw_response.txt，可查看内容；"
        "通常重试一次即可，若持续出现请检查 prompts.py 中的输出格式要求。"
    )


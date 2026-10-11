"""RobotOps AI 自定义异常。

统一异常类型，方便上层（命令行入口、测试）按类型给出友好提示。
"""

from __future__ import annotations


class RobotOpsError(Exception):
    """项目基础异常类型。"""
    #: 面向用户的排查建议（子类可覆盖）
    default_hint: str = ""

    def describe(self) -> str:
        """返回「错误信息 + 排查建议」的用户可读文本（不含 traceback）。"""

        text = str(self).strip()
        hint = getattr(self, "hint", "") or getattr(self, "default_hint", "")
        if hint:
            text = f"{text}\n排查建议：{hint}"
        return text


def describe_error(error: BaseException, *, one_line: bool = False) -> str:
    """把任意异常转换为面向用户的提示，绝不包含 traceback。

    Args:
        error: 捕获到的异常对象。
        one_line: 是否压成单行（前端展示用；命令行保留换行更易读）。
    """

    describe = getattr(error, "describe", None)
    text = str(describe()) if callable(describe) else str(error)
    if one_line:
        return " ".join(text.split())
    return text.strip()


class DataLoadError(RobotOpsError):
    """数据文件不存在、格式不支持或读取失败。"""


class DataValidationError(RobotOpsError):
    """数据列缺失、列类型不符合模型要求。"""


class ConfigurationError(RobotOpsError):
    """配置参数不合法（例如阈值为空、输出目录不可写）。"""


class ReportGenerationError(RobotOpsError):
    """报告文件生成失败（输出目录不可写、Excel 文件被占用等）。"""

    default_hint = (
        "请检查输出目录是否可写；"
        "若正在用 Excel/WPS 打开同名报告文件，请先关闭后重试。"
    )

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint or self.default_hint

    def describe(self) -> str:
        """返回“错误信息 + 排查建议”的完整文本。"""

        text = str(self)
        if self.hint:
            text = f"{text}\n排查建议：{self.hint}"
        return text


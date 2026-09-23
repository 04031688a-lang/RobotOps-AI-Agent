"""RobotOps AI 自定义异常。

统一异常类型，方便上层（命令行入口、测试）按类型给出友好提示。
"""

from __future__ import annotations


class RobotOpsError(Exception):
    """项目基础异常类型。"""


class DataLoadError(RobotOpsError):
    """数据文件不存在、格式不支持或读取失败。"""


class DataValidationError(RobotOpsError):
    """数据列缺失、列类型不符合模型要求。"""


class ConfigurationError(RobotOpsError):
    """配置参数不合法（例如阈值为空、输出目录不可写）。"""


"""控制台输出编码处理（Windows 中文环境兼容）。

Python 在 Windows 下默认使用系统区域编码（中文系统通常为 cp936），
当终端代码页与 Python 输出编码不一致时，中文会显示为乱码。

本模块在程序启动时：
1. 把 Windows 控制台输入/输出代码页切换为 UTF-8（65001）；
2. 把 ``sys.stdout`` / ``sys.stderr`` 重新配置为 UTF-8。

这样无论是在 VS Code 集成终端、PowerShell 还是 cmd 中运行，
中文与特殊字符都能正常显示；非 Windows 平台调用本函数不会有副作用。
"""

from __future__ import annotations

import os
import sys


def setup_console_encoding() -> None:
    """确保控制台以 UTF-8 输出，避免中文乱码。"""

    if os.name == "nt":
        try:  # pragma: no cover - 依赖具体终端环境
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 非常规流对象
            pass


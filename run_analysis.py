"""RobotOps AI Phase 1 命令行入口（Windows + VS Code 直接可运行）。

用法：
    python run_analysis.py                 # 使用默认演示数据
    python run_analysis.py --help          # 查看全部参数
"""

from __future__ import annotations

from robotops.cli import main

if __name__ == "__main__":
    raise SystemExit(main())


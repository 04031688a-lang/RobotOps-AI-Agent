"""支持 ``python -m robotops`` 方式运行分析流程。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())


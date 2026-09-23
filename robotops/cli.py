"""RobotOps AI —— Phase 1 命令行入口实现。

示例：
    python run_analysis.py
    python run_analysis.py --data data/raw/robot_operation_demo.xlsx
    python run_analysis.py --list-sheets
    python run_analysis.py --no-export
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config
from .console_io import setup_console_encoding
from .data_loader import list_excel_sheets
from .exceptions import RobotOpsError
from .pipeline import run_analysis
from .report import print_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_analysis.py",
        description="RobotOps AI Phase 1 —— 机器人运营数据分析基础模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run_analysis.py\n"
            "  python run_analysis.py --data data/raw/robot_operation_demo.xlsx --output output\n"
            "  python run_analysis.py --data my_data.csv --encoding gbk\n"
        ),
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data_path",
        default=None,
        help=f"数据文件路径（.xlsx/.xlsm/.csv），默认 {config.DEFAULT_DATA_FILE.name}",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default=None,
        help="结果输出目录，默认 output/",
    )
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Excel 工作表名称")
    parser.add_argument("--encoding", dest="encoding", default=None, help="CSV 文件编码，如 gbk")
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="仅列出 Excel 工作表名称后退出",
    )
    parser.add_argument("--no-export", action="store_true", help="只打印报告，不导出结果文件")
    parser.add_argument("--no-excel", action="store_true", help="导出结果时不生成 Excel 汇总工作簿")
    parser.add_argument("-q", "--quiet", action="store_true", help="只输出最终报告，隐藏过程日志")
    parser.add_argument(
        "--fault-rate-upper",
        type=float,
        default=config.DEFAULT_THRESHOLDS.fault_rate_upper,
        help="故障率上限阈值(%%)，默认 %(default)s",
    )
    parser.add_argument(
        "--satisfaction-lower",
        type=float,
        default=config.DEFAULT_THRESHOLDS.satisfaction_lower,
        help="满意度下限阈值(分)，默认 %(default)s",
    )
    parser.add_argument(
        "--uptime-rate-lower",
        type=float,
        default=config.DEFAULT_THRESHOLDS.uptime_rate_lower,
        help="运行率下限阈值(%%)，默认 %(default)s",
    )
    parser.add_argument(
        "--saving-rate-lower",
        type=float,
        default=config.DEFAULT_THRESHOLDS.saving_rate_lower,
        help="节降率下限阈值(%%)，默认 %(default)s",
    )
    parser.add_argument(
        "--fault-rate-scope",
        choices=config.FAULT_RATE_SCOPE_CHOICES,
        default=config.FAULT_RATE_ANOMALY_SCOPE,
        help=(
            "故障率判定口径：robot_period=按机器人周期累计(默认)；"
            "record=按每日记录逐条判定"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_sheets:
            target = Path(args.data_path) if args.data_path else config.DEFAULT_DATA_FILE
            sheets = list_excel_sheets(target)
            print(f"{target} 的工作表：")
            for name in sheets:
                print(f"  - {name}")
            return 0

        thresholds = config.AnomalyThresholds(
            fault_rate_upper=args.fault_rate_upper,
            satisfaction_lower=args.satisfaction_lower,
            uptime_rate_lower=args.uptime_rate_lower,
            saving_rate_lower=args.saving_rate_lower,
        )

        result = run_analysis(
            data_path=args.data_path,
            output_dir=args.output_dir,
            sheet_name=args.sheet_name,
            encoding=args.encoding,
            thresholds=thresholds,
            fault_rate_scope=args.fault_rate_scope,
            export=not args.no_export,
            write_excel=not args.no_excel,
            verbose=not args.quiet,
        )
    except BrokenPipeError:
        # 输出被管道提前关闭（例如 PowerShell 的 `| Select-Object -First 5`）：
        # 把标准输出重定向到空设备，避免退出时抛出二次异常
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover - 极少数终端不支持
            pass
        return 0
    except RobotOpsError as exc:
        print(f"[分析失败] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[分析失败] 文件不存在：{exc}", file=sys.stderr)
        return 2
    except PermissionError as exc:
        print(
            f"[分析失败] 文件被占用或无写入权限：{exc}\n"
            f"提示：请先关闭正在打开该文件的 Excel 窗口后重试。",
            file=sys.stderr,
        )
        return 2

    try:
        print()
        print_report(result)
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # pragma: no cover
            pass
    except UnicodeEncodeError as exc:
        print(f"[提示] 当前终端无法显示部分字符：{exc}", file=sys.stderr)
    return 0

"""生成 Phase 1 测试用模拟数据：data/raw/robot_operation_demo.xlsx

数据规模：6 个项目 / 23 台机器人 / 30 天 = 690 条正常记录，
并在文件末尾追加 7 条“脏数据”用于验证清洗逻辑
（2 条与正常记录同日期同机器人，用于演示去重；5 条放在区间之后一天，用于演示字段级清洗）。

工作簿包含 4 个工作表：
1. 运营数据      —— 主数据表（含注入的脏数据）
2. 字段说明      —— 字段字典
3. 项目与机器人  —— 项目/机器人/类型/计划运行时长清单
4. 数据说明      —— 注入的脏数据清单与预期清洗动作

用法：
    python scripts/generate_demo_data.py                 # 生成默认演示数据
    python scripts/generate_demo_data.py --days 7        # 自定义天数
    python scripts/generate_demo_data.py --csv           # 额外导出 CSV
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotops import config  # noqa: E402
from robotops.console_io import setup_console_encoding  # noqa: E402
from robotops.models import RobotType, data_dictionary_frame  # noqa: E402
from robotops.report import autofit_columns  # noqa: E402

DEFAULT_SEED = 20260801
DEFAULT_START_DATE = date(2026, 8, 1)
DEFAULT_DAYS = 30

# 基础参数（用于生成“看起来合理”的运营数据）
BASE_FAULT_LAMBDA = 0.12  # 正常机器人日均故障次数期望
BASE_UPTIME_RATIO_MEAN = 0.93  # 正常机器人运行率期望
BASE_UPTIME_RATIO_STD = 0.045
SATISFACTION_STD = 2.4
SAVING_RATE_STD = 1.4

# 各类型机器人每日作业（巡检/配送/搬运）次数基准
INSPECTION_BASE_BY_TYPE: dict[str, int] = {
    RobotType.CLEANING.value: 8,
    RobotType.INSPECTION.value: 12,
    RobotType.SECURITY_PATROL.value: 24,
    RobotType.DELIVERY.value: 20,
    RobotType.DISINFECTION.value: 10,
    RobotType.AGV.value: 30,
}

# 项目蓝图：robot_overrides 中的键为机器人序号（如 "05"）
PROJECT_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "project_name": "华东-苏州工业园区清洁项目",
        "prefix": "SZ-CLN",
        "robot_type": RobotType.CLEANING.value,
        "robot_count": 4,
        "cost_base": 1180.0,
        "saving_base": 19.5,
        "satisfaction_base": 93.5,
        "note": "整体稳定；04 号机器人满意度偏低，累计故障率偏高",
        "robot_overrides": {
            "04": {"satisfaction_bias": -6.5, "fault_factor": 6.5, "cost_factor": 1.12},
        },
    },
    {
        "project_name": "华北-天津港智能巡检项目",
        "prefix": "TJ-INS",
        "robot_type": RobotType.INSPECTION.value,
        "robot_count": 5,
        "cost_base": 1560.0,
        "saving_base": 16.0,
        "satisfaction_base": 92.0,
        "note": "05 号机器人故障频发，用于演示“故障率偏高”（周期累计口径）",
        "robot_overrides": {
            "05": {"fault_factor": 12.0, "satisfaction_bias": -3.0, "cost_factor": 1.25},
        },
    },
    {
        "project_name": "华南-深圳前海安防项目",
        "prefix": "SZ-SEC",
        "robot_type": RobotType.SECURITY_PATROL.value,
        "robot_count": 3,
        "cost_base": 2050.0,
        "saving_base": 14.5,
        "satisfaction_base": 90.5,
        "note": "02 号运行率偏低、03 号故障偏多，演示“运行率偏低”",
        "robot_overrides": {
            "02": {"uptime_factor": 0.88, "satisfaction_bias": -4.0},
            "03": {"uptime_factor": 0.92, "fault_factor": 18.0},
        },
    },
    {
        "project_name": "西南-成都高新园区配送项目",
        "prefix": "CD-DLV",
        "robot_type": RobotType.DELIVERY.value,
        "robot_count": 4,
        "cost_base": 1320.0,
        "saving_base": 17.0,
        "satisfaction_base": 91.5,
        "note": "01 号满意度偏低，演示“满意度偏低”",
        "robot_overrides": {
            "01": {"satisfaction_bias": -5.0},
            "02": {"satisfaction_bias": -3.5, "fault_factor": 12.0},
        },
    },
    {
        "project_name": "华中-武汉光谷消杀项目",
        "prefix": "WH-DIS",
        "robot_type": RobotType.DISINFECTION.value,
        "robot_count": 3,
        "cost_base": 980.0,
        "saving_base": 12.0,
        "satisfaction_base": 89.5,
        "note": "整体节降率偏低，演示“节降率偏低”",
        "robot_overrides": {
            "01": {"saving_bias": -1.5},
            "02": {"saving_bias": -2.5},
            "03": {"saving_bias": -0.5},
        },
    },
    {
        "project_name": "华东-上海临港AGV搬运项目",
        "prefix": "SH-AGV",
        "robot_type": RobotType.AGV.value,
        "robot_count": 4,
        "cost_base": 2380.0,
        "saving_base": 21.0,
        "satisfaction_base": 94.0,
        "note": "正常对照组，未刻意制造异常",
        "robot_overrides": {},
    },
)

# 注入的脏数据说明（与 _inject_dirty_rows 一一对应）
DIRTY_ROW_NOTES: tuple[dict[str, str], ...] = (
    {
        "问题类型": "完全重复记录",
        "所在列": "整行",
        "注入内容": "复制一条正常记录，所有字段完全相同",
        "预期清洗动作": "删除完全重复行",
    },
    {
        "问题类型": "同一机器人同一天重复上报",
        "所在列": "日期 + 机器人ID",
        "注入内容": "重复某条记录，但用户满意度改为 77.0、故障次数改为 2",
        "预期清洗动作": "按「日期 + 机器人ID」去重，保留最后一条",
    },
    {
        "问题类型": "满意度缺失",
        "所在列": "用户满意度",
        "注入内容": "日期 {dirty_date} 增加一条记录，用户满意度留空",
        "预期清洗动作": "保留记录，不计入平均满意度",
    },
    {
        "问题类型": "满意度越界",
        "所在列": "用户满意度",
        "注入内容": "日期 {dirty_date} 写入 120（超出 0-100 有效区间）",
        "预期清洗动作": "置为空值并记录警告",
    },
    {
        "问题类型": "文本含多余空格",
        "所在列": "项目名称",
        "注入内容": "日期 {dirty_date} 在项目名称首尾各加一个空格",
        "预期清洗动作": "去除首尾空格后归并到原项目",
    },
    {
        "问题类型": "运营成本为负",
        "所在列": "运营成本",
        "注入内容": "日期 {dirty_date} 写入 -300.00",
        "预期清洗动作": "置为空值，不计入总运营成本",
    },
    {
        "问题类型": "运行时长缺失",
        "所在列": "运行时长",
        "注入内容": "日期 {dirty_date} 增加一条记录，运行时长留空",
        "预期清洗动作": "保留记录，运行率与平均运行时长跳过该条",
    },
)


def build_demo_dataframe(
    *,
    seed: int = DEFAULT_SEED,
    start_date: date = DEFAULT_START_DATE,
    days: int = DEFAULT_DAYS,
    inject_dirty: bool = True,
    dirty_date: date | None = None,
) -> pd.DataFrame:
    """构造演示数据 DataFrame（含可选的脏数据）。"""

    if days <= 0:
        raise ValueError("days 必须为正整数")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for blueprint in PROJECT_BLUEPRINTS:
        robot_type = str(blueprint["robot_type"])
        plan_hours = config.PLANNED_HOURS_BY_TYPE.get(robot_type, config.DEFAULT_PLANNED_HOURS)
        inspection_base = INSPECTION_BASE_BY_TYPE.get(robot_type, 8)
        overrides: dict[str, dict[str, float]] = blueprint["robot_overrides"]

        for sequence in range(1, int(blueprint["robot_count"]) + 1):
            robot_no = f"{sequence:02d}"
            robot_id = f"{blueprint['prefix']}-{robot_no}"
            robot_params = overrides.get(robot_no, {})
            fault_factor = float(robot_params.get("fault_factor", 1.0))
            uptime_factor = float(robot_params.get("uptime_factor", 1.0))
            satisfaction_bias = float(robot_params.get("satisfaction_bias", 0.0))
            saving_bias = float(robot_params.get("saving_bias", 0.0))
            cost_factor = float(robot_params.get("cost_factor", 1.0))

            for offset in range(days):
                current_date = start_date + timedelta(days=offset)
                uptime_ratio = float(
                    np.clip(
                        rng.normal(
                            BASE_UPTIME_RATIO_MEAN * uptime_factor, BASE_UPTIME_RATIO_STD
                        ),
                        0.35,
                        1.0,  # 实际运行时长不应超过计划运行时长
                    )
                )
                running_hours = round(plan_hours * uptime_ratio, 2)
                fault_count = int(rng.poisson(BASE_FAULT_LAMBDA * fault_factor))
                inspection_count = max(1, int(rng.poisson(inspection_base)))
                if fault_count >= 2:
                    repair_count = int(rng.random() < 0.55)
                elif fault_count == 1:
                    repair_count = int(rng.random() < 0.15)
                else:
                    repair_count = 0
                satisfaction = round(
                    float(
                        np.clip(
                            rng.normal(
                                float(blueprint["satisfaction_base"]) + satisfaction_bias,
                                SATISFACTION_STD,
                            ),
                            62.0,
                            99.5,
                        )
                    ),
                    1,
                )
                cost = round(
                    float(blueprint["cost_base"])
                    * cost_factor
                    * float(np.clip(rng.normal(1.0, 0.06), 0.75, 1.35)),
                    2,
                )
                saving_rate = round(
                    float(blueprint["saving_base"]) + saving_bias + float(rng.normal(0, SAVING_RATE_STD)),
                    2,
                )

                rows.append(
                    {
                        config.COL_DATE: current_date,
                        config.COL_PROJECT: blueprint["project_name"],
                        config.COL_ROBOT_ID: robot_id,
                        config.COL_ROBOT_TYPE: robot_type,
                        config.COL_RUNNING_HOURS: running_hours,
                        config.COL_PLANNED_HOURS: plan_hours,
                        config.COL_FAULT_COUNT: fault_count,
                        config.COL_INSPECTION_COUNT: inspection_count,
                        config.COL_REPAIR_COUNT: repair_count,
                        config.COL_SATISFACTION: satisfaction,
                        config.COL_COST: cost,
                        config.COL_SAVING_RATE: saving_rate,
                    }
                )

    frame = pd.DataFrame(rows, columns=list(config.ALL_COLUMNS))
    if inject_dirty:
        frame = _inject_dirty_rows(
            frame,
            dirty_date if dirty_date is not None else start_date + timedelta(days=days),
        )
    return frame


def _inject_dirty_rows(frame: pd.DataFrame, dirty_date: date) -> pd.DataFrame:
    """在数据末尾追加脏数据（顺序与 DIRTY_ROW_NOTES 一致）。

    设计要点：
    - 前 2 条与正常记录使用相同的「日期 + 机器人ID」，用于演示去重规则
      （第 1 条完全重复；第 2 条为同一机器人当天的重复上报）；
    - 后 5 条使用 ``dirty_date``（默认取数据区间之后的一天）且机器人各不相同，
      避免与正常记录冲突，只演示字段级清洗规则，不会覆盖原始数据。
    """

    template = frame.iloc[0].to_dict()
    directory = build_robot_directory().to_dict("records")

    def template_for(robot_id: str) -> dict[str, Any]:
        """取该机器人第一条真实记录作为模板，保证脏数据数值自然。"""

        matched = frame.loc[frame[config.COL_ROBOT_ID] == robot_id]
        return matched.iloc[0].to_dict() if len(matched) else dict(template)

    def dirty_row(entry: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        record = template_for(str(entry["机器人ID"]))
        record.update(
            {
                config.COL_DATE: dirty_date,
                config.COL_PROJECT: entry["项目名称"],
                config.COL_ROBOT_ID: entry["机器人ID"],
                config.COL_ROBOT_TYPE: entry["机器人类型"],
                config.COL_PLANNED_HOURS: entry["每日计划运行时长(小时)"],
            }
        )
        record.update(overrides)
        return record

    dirty_rows = [
        # 1. 与第一条正常记录完全相同
        dict(template),
        # 2. 同一机器人同一天重复上报，但内容不同
        {
            **template,
            config.COL_SATISFACTION: 77.0,
            config.COL_FAULT_COUNT: 2,
        },
        # 3. 用户满意度缺失
        dirty_row(directory[1], **{config.COL_SATISFACTION: None}),
        # 4. 用户满意度越界
        dirty_row(directory[2], **{config.COL_SATISFACTION: 120.0}),
        # 5. 项目名称含首尾空格
        dirty_row(directory[3], **{config.COL_PROJECT: f"  {directory[3]['项目名称']}  "}),
        # 6. 运营成本为负
        dirty_row(directory[4], **{config.COL_COST: -300.0}),
        # 7. 运行时长缺失
        dirty_row(directory[5], **{config.COL_RUNNING_HOURS: None}),
    ]
    return pd.concat([frame, pd.DataFrame(dirty_rows, columns=frame.columns)], ignore_index=True)


def build_robot_directory() -> pd.DataFrame:
    """项目与机器人清单（便于人工核对数据）。"""

    rows: list[dict[str, Any]] = []
    for blueprint in PROJECT_BLUEPRINTS:
        robot_type = str(blueprint["robot_type"])
        plan_hours = config.PLANNED_HOURS_BY_TYPE.get(robot_type, config.DEFAULT_PLANNED_HOURS)
        for sequence in range(1, int(blueprint["robot_count"]) + 1):
            rows.append(
                {
                    "项目名称": blueprint["project_name"],
                    "机器人ID": f"{blueprint['prefix']}-{sequence:02d}",
                    "机器人类型": robot_type,
                    "每日计划运行时长(小时)": plan_hours,
                    "项目说明": blueprint["note"],
                }
            )
    return pd.DataFrame(rows)


def generate_demo_file(
    output_path: str | Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    start_date: date = DEFAULT_START_DATE,
    days: int = DEFAULT_DAYS,
    inject_dirty: bool = True,
    also_csv: bool = False,
) -> list[Path]:
    """生成演示数据文件，返回已生成的文件路径列表。"""

    target = Path(output_path) if output_path is not None else config.DEMO_EXCEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    dirty_date = start_date + timedelta(days=days)  # 脏数据统一放在正常数据区间之后一天
    frame = build_demo_dataframe(
        seed=seed,
        start_date=start_date,
        days=days,
        inject_dirty=inject_dirty,
        dirty_date=dirty_date,
    )

    sheets: dict[str, pd.DataFrame] = {
        "运营数据": frame,
        "字段说明": data_dictionary_frame(),
        "项目与机器人": build_robot_directory(),
        "数据说明": pd.DataFrame(
            [
                {
                    **{
                        key: value.format(dirty_date=dirty_date.isoformat())
                        for key, value in note.items()
                    },
                    "序号": index + 1,
                }
                for index, note in enumerate(
                    DIRTY_ROW_NOTES if inject_dirty else ()
                )
            ],
            columns=["序号", "问题类型", "所在列", "注入内容", "预期清洗动作"],
        ),
    }

    written: list[Path] = []
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for sheet_name, sheet_frame in sheets.items():
            sheet_frame.to_excel(writer, sheet_name=sheet_name, index=False, na_rep="")
        autofit_columns(writer)
    written.append(target)

    if also_csv:
        csv_path = target.with_suffix(".csv")
        frame.to_csv(csv_path, index=False, encoding=config.CSV_ENCODING)
        written.append(csv_path)

    return written


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_demo_data.py",
        description="生成 RobotOps AI Phase 1 演示数据（robot_operation_demo.xlsx）",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help=f"输出文件路径，默认 {config.DEMO_EXCEL_FILE}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，默认 %(default)s")
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=DEFAULT_START_DATE,
        help="起始日期 YYYY-MM-DD，默认 %(default)s",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="生成天数，默认 %(default)s")
    parser.add_argument("--no-dirty", action="store_true", help="不注入脏数据")
    parser.add_argument("--csv", action="store_true", help="同时导出一份 CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console_encoding()
    args = build_parser().parse_args(argv)
    written = generate_demo_file(
        args.output_path,
        seed=args.seed,
        start_date=args.start_date,
        days=args.days,
        inject_dirty=not args.no_dirty,
        also_csv=args.csv,
    )

    frame = build_demo_dataframe(
        seed=args.seed,
        start_date=args.start_date,
        days=args.days,
        inject_dirty=not args.no_dirty,
        dirty_date=args.start_date + timedelta(days=args.days),
    )
    print("演示数据生成完成：")
    for path in written:
        print(f"  - {path}")
    print(
        f"  记录数：{len(frame)}（{len(PROJECT_BLUEPRINTS)} 个项目 / "
        f"{sum(int(item['robot_count']) for item in PROJECT_BLUEPRINTS)} 台机器人 / "
        f"{args.days} 天"
        + ("，另含 7 条脏数据" if not args.no_dirty else "")
        + "）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

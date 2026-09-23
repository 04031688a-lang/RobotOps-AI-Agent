"""RobotOps AI —— Phase 1 数据读取模块。

职责：
1. 解析并校验数据文件路径；
2. 按后缀读取 Excel（.xlsx/.xlsm/.xls）或 CSV；
3. CSV 自动尝试多种常见编码（utf-8-sig / gbk / utf-8）；
4. 规范表头（去空格、全角空格、BOM）并按别名映射为标准列名；
5. 校验必需列是否齐全，缺失时抛出带明确提示的异常。

本模块只负责“把数据读进来”，不做业务清洗（清洗在 data_cleaner 中完成）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .exceptions import DataLoadError, DataValidationError

CSV_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "gbk", "utf-8", "utf-16")


@dataclass
class LoadResult:
    """数据读取结果。"""

    data: pd.DataFrame
    path: Path
    sheet_name: str | None = None
    encoding: str | None = None
    renamed_columns: dict[str, str] = field(default_factory=dict)
    missing_optional_columns: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return int(len(self.data))

    @property
    def column_count(self) -> int:
        return int(self.data.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "数据文件": str(self.path),
            "工作表": self.sheet_name or "-",
            "文件编码": self.encoding or "-",
            "读取记录数": self.row_count,
            "读取列数": self.column_count,
            "表头重命名": self.renamed_columns,
            "缺少的可选列": self.missing_optional_columns,
        }


def list_excel_sheets(path: str | Path) -> list[str]:
    """列出 Excel 文件的所有工作表名称。"""

    resolved = _resolve_path(path)
    if resolved.suffix.lower() == ".xls":
        raise DataLoadError(
            "旧版 .xls 格式需要额外安装 xlrd，请将文件另存为 .xlsx 后再读取。"
        )
    try:
        with pd.ExcelFile(resolved, engine="openpyxl") as workbook:
            return list(workbook.sheet_names)
    except Exception as exc:  # pragma: no cover - 依赖具体文件状态
        raise DataLoadError(f"读取工作表列表失败：{resolved}（{exc}）") from exc


def load_operation_data(
    path: str | Path | None = None,
    *,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    validate: bool = True,
) -> LoadResult:
    """读取机器人运营数据文件。

    Args:
        path: 数据文件路径；为 ``None`` 时使用默认演示数据文件。
        sheet_name: Excel 工作表名称或索引，默认读取第一个工作表。
        encoding: CSV 编码，默认自动尝试常见编码。
        validate: 是否校验必需列。

    Returns:
        ``LoadResult``，其中 ``data`` 为原始 DataFrame（仅规范了表头）。
    """

    resolved = _resolve_path(path)
    suffix = resolved.suffix.lower()

    if suffix not in config.SUPPORTED_SUFFIXES:
        raise DataLoadError(
            f"不支持的文件类型：{suffix or '（无后缀）'}。"
            f"支持的格式：{'、'.join(config.SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".csv":
        frame, used_encoding = _read_csv(resolved, encoding=encoding)
        used_sheet = None
    else:
        frame, used_sheet = _read_excel(resolved, sheet_name=sheet_name)
        used_encoding = None

    if frame.empty:
        raise DataLoadError(f"数据文件没有可读取的记录：{resolved}")

    frame, renamed = normalize_headers(frame)
    missing_optional = [name for name in config.OPTIONAL_COLUMNS if name not in frame.columns]

    if validate:
        _ensure_required_columns(frame, resolved)

    return LoadResult(
        data=frame,
        path=resolved,
        sheet_name=used_sheet,
        encoding=used_encoding,
        renamed_columns=renamed,
        missing_optional_columns=missing_optional,
    )


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------
def _resolve_path(path: str | Path | None) -> Path:
    resolved = Path(path) if path is not None else config.DEFAULT_DATA_FILE
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = (config.PROJECT_ROOT / resolved).resolve()

    if not resolved.exists():
        hint = (
            f"未找到数据文件：{resolved}\n"
            f"提示：可先运行 `python scripts/generate_demo_data.py` 生成演示数据，"
            f"或通过 --data 参数指定已有数据文件。"
        )
        raise DataLoadError(hint)
    if resolved.is_dir():
        raise DataLoadError(f"路径指向的是目录而非数据文件：{resolved}")
    return resolved


def _read_excel(path: Path, *, sheet_name: str | int | None) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".xls":
        raise DataLoadError(
            f"旧版 .xls 格式未在 Phase 1 支持：{path.name}，请另存为 .xlsx 后再读取。"
        )
    try:
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            target = workbook.sheet_names[0] if sheet_name is None else sheet_name
            frame = pd.read_excel(workbook, sheet_name=target)
            used = target if isinstance(target, str) else workbook.sheet_names[target]
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"读取 Excel 失败：{path}（{exc}）") from exc
    return frame, str(used)


def _read_csv(path: Path, *, encoding: str | None) -> tuple[pd.DataFrame, str]:
    candidates = (encoding,) if encoding else CSV_ENCODING_CANDIDATES
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            frame = pd.read_csv(path, encoding=candidate)
            return frame, candidate
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            raise DataLoadError(f"读取 CSV 失败：{path}（{exc}）") from exc

    raise DataLoadError(
        f"CSV 编码识别失败：{path}，已尝试 {list(candidates)}。"
        f"请另存为 UTF-8 编码后重试。（{last_error}）"
    )


def normalize_headers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """规范表头并映射别名列名。"""

    frame = frame.copy(deep=True)
    cleaned: list[str] = []
    seen: dict[str, int] = {}

    for index, raw in enumerate(frame.columns):
        name = str(raw).replace("\ufeff", "").replace("\u3000", " ").strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"未命名列{index + 1}"
        name = config.COLUMN_ALIASES.get(name, name)
        name = config.COLUMN_ALIASES.get(name.lower(), name)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen[name] = seen.get(name, 0) + 1
        cleaned.append(name)

    renamed = {
        str(old): new for old, new in zip(frame.columns, cleaned) if str(old).strip() != new
    }
    frame.columns = cleaned
    return frame, renamed


def _ensure_required_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [name for name in config.REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DataValidationError(
            f"数据文件缺少必需列：{'、'.join(missing)}\n"
            f"文件：{path}\n"
            f"实际列名：{'、'.join(map(str, frame.columns))}\n"
            f"必需列清单：{'、'.join(config.REQUIRED_COLUMNS)}"
        )

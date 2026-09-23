"""RobotOps AI —— Phase 2 .env 配置读取。

要求：API Key 必须通过 .env 配置，禁止硬编码。

实现说明：
- 优先使用第三方库 ``python-dotenv``（若已安装）；
- 未安装时使用本模块内置的轻量解析器，保证「零新增依赖」也能读取 .env；
- 已存在的系统环境变量优先，不会被 .env 覆盖（与 python-dotenv 行为一致）；
- 支持 ``KEY=VALUE``、``export KEY=VALUE``、引号包裹的值、``#`` 注释、行尾注释。
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import config as project_config

ENV_FILE_NAME = ".env"
ENV_FILE_OVERRIDE_VAR = "ROBOTOPS_ENV_FILE"


def parse_env_text(text: str) -> dict[str, str]:
    """解析 .env 文本内容，返回键值对字典。"""

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_value(value.strip())
    return values


def _clean_value(value: str) -> str:
    """去除包裹引号，并处理未加引号时的行尾注释。"""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    if "#" in value:
        head, _, _ = value.partition("#")
        value = head.strip()
    return value


def find_env_file(env_file: str | Path | None = None) -> Path | None:
    """按优先级查找 .env 文件：显式路径 -> 环境变量 -> 项目根目录 -> 当前目录。"""

    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file))
    else:
        from_env = os.environ.get(ENV_FILE_OVERRIDE_VAR)
        if from_env:
            candidates.append(Path(from_env))
        candidates.append(project_config.PROJECT_ROOT / ENV_FILE_NAME)
        candidates.append(Path.cwd() / ENV_FILE_NAME)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        if resolved.is_file():
            return resolved
    return None


def load_env_file(
    env_file: str | Path | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """把 .env 中的键值写入 ``os.environ``，返回本次加载的键值对。

    Args:
        env_file: 指定 .env 路径；为 None 时按默认顺序查找。
        override: 为 True 时覆盖已存在的系统环境变量（默认不覆盖）。
    """

    path = find_env_file(env_file)
    if path is None:
        return {}

    loaded = _load_with_optional_dotenv(path, override=override)
    if loaded is not None:
        return loaded

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}

    values = parse_env_text(text)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _load_with_optional_dotenv(path: Path, *, override: bool) -> dict[str, str] | None:
    """若安装了 python-dotenv 则优先使用；未安装返回 None。"""

    try:
        from dotenv import dotenv_values, load_dotenv  # type: ignore
    except ImportError:
        return None

    load_dotenv(dotenv_path=path, override=override, encoding="utf-8-sig")
    raw = dotenv_values(path, encoding="utf-8-sig")
    return {key: value for key, value in raw.items() if value is not None}


def ensure_env_loaded(env_file: str | Path | None = None) -> Path | None:
    """加载 .env（幂等），返回实际使用的文件路径（未找到返回 None）。"""

    path = find_env_file(env_file)
    if path is not None:
        load_env_file(path)
    return path


def get_env(name: str, default: str | None = None) -> str | None:
    """读取环境变量，自动去除首尾空白，空字符串按未设置处理。"""

    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


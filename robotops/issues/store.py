"""RobotOps AI —— Phase 6 运营问题存储（SQLite，标准库 sqlite3）。

设计要点：
- 只使用 Python 标准库 ``sqlite3``，不引入任何外部数据库；
- 默认数据库文件：``data/issues.db``（已在 .gitignore 中忽略，可随时删除重建）；
- **不保存任何密钥**：本项目密钥统一放在 .env，问题表里只存业务字段；
- 表结构：``issues``（问题主表）+ ``issue_notes``（备注与处理记录）+ ``meta``（版本信息）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config as project_config
from .errors import IssueNotFoundError, IssueStorageError
from .models import Issue, IssueNote, IssueStatus

SCHEMA_VERSION = "phase6-issues-v1"
#: 环境变量可覆盖数据库位置（便于测试与多环境部署）
ENV_ISSUES_DB = "ROBOTOPS_ISSUES_DB"
DEFAULT_DB_PATH: Path = Path(
    os.environ.get(ENV_ISSUES_DB) or (project_config.DATA_DIR / "issues.db")
)

_JSON_FIELDS: tuple[str, ...] = (
    "metrics",
    "current_data",
    "diagnosis",
    "cases",
    "recommendations",
    "verification",
)

_COLUMNS: tuple[str, ...] = (
    "issue_id",
    "project",
    "title",
    "anomaly_type",
    "priority",
    "owner",
    "status",
    "created_at",
    "updated_at",
    "closed_at",
    "metrics",
    "current_data",
    "diagnosis",
    "cases",
    "recommendations",
    "resolution",
    "verification",
    "source_file",
)


class IssueStore:
    """运营问题的 SQLite 存储。"""

    def __init__(self, path: str | Path | None = None) -> None:
        # 未显式指定路径时，每次实例化都重新解析环境变量，保证多环境下行为一致
        if path is not None:
            self.path = Path(path)
        else:
            self.path = Path(os.environ.get(ENV_ISSUES_DB) or DEFAULT_DB_PATH)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=15, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_schema()
        except sqlite3.Error as error:  # pragma: no cover - 依赖文件系统状态
            raise IssueStorageError(f"打开问题数据库失败：{self.path}（{error}）") from error

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    def __enter__(self) -> "IssueStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS issues (
                issue_id        TEXT PRIMARY KEY,
                project         TEXT NOT NULL,
                title           TEXT NOT NULL,
                anomaly_type    TEXT NOT NULL,
                priority        TEXT NOT NULL,
                owner           TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                closed_at       TEXT NOT NULL DEFAULT '',
                metrics         TEXT NOT NULL DEFAULT '[]',
                current_data    TEXT NOT NULL DEFAULT '{}',
                diagnosis       TEXT NOT NULL DEFAULT '{}',
                cases           TEXT NOT NULL DEFAULT '[]',
                recommendations TEXT NOT NULL DEFAULT '[]',
                resolution      TEXT NOT NULL DEFAULT '',
                verification    TEXT NOT NULL DEFAULT '{}',
                source_file     TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS issue_notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id   TEXT NOT NULL,
                created_at TEXT NOT NULL,
                author     TEXT NOT NULL DEFAULT '',
                kind       TEXT NOT NULL DEFAULT '备注',
                text       TEXT NOT NULL,
                FOREIGN KEY (issue_id) REFERENCES issues(issue_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
            CREATE INDEX IF NOT EXISTS idx_issues_project ON issues(project);
            CREATE INDEX IF NOT EXISTS idx_notes_issue ON issue_notes(issue_id);

            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

    # -- 编号 -------------------------------------------------------------
    def next_issue_id(self, when: datetime | None = None) -> str:
        """生成问题编号：ISSUE-YYYYMM-NNN（按月自增）。"""

        stamp = (when or datetime.now()).strftime("%Y%m")
        prefix = f"ISSUE-{stamp}-"
        row = self._conn.execute(
            "SELECT issue_id FROM issues WHERE issue_id LIKE ? ORDER BY issue_id DESC LIMIT 1",
            (f"{prefix}%",),
        ).fetchone()
        sequence = 1
        if row is not None:
            try:
                sequence = int(str(row["issue_id"]).rsplit("-", 1)[-1]) + 1
            except ValueError:  # pragma: no cover - 编号被人为修改时忽略
                sequence = 1
        return f"{prefix}{sequence:03d}"

    # -- 写入 -------------------------------------------------------------
    def create_issue(self, issue: Issue) -> Issue:
        """新增问题（含首条备注）。"""

        payload = self._to_row(issue)
        try:
            self._conn.execute(
                f"INSERT INTO issues ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                tuple(payload[column] for column in _COLUMNS),
            )
            for note in issue.notes:
                self._insert_note(issue.issue_id, note)
            self._conn.commit()
        except sqlite3.IntegrityError as error:
            raise IssueStorageError(f"问题编号已存在：{issue.issue_id}（{error}）") from error
        except sqlite3.Error as error:  # pragma: no cover
            raise IssueStorageError(f"保存问题失败：{issue.issue_id}（{error}）") from error
        return self.get(issue.issue_id) or issue

    def update_fields(self, issue_id: str, **fields: Any) -> Issue:
        """更新问题字段（仅允许业务字段）。"""

        allowed = {
            "project",
            "title",
            "anomaly_type",
            "priority",
            "owner",
            "status",
            "updated_at",
            "closed_at",
            "metrics",
            "current_data",
            "diagnosis",
            "cases",
            "recommendations",
            "resolution",
            "verification",
            "source_file",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return self.get_required(issue_id)

        for key in _JSON_FIELDS:
            if key in updates and not isinstance(updates[key], str):
                updates[key] = json.dumps(updates[key], ensure_ascii=False)
        updates.setdefault("updated_at", _now())

        assignments = ", ".join(f"{key} = ?" for key in updates)
        try:
            cursor = self._conn.execute(
                f"UPDATE issues SET {assignments} WHERE issue_id = ?",
                (*updates.values(), issue_id),
            )
            self._conn.commit()
        except sqlite3.Error as error:  # pragma: no cover
            raise IssueStorageError(f"更新问题失败：{issue_id}（{error}）") from error
        if cursor.rowcount == 0:
            raise IssueNotFoundError(f"问题不存在：{issue_id}")
        return self.get_required(issue_id)

    def add_note(self, issue_id: str, note: IssueNote) -> Issue:
        """追加备注 / 处理记录。"""

        self.get_required(issue_id)
        try:
            self._insert_note(issue_id, note)
            self._conn.execute(
                "UPDATE issues SET updated_at = ? WHERE issue_id = ?", (_now(), issue_id)
            )
            self._conn.commit()
        except sqlite3.Error as error:  # pragma: no cover
            raise IssueStorageError(f"保存备注失败：{issue_id}（{error}）") from error
        return self.get_required(issue_id)

    def delete_issue(self, issue_id: str) -> bool:
        """删除问题（用于清理测试数据）。"""

        try:
            cursor = self._conn.execute("DELETE FROM issues WHERE issue_id = ?", (issue_id,))
            self._conn.execute("DELETE FROM issue_notes WHERE issue_id = ?", (issue_id,))
            self._conn.commit()
        except sqlite3.Error as error:  # pragma: no cover
            raise IssueStorageError(f"删除问题失败：{issue_id}（{error}）") from error
        return cursor.rowcount > 0

    # -- 读取 -------------------------------------------------------------
    def get(self, issue_id: str) -> Issue | None:
        row = self._conn.execute(
            "SELECT * FROM issues WHERE issue_id = ?", (issue_id,)
        ).fetchone()
        if row is None:
            return None
        notes = self._load_notes([issue_id]).get(issue_id, [])
        return self._to_issue(row, notes)

    def get_required(self, issue_id: str) -> Issue:
        issue = self.get(issue_id)
        if issue is None:
            raise IssueNotFoundError(f"问题不存在：{issue_id}")
        return issue

    def list_issues(
        self,
        *,
        status: str | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[Issue]:
        """查询问题列表（按创建时间倒序）。"""

        sql = "SELECT * FROM issues"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if project:
            clauses.append("project = ?")
            params.append(project)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, issue_id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(sql, tuple(params)).fetchall()
        notes_map = self._load_notes([str(row["issue_id"]) for row in rows])
        return [self._to_issue(row, notes_map.get(str(row["issue_id"]), [])) for row in rows]

    def count(self, *, status: str | None = None) -> int:
        if status:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM issues WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS total FROM issues").fetchone()
        return int(row["total"]) if row else 0

    def projects(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT project FROM issues ORDER BY project"
        ).fetchall()
        return [str(row["project"]) for row in rows]

    # -- 统计 -------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """问题统计：各状态数量、关闭率、平均处理周期。"""

        total = self.count()
        by_status = {status.value: self.count(status=status.value) for status in IssueStatus}
        closed = by_status[IssueStatus.CLOSED.value]
        close_rate = round(closed / total, 4) if total else 0.0

        rows = self._conn.execute(
            "SELECT created_at, closed_at FROM issues WHERE status = ? AND closed_at <> ''",
            (IssueStatus.CLOSED.value,),
        ).fetchall()
        durations: list[float] = []
        for row in rows:
            try:
                created = datetime.strptime(str(row["created_at"]), "%Y-%m-%d %H:%M:%S")
                closed_at = datetime.strptime(str(row["closed_at"]), "%Y-%m-%d %H:%M:%S")
            except ValueError:  # pragma: no cover - 兼容仅日期格式
                continue
            durations.append((closed_at - created).total_seconds() / 86400)

        average_days = round(sum(durations) / len(durations), 1) if durations else None
        return {
            "总问题数": total,
            IssueStatus.PENDING.value: by_status[IssueStatus.PENDING.value],
            IssueStatus.IN_PROGRESS.value: by_status[IssueStatus.IN_PROGRESS.value],
            IssueStatus.PENDING_VERIFICATION.value: by_status[IssueStatus.PENDING_VERIFICATION.value],
            IssueStatus.COMPLETED.value: by_status[IssueStatus.COMPLETED.value],
            IssueStatus.CLOSED.value: closed,
            "未关闭问题数": total - closed,
            "问题关闭率": close_rate,
            "平均处理周期(天)": average_days,
            "已关闭问题数(计入周期)": len(durations),
        }

    # -- 内部 -------------------------------------------------------------
    def _insert_note(self, issue_id: str, note: IssueNote) -> None:
        self._conn.execute(
            "INSERT INTO issue_notes (issue_id, created_at, author, kind, text) VALUES (?, ?, ?, ?, ?)",
            (issue_id, note.created_at, note.author, note.kind, note.text),
        )

    def _load_notes(self, issue_ids: Sequence[str]) -> dict[str, list[IssueNote]]:
        if not issue_ids:
            return {}
        placeholders = ", ".join("?" for _ in issue_ids)
        rows = self._conn.execute(
            f"SELECT * FROM issue_notes WHERE issue_id IN ({placeholders}) ORDER BY id",
            tuple(issue_ids),
        ).fetchall()
        notes: dict[str, list[IssueNote]] = {}
        for row in rows:
            notes.setdefault(str(row["issue_id"]), []).append(
                IssueNote(
                    created_at=str(row["created_at"]),
                    text=str(row["text"]),
                    author=str(row["author"]),
                    kind=str(row["kind"]),
                )
            )
        return notes

    def _to_row(self, issue: Issue) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "issue_id": issue.issue_id,
            "project": issue.project,
            "title": issue.title,
            "anomaly_type": issue.anomaly_type,
            "priority": issue.priority,
            "owner": issue.owner,
            "status": issue.status,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "closed_at": issue.closed_at,
            "resolution": issue.resolution,
            "source_file": issue.source_file,
        }
        for field_name in _JSON_FIELDS:
            payload[field_name] = json.dumps(
                getattr(issue, field_name), ensure_ascii=False, default=str
            )
        return payload

    @staticmethod
    def _to_issue(row: sqlite3.Row, notes: Iterable[IssueNote]) -> Issue:
        def load_json(name: str, default: Any) -> Any:
            raw = row[name]
            if raw in (None, "", b""):
                return default
            try:
                return json.loads(raw)
            except (TypeError, ValueError):  # pragma: no cover - 数据被外部修改
                return default

        return Issue(
            issue_id=str(row["issue_id"]),
            project=str(row["project"]),
            title=str(row["title"]),
            anomaly_type=str(row["anomaly_type"]),
            priority=str(row["priority"]),
            owner=str(row["owner"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            closed_at=str(row["closed_at"] or ""),
            metrics=load_json("metrics", []),
            current_data=load_json("current_data", {}),
            diagnosis=load_json("diagnosis", {}),
            cases=load_json("cases", []),
            recommendations=load_json("recommendations", []),
            resolution=str(row["resolution"] or ""),
            verification=load_json("verification", {}),
            source_file=str(row["source_file"] or ""),
            notes=list(notes),
        )


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

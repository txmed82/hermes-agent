"""Bounded, cursor-paged reads of a task's history (comments, runs, events).

A long-lived card accumulates hundreds of comments and runs; ``kanban_show`` must
orient a worker without dumping all of it. Every page reports the real DB total
and an exact ``before`` cursor (the oldest returned row id), so nothing is hidden
and any older row stays reachable. Pure reads: nothing here writes the board.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

import hermes_cli.kanban_db as _kb

# kind -> (table name, row class on the kanban_db facade). Paging is by rowid: ids are
# monotonic per insert, so a same-second burst is never split or skipped.
_HISTORY_TABLES = {
    "comments": ("task_comments", "Comment"),
    "runs": ("task_runs", "Run"),
    "events": ("task_events", "Event"),
}


@dataclass(frozen=True)
class HistoryPage:
    items: list        # oldest -> newest within the page
    total: int         # every row of this kind on the task, as stored
    before: Optional[int]   # cursor this page was read with (None = newest)
    limit: int
    next_before: Optional[int]  # pass as ``before`` to read the next-older page; None = no older rows
    older_count: int   # rows strictly older than this page

    def meta(self) -> dict[str, Any]:
        return {"total": self.total, "returned": len(self.items), "limit": self.limit,
                "before": self.before, "older_count": self.older_count,
                "has_more": self.older_count > 0, "next_before": self.next_before}


def page_history(
    conn: sqlite3.Connection, task_id: str, kind: str, *, limit: int,
    before: Optional[int] = None,
) -> HistoryPage:
    """The newest ``limit`` rows of ``kind`` with ``id < before`` (all rows when
    ``before`` is None), returned oldest-first like the unbounded lists."""
    table, cls_name = _HISTORY_TABLES[kind]
    row_cls = getattr(_kb, cls_name)
    total = history_total(conn, task_id, kind)
    where, params = "task_id = ?", list[Any]([task_id])
    if before is not None:
        where += " AND id < ?"
        params.append(int(before))
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE {where} ORDER BY id DESC LIMIT ?",
        (*params, int(limit))).fetchall() if limit > 0 else []
    items = [row_cls.from_row(r) for r in reversed(rows)]
    # Everything strictly below the page's oldest row (or below the cursor when the
    # page is empty, e.g. limit=0) is older history the caller has not seen.
    floor = items[0].id if items else before
    if floor is None:  # empty newest page: no rows at all, or limit=0
        older = total
        next_before = _max_id(conn, table, task_id) + 1 if total else None
    else:
        older = int(conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE task_id = ? AND id < ?",
            (task_id, int(floor))).fetchone()[0])
        next_before = int(floor) if older else None
    return HistoryPage(items=items, total=total, before=before, limit=limit,
                       next_before=next_before, older_count=older)


def _max_id(conn: sqlite3.Connection, table: str, task_id: str) -> int:
    return int(conn.execute(
        f"SELECT MAX(id) FROM {table} WHERE task_id = ?", (task_id,)).fetchone()[0])


def latest_comment_not_by(
    conn: sqlite3.Connection, task_id: str, author: Optional[str],
) -> Optional[Any]:
    """Newest comment whose author differs from ``author`` (one indexed query)."""
    row = conn.execute(
        "SELECT * FROM task_comments WHERE task_id = ? AND COALESCE(author, '') != COALESCE(?, '') "
        "ORDER BY id DESC LIMIT 1", (task_id, author)).fetchone()
    return _kb.Comment.from_row(row) if row is not None else None


def closed_runs_tail(conn: sqlite3.Connection, task_id: str, limit: int) -> tuple[list[Any], int]:
    """The newest ``limit`` ended runs (oldest-first) and the exact count of ended runs."""
    total = int(conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL",
        (task_id,)).fetchone()[0])
    rows = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC, id DESC LIMIT ?",  # list_runs order, reversed
        (task_id, int(limit))).fetchall() if limit > 0 else []
    return [_kb.Run.from_row(r) for r in reversed(rows)], total


def newest_completed_run(conn: sqlite3.Connection, task_id: str) -> Optional[Any]:
    """The run ``outcome='completed'`` with the latest ``started_at`` (lowest id on a tie,
    matching a stable newest-first sort of :func:`list_runs`), in one query."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
        "ORDER BY started_at DESC, id ASC LIMIT 1", (task_id,)).fetchone()
    return _kb.Run.from_row(row) if row is not None else None


def history_total(conn: sqlite3.Connection, task_id: str, kind: str) -> int:
    """Exact row count of ``kind`` on the task."""
    table = _HISTORY_TABLES[kind][0]
    return int(conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE task_id = ?", (task_id,)).fetchone()[0])


def comment_author_counts(conn: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    """Per-author scalars (count, newest id/time) for the whole thread, newest author first."""
    rows = conn.execute(
        "SELECT author, COUNT(*) AS n, MAX(id) AS latest_id, MAX(created_at) AS latest_at "
        "FROM task_comments WHERE task_id = ? GROUP BY author ORDER BY latest_id DESC",
        (task_id,)).fetchall()
    return [{"author": r["author"], "count": int(r["n"]), "latest_id": int(r["latest_id"]),
             "latest_at": r["latest_at"]} for r in rows]

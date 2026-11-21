"""Bounded, cursor-paged reads of a task's history (comments, runs, events).

A long-lived card accumulates hundreds of comments and runs; ``kanban_show`` must
orient a worker without dumping all of it. Every page reports the real DB total
and an exact ``before`` cursor (the oldest returned row id), so nothing is hidden
and any older row stays reachable. Pure reads: nothing here writes the board.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Optional

import hermes_cli.kanban_db as _kb

# table -> (row class name on the facade, id column). Paging is by rowid: ids are
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
    total = int(conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE task_id = ?", (task_id,)).fetchone()[0])
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


def latest_comment_where(
    conn: sqlite3.Connection, task_id: str, keep: Callable[[Any], bool],
) -> Optional[Any]:
    """Newest comment passing ``keep``, scanning newest-first in bounded batches."""
    before: Optional[int] = None
    while True:
        page = page_history(conn, task_id, "comments", limit=200, before=before)
        for comment in reversed(page.items):
            if keep(comment):
                return comment
        if page.next_before is None:
            return None
        before = page.next_before


def comment_author_counts(conn: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    """Per-author scalars (count, newest id/time) for the whole thread, newest author first."""
    rows = conn.execute(
        "SELECT author, COUNT(*) AS n, MAX(id) AS latest_id, MAX(created_at) AS latest_at "
        "FROM task_comments WHERE task_id = ? GROUP BY author ORDER BY latest_id DESC",
        (task_id,)).fetchall()
    return [{"author": r["author"], "count": int(r["n"]), "latest_id": int(r["latest_id"]),
             "latest_at": r["latest_at"]} for r in rows]

"""kanban_show orients a worker on a long-lived card without dumping its whole history.

A real research card reached ~540 comments / 35 runs and a bare ``kanban_show()``
returned ~1 MB. The contract: the bare call is bounded, every page states the real
DB totals with an exact cursor (nothing silently hidden, all older rows reachable),
and the body, dependencies, active owner and the newest non-assignee comment
survive however much self-commentary buried them. Synthetic fixture only.
"""
import json

import pytest

ASSIGNEE = "researcher"
STEER = "OPERATOR STEERING: write every artifact to verified external storage only."
BODY = "Opening post. " + "b" * 8000


@pytest.fixture
def long_card(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", ASSIGNEE)
    for key in ("HERMES_SESSION_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="upstream", assignee="planner")
        tid = kb.create_task(conn, title="long research", body=BODY, assignee=ASSIGNEE)
        kb.link_tasks(conn, parent, tid)
        for i in range(34):  # closed prior attempts
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, "
                "summary, metadata) VALUES (?, ?, 'released', ?, ?, 'reclaimed', ?, ?)",
                (tid, ASSIGNEE, 1_000 + i, 1_001 + i, f"attempt {i} " + "s" * 700,
                 json.dumps({"n": i})))
        kb.add_comment(conn, tid, "operator", "early note " + "e" * 3000)
        kb.add_comment(conn, tid, "operator", STEER)
        for i in range(300):  # transcript-sized self-commentary burying the steer
            kb.add_comment(conn, tid, ASSIGNEE, f"progress {i} " + "x" * 3000)
        conn.commit()
        # Dependency cleared, then the active run: the 35th run, owned by the assignee.
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
        assert kb.claim_task(conn, tid) is not None
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return {"tid": tid, "parent": parent, "run_id": run_id}


def _show(**args):
    import tools.kanban_tools  # registers the handlers
    from tools.registry import registry
    raw = registry.dispatch("kanban_show", args)
    return raw, json.loads(raw)


def _db_ids(table, tid):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return [r[0] for r in conn.execute(
            f"SELECT id FROM {table} WHERE task_id = ? ORDER BY id", (tid,))]


def test_bare_show_is_bounded_and_counts_reflect_the_db(long_card):
    raw, out = _show()
    assert len(raw.encode()) < 64 * 1024, len(raw.encode())
    comment_ids = _db_ids("task_comments", long_card["tid"])
    meta = out["history"]["comments"]
    assert meta["total"] == len(comment_ids) == 302
    assert meta["returned"] == len(out["comments"]) == 5
    assert meta["has_more"] is True
    assert meta["older_count"] == 297
    assert [c["id"] for c in out["comments"]] == comment_ids[-5:]
    assert meta["next_before"] == comment_ids[-5]
    runs = out["history"]["runs"]
    assert runs["total"] == 35 and runs["returned"] == 5 and runs["has_more"] is True
    assert out["runs"][-1]["id"] == long_card["run_id"]
    events = out["history"]["events"]
    assert events["total"] == len(_db_ids("task_events", long_card["tid"]))
    assert events["returned"] == len(out["events"]) <= 50


def test_cursor_pages_reconstruct_the_exact_thread(long_card):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    seen, before = [], None
    while True:
        args = {"comment_limit": 200, "worker_context": "none"}
        if before is not None:
            args["comment_before"] = before
        _, out = _show(**args)
        seen = out["comments"] + seen
        before = out["history"]["comments"]["next_before"]
        if before is None:
            assert out["history"]["comments"]["has_more"] is False
            break
    with kbc.connect_closing() as conn:
        stored = kb.list_comments(conn, long_card["tid"])
    assert [(c["id"], c["author"], c["body"]) for c in seen] == [
        (c.id, c.author, c.body) for c in stored]


def test_body_dependencies_owner_and_buried_steering_survive(long_card):
    _, out = _show()
    assert out["task"]["body"] == BODY
    assert out["task"]["title"] == "long research"
    assert out["parents"] == [long_card["parent"]]
    assert out["unsatisfied_parents"] == []
    owner = out["active_owner"]
    assert owner["assignee"] == ASSIGNEE and owner["status"] == "running" and owner["claimed"]
    assert owner["run"]["id"] == long_card["run_id"] and owner["run"]["profile"] == ASSIGNEE
    assert set(owner["run"]) == {"id", "profile", "status", "step_key", "started_at",
                                 "last_heartbeat_at", "max_runtime_seconds"}
    assert "latest_steering" not in out  # the author is unverified; the key claims no authority
    assert out["latest_non_assignee_comment"]["body"] == STEER
    assert out["latest_non_assignee_comment"]["author"] == "operator"
    assert STEER not in [c["body"] for c in out["comments"]]  # buried, yet retained above
    authors = {a["author"]: a["count"] for a in out["comment_authors"]}
    assert authors == {"operator": 2, ASSIGNEE: 300}
    assert out["worker_context_mode"] == "compact"
    ctx = out["worker_context"]
    assert STEER in ctx and BODY in ctx
    assert "302 comment(s) and 35 run(s) in total" in ctx
    assert "_(297 earlier comments omitted; showing most recent 5)_" in ctx
    assert "author unverified: operator or another worker" in ctx
    assert "operator steering" not in ctx


def test_compact_context_reads_only_the_newest_pages(long_card, monkeypatch):
    from hermes_cli import kanban_db as kb
    real_comments, real_runs = kb.list_comments, kb.list_runs

    def _guard(real):
        def call(conn, task_id, *a, **k):
            if task_id == long_card["tid"]:
                raise AssertionError("compact rendering must not load this card's whole history")
            return real(conn, task_id, *a, **k)
        return call

    monkeypatch.setattr(kb, "list_comments", _guard(real_comments))
    monkeypatch.setattr(kb, "list_runs", _guard(real_runs))
    _, out = _show()
    ctx = out["worker_context"]
    assert ctx.count("comment from worker `researcher`") == 5
    assert "_(31 earlier attempts omitted; showing most recent 3)_" in ctx
    assert "### Attempt 32 — reclaimed" in ctx and "### Attempt 34 — reclaimed" in ctx
    assert "### Attempt 31 " not in ctx


def test_huge_comment_bodies_are_capped_and_still_readable_in_full(long_card):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    huge = "H" * 300_000
    with kbc.connect_closing() as conn:
        cid = kb.add_comment(conn, long_card["tid"], "sibling-worker", huge)
        conn.commit()
    raw, out = _show()
    assert len(raw.encode()) < 64 * 1024, len(raw.encode())
    newest = out["comments"][-1]
    assert newest["id"] == cid and newest["body_truncated"] is True
    assert newest["body_chars"] == len(huge)
    assert newest["body"].startswith("H" * 4096)
    assert newest["body"].endswith(f"… [truncated, {len(huge) - 4096} chars omitted]")
    other = out["latest_non_assignee_comment"]
    assert other["id"] == cid and other["author"] == "sibling-worker" and other["body_truncated"]
    _, one = _show(comment_before=cid + 1, comment_limit=1,
                   comment_body_limit=newest["body_chars"], worker_context="none")
    assert one["comments"][0]["body"] == huge and "body_truncated" not in one["comments"][0]


def test_full_worker_context_is_the_unchanged_spawn_time_string(long_card):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    _, out = _show(worker_context="full", comment_limit=0, run_limit=0, event_limit=0)
    with kbc.connect_closing() as conn:
        spawn = kb.build_worker_context(conn, long_card["tid"])
    assert out["worker_context"] == spawn
    assert "History bounds" not in spawn
    assert out["comments"] == [] and out["history"]["comments"]["has_more"] is True
    assert out["history"]["comments"]["older_count"] == 302
    # An empty newest page still hands back a cursor that reaches the newest row.
    _, page = _show(comment_limit=1, comment_before=out["history"]["comments"]["next_before"],
                    worker_context="none")
    assert page["comments"][0]["id"] == _db_ids("task_comments", long_card["tid"])[-1]


@pytest.mark.parametrize("args, needle", [
    ({"comment_limit": -1}, "comment_limit must be >= 0"),
    ({"run_limit": 101}, "run_limit must be <= 100"),
    ({"event_before": "x"}, "event_before must be an integer"),
    ({"worker_context": "everything"}, "worker_context must be one of"),
    ({"comment_body_limit": 0}, "comment_body_limit must be >= 1"),
    ({"comment_limit": 2.5}, "comment_limit must be an integer"),
    ({"run_limit": True}, "run_limit must be an integer"),
])
def test_bad_bounds_are_rejected(long_card, args, needle):
    _, out = _show(**args)
    assert needle in out.get("error", ""), out

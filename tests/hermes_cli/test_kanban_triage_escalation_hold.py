"""Triage cards escalated for a human must not be re-promoted by automation.

Regression for the live loop seen on 2026-11-26: a worker blocked
``needs_input`` (a decision only the owner can make), the unblock-loop breaker
routed the card to ``triage`` with ``block_loop_detected``, and within a
minute the gateway's auto-decompose tick asked the aux LLM to re-specify it,
rewrote its body, promoted it to ``todo``, and the dispatcher respawned a
worker that could only block again. The escalation never stuck and every
round paid for another worker run.

The contract pinned here:

* A card whose newest triage entry is ``block_loop_detected`` is held from
  unattended automation (auto-decompose tick, ``--all`` sweeps) until a human
  answers: an ``unblock``, or a comment by someone other than the assignee.
* A comment answer promotes the card unchanged: no aux-LLM call, no rewrite.
* Unattended automation never rewrites a card that already has a body and a
  run history.
* Fresh triage cards are still decomposed as before, and an explicit
  ``decompose <id>`` by a human still works on any triage card.
"""

from __future__ import annotations

import argparse
import asyncio
import json as jsonlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec

ORIGINAL_BODY = "Decide whether to spend $40 on the probe run. Only the owner can approve this."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _llm_reply(payload: dict):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = jsonlib.dumps(payload)
    return resp


_REWRITE = {"fanout": False, "title": "Rewritten by the decomposer", "body": "A brand new body.",
            "assignee": "worker"}


def _patch_llm():
    return patch("agent.auxiliary_client.call_llm", return_value=_llm_reply(_REWRITE))


def _patch_profiles():
    fake = [SimpleNamespace(name="worker", is_default=True, description="w", description_auto=False,
                            model="m", provider="p", skill_count=1)]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x == "worker"),
        patch("hermes_cli.profiles.get_active_profile_name", return_value="worker"),
    ]


def _claim(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _escalated_task(conn, *, kind="needs_input") -> str:
    """Drive a card through block -> unblock -> re-block until the breaker sends it to triage."""
    tid = kb.create_task(conn, title="spend decision", body=ORIGINAL_BODY, assignee="worker")
    _claim(conn, tid)
    assert kb.block_task(conn, tid, reason="needs owner approval to spend", kind=kind)
    assert kb.unblock_task(conn, tid)
    _claim(conn, tid)
    assert kb.block_task(conn, tid, reason="still needs owner approval", kind=kind)
    task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
    return tid


def _dispatcher():
    from gateway import kanban_watchers_dispatcher as kwd

    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def _run_auto_tick(monkeypatch, per_tick: int = 5) -> int:
    from gateway import kanban_watchers_dispatcher as kwd
    from gateway.kanban_watchers_common import _to_thread_process_service

    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default"])
    patches = _patch_profiles()
    for p in patches:
        p.start()
    try:
        return asyncio.run(_to_thread_process_service(_dispatcher().auto_decompose_tick, per_tick))
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# The live loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["needs_input", "capability"])
def test_auto_decompose_tick_leaves_block_loop_escalation_in_triage(kanban_home, monkeypatch, kind):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn, kind=kind)

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch) == 0
        assert _run_auto_tick(monkeypatch) == 0

    llm.assert_not_called()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert (task.title, task.body) == ("spend decision", ORIGINAL_BODY)
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        assert not [c for c in kb.list_comments(conn, tid) if c.author == "auto-decomposer"]


def test_held_escalations_do_not_use_up_the_per_tick_budget(kanban_home, monkeypatch):
    """With a budget of one decompose per tick, held cards ahead of a fresh one must not starve it."""
    with kbc.connect_closing() as conn:
        held = [_escalated_task(conn) for _ in range(2)]
        fresh = kb.create_task(conn, title="fresh idea", triage=True)

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch, per_tick=1) == 1
    llm.assert_called_once()
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, fresh).status in {"todo", "ready"}
        assert all(kb.get_task(conn, tid).status == "triage" for tid in held)


def test_held_escalation_is_not_listed_for_automation(kanban_home):
    with kbc.connect_closing() as conn:
        held = _escalated_task(conn)
        fresh = kb.create_task(conn, title="fresh idea", triage=True)
    assert set(decomp.list_triage_ids()) == {held, fresh}
    assert decomp.list_triage_ids(automated=True) == [fresh]
    assert spec.list_triage_ids(automated=True) == [fresh]


def test_automated_decompose_reports_the_hold(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
    with _patch_llm() as llm:
        outcome = decomp.decompose_task(tid, author="auto-decomposer", automated=True)
    llm.assert_not_called()
    assert outcome.ok is False
    assert "block_loop" in outcome.reason or "unblock-loop" in outcome.reason
    assert "needs_input" in outcome.reason


def test_assignee_comment_after_escalation_does_not_release(kanban_home, monkeypatch):
    """A blocked worker that comments after blocking is not a human answering."""
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
        kb.add_comment(conn, tid, "worker", "Blocked: waiting on the owner's spend decision.")
        kb.add_comment(conn, tid, "auto-decomposer", "noise from automation")

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch) == 0
    llm.assert_not_called()
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_comment_before_escalation_does_not_release(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="spend decision", body=ORIGINAL_BODY, assignee="worker")
        kb.add_comment(conn, tid, "colin", "an old remark before the loop")
        _claim(conn, tid)
        kb.block_task(conn, tid, reason="r", kind="needs_input")
        kb.unblock_task(conn, tid)
        _claim(conn, tid)
        kb.block_task(conn, tid, reason="r", kind="needs_input")
        assert kb.triage_automation_policy(conn, tid)[0] == "hold"


def test_human_comment_releases_and_promotes_unchanged(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
        kb.add_comment(conn, tid, "colin", "Approved: spend up to $40 on the probe.")
        assert kb.triage_automation_policy(conn, tid)[0] == "promote"

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch) == 1
    llm.assert_not_called()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status in {"todo", "ready"}
        assert (task.title, task.body) == ("spend decision", ORIGINAL_BODY)


def test_human_unblock_releases_escalated_triage(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
        assert kb.unblock_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        # Loop memory survives the unblock: a same-cause re-block goes straight back to triage.
        assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
        _claim(conn, tid)
        kb.block_task(conn, tid, reason="still", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"


def test_unblock_refuses_a_fresh_triage_card(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="fresh idea", triage=True)
        assert kb.unblock_task(conn, tid) is False
        assert kb.get_task(conn, tid).status == "triage"


def test_explicit_human_decompose_still_works_on_escalated_card(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
    patches = _patch_profiles()
    for p in patches:
        p.start()
    try:
        with _patch_llm() as llm:
            outcome = decomp.decompose_task(tid, author="colin")
    finally:
        for p in patches:
            p.stop()
    llm.assert_called_once()
    assert outcome.ok is True


# ---------------------------------------------------------------------------
# Worked cards are never rewritten by automation
# ---------------------------------------------------------------------------


def test_automation_never_rewrites_a_worked_card_moved_to_triage(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="worked", body=ORIGINAL_BODY, assignee="worker")
        _claim(conn, tid)
        kb.complete_task(conn, tid, result="done once")
        # A direct move back to triage (dashboard drag) records a ``status`` event.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
            kb._append_event(conn, tid, "status", {"status": "triage"})
        action, reason = kb.triage_automation_policy(conn, tid)
        assert action == "hold"
        assert "run history" in reason

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch) == 0
    llm.assert_not_called()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert (task.status, task.body) == ("triage", ORIGINAL_BODY)


def test_fresh_triage_card_is_still_auto_decomposed(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="fresh idea", triage=True)
        assert kb.triage_automation_policy(conn, tid) == ("specify", "")

    with _patch_llm() as llm:
        assert _run_auto_tick(monkeypatch) == 1
    llm.assert_called_once()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status in {"todo", "ready"}
        assert task.body == "A brand new body."


# ---------------------------------------------------------------------------
# --all sweeps are unattended too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["decompose", "specify"])
def test_cli_all_sweep_skips_held_escalation(kanban_home, verb, capsys):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
    args = argparse.Namespace(task_id=None, all_triage=True, author=None, json=False, tenant=None)
    handler = kanban_cli._cmd_decompose if verb == "decompose" else kanban_cli._cmd_specify
    with _patch_llm() as llm:
        assert handler(args) == 0
    llm.assert_not_called()
    assert "No triage tasks" in capsys.readouterr().out
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert (task.status, task.body) == ("triage", ORIGINAL_BODY)


def test_automated_specify_promotes_answered_escalation_unchanged(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _escalated_task(conn)
        kb.add_comment(conn, tid, "dashboard", "Go ahead.")
    with _patch_llm() as llm:
        outcome = spec.specify_task(tid, author="auto-specifier", automated=True)
    llm.assert_not_called()
    assert outcome.ok is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status in {"todo", "ready"}
        assert task.body == ORIGINAL_BODY

"""A Desktop file card downloads from the profile that OWNS its session.

Desktop builds before the renderer owner fix paired a Bot Chat card's session id with the
window's ambient connection profile, so ``/api/fs/download`` looked the session up in the wrong
profile and answered 404 "Session not found". The download route now resolves the single
profile whose store holds that exact session id, for a principal that may already address every
profile. It never guesses between owners and never widens a profile-bound native grant.
"""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hermes_cli import web_server
from hermes_cli.web_routers import files as file_routes

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


@pytest.fixture
def owners(monkeypatch, tmp_path):
    """Profiles ``default``, ``seldinger`` and ``work`` on disk; sessions keyed by exact id."""
    from hermes_cli import profiles

    default_home = tmp_path / "default-home"
    homes = {"default": default_home, "seldinger": default_home / "profiles" / "seldinger",
             "work": default_home / "profiles" / "work"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    rows = {
        "bot-chat": {"seldinger"},
        "shared-id": {"seldinger", "work"},
        "default-chat": {"default"},
    }
    for name, home in homes.items():  # Each profile's own session store, by exact id.
        with sqlite3.connect(home / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO sessions (id) VALUES (?)",
                             [(sid,) for sid, owned in rows.items() if name in owned])
        conn.close()
    lookups = []

    async def detail(session_id, profile=None):
        lookups.append((session_id, profile or "default"))
        if (profile or "default") not in {"default", "seldinger", "work"}:
            raise HTTPException(status_code=404, detail=f"Profile '{profile}' does not exist.")
        if (profile or "default") not in rows.get(session_id, ()):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"id": session_id, "cwd": str(tmp_path)}

    monkeypatch.setattr("hermes_cli.web_routers.sessions.get_session_detail", detail)
    report = tmp_path / "Applicant-Workbook.docx"
    report.write_bytes(b"workbook-bytes")
    return {"report": report, "lookups": lookups}


@pytest.mark.parametrize("profile", [None, "default", "work"])
def test_a_bot_chat_card_downloads_from_its_owning_profile(client, owners, profile):
    params = {"path": str(owners["report"]), "session_id": "bot-chat"}
    if profile is not None:
        params["profile"] = profile

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 200, response.text
    assert response.content == b"workbook-bytes"
    assert ("bot-chat", "seldinger") in owners["lookups"]


def test_the_requested_owner_is_used_without_searching(client, owners):
    params = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "seldinger"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 200, response.text
    assert owners["lookups"] == [("bot-chat", "seldinger")]


@pytest.mark.parametrize("session_id", ["missing-session", "shared-id", " "])
def test_an_unknown_or_ambiguous_session_still_refuses(client, owners, session_id):
    params = {"path": str(owners["report"]), "session_id": session_id, "profile": "default"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_previews_and_saves_keep_refusing_a_foreign_session(client, owners):
    note = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "default"}

    assert client.get("/api/fs/read-text", params=note).status_code == 404
    assert client.post("/api/fs/write-text", json={**note, "content": "x"}).status_code == 404


def test_a_profile_bound_native_grant_never_resolves_another_owner(owners):
    request = SimpleNamespace(state=SimpleNamespace(native_http_principal={"profile_id": "/default"}))

    with pytest.raises(HTTPException) as refused:
        asyncio.run(file_routes._fs_download_session(request, "bot-chat", "default"))

    assert refused.value.status_code == 404
    assert ("bot-chat", "seldinger") not in owners["lookups"]


def test_a_profile_without_a_session_store_is_not_a_candidate(client, owners, tmp_path):
    (tmp_path / "default-home" / "profiles" / "work" / "state.db").unlink()
    params = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "default"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 200, response.text
    assert owners["lookups"] == [("bot-chat", "default"), ("bot-chat", "seldinger")]


def test_a_store_that_cannot_answer_fails_closed(client, owners, caplog, tmp_path):
    (tmp_path / "default-home" / "profiles" / "work" / "state.db").write_bytes(b"not a database" * 512)
    params = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "default"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert "owner search stopped; profile 'work' store failed: DatabaseError" in caplog.text
    assert ("bot-chat", "seldinger") not in owners["lookups"]


def test_an_unknown_profile_name_is_not_reinterpreted(client, owners):
    params = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "missing"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 404
    assert ("bot-chat", "seldinger") not in owners["lookups"]


def test_the_search_never_opens_a_store_through_the_browse_path(client, owners):
    params = {"path": str(owners["report"]), "session_id": "bot-chat", "profile": "work"}

    response = client.get("/api/fs/download", params=params)

    assert response.status_code == 200, response.text
    # Only the requested profile and the one owner go through the session-detail read.
    assert owners["lookups"] == [("bot-chat", "work"), ("bot-chat", "seldinger")]

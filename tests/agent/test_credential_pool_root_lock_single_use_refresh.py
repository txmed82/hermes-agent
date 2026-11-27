"""Regression: a single-use OAuth refresh in a profile must hold BOTH the profile
auth.json lock and the global-root auth.json lock, profile-first.

Before the fix only the profile lock was taken, so the default profile (root lock)
and a named profile (profile lock) could both sync + POST the same single-use
refresh token; the loser got ``refresh_token_reused`` and the login died.
"""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Root store at ``<tmp>/hermes-root`` with one named profile beneath it."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    fakehome = tmp_path / "fakehome"
    fakehome.mkdir()
    monkeypatch.setenv("HOME", str(fakehome))

    expired = int((time.time() - 3600) * 1000)
    row = {
        "id": "x1", "label": "xai", "auth_type": "oauth", "priority": 0,
        "source": "manual:device_code", "access_token": "at-old",
        "refresh_token": "rt-old", "expires_at_ms": expired,
    }
    (root / "auth.json").write_text(json.dumps(
        {"version": 1, "providers": {}, "credential_pool": {"xai-oauth": [dict(row)]}}))
    profile = root / "profiles" / "kid"
    profile.mkdir(parents=True)
    (profile / "auth.json").write_text(json.dumps(
        {"version": 1, "providers": {}, "credential_pool": {"xai-oauth": [dict(row)]}}))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return root, profile


def _refresh_with_recorded_locks(monkeypatch, pool, entry):
    from agent import credential_pool as cp

    calls: list = []
    real_lock = cp._auth_store_lock

    def recording_lock(*args, **kwargs):
        calls.append(kwargs.get("target_path"))
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(cp, "_auth_store_lock", recording_lock)

    # Record which locks are held at the moment the refresh POST would happen; the POST itself
    # is out of scope here.
    held_at_post: list = []

    def fake_impl(synced, *, force, failed_key=None):
        held_at_post.extend(calls)
        return synced

    monkeypatch.setattr(pool, "_refresh_entry_impl", fake_impl)
    updated = pool._refresh_entry(entry, force=True)
    return held_at_post, updated


def test_profile_single_use_refresh_also_holds_root_lock(profile_home, monkeypatch):
    from agent.credential_pool import load_pool

    _root, _profile_dir = profile_home
    pool = load_pool("xai-oauth")
    entry = next(e for e in pool.entries() if e.id == "x1")

    calls, updated = _refresh_with_recorded_locks(monkeypatch, pool, entry)
    assert updated is not None

    target_paths = [None if c is None else str(c) for c in calls]
    root_locks = [p for p in target_paths if p and p.endswith("hermes-root/auth.json")]
    profile_locks = [p for p in target_paths if p is None]  # default = active profile store
    assert root_locks, "the global-root auth.json lock was never taken during the profile refresh"
    assert profile_locks, "the profile auth.json lock was never taken"
    # Ordering invariant: profile lock outer (first), root lock inner.
    assert target_paths.index(None) < min(
        i for i, p in enumerate(target_paths) if p and p.endswith("hermes-root/auth.json"))


def test_root_lock_helper_is_nullcontext_in_classic_mode(profile_home, monkeypatch):
    from agent.credential_pool import CredentialPool
    import contextlib

    # Classic mode: HERMES_HOME is the root itself -> no global fallback path.
    root = profile_home[0]
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    pool = CredentialPool("xai-oauth", [])
    ctx = pool._root_store_lock_if_borrowing()
    assert isinstance(ctx, contextlib.nullcontext)

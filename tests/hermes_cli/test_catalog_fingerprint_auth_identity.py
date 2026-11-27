"""Catalog cache keys must survive token refreshes and pool status writes in auth.json."""
import json

from hermes_cli import models as m


def _write(path, pool, extra=None):
    data = {"credential_pool": {"antigravity": pool}, "providers": {}}
    data.update(extra or {})
    path.write_text(json.dumps(data))


def test_token_and_status_writes_do_not_change_identity(tmp_path):
    p = tmp_path / "auth.json"
    _write(p, [{"id": "a1", "source": "env:X", "auth_type": "api_key", "access_token": "t1", "last_status": "ok"}])
    before = m._auth_store_identity(p, "antigravity")
    _write(p, [{"id": "a1", "source": "env:X", "auth_type": "api_key", "access_token": "t2",
                "last_status": "exhausted", "model_cooldowns": {"x": 1}}],
           {"credential_pool": {"antigravity": [{"id": "a1", "source": "env:X", "auth_type": "api_key",
                                                 "access_token": "t2", "last_status": "exhausted"}],
                                "anthropic": [{"id": "z", "access_token": "other"}]}})
    assert m._auth_store_identity(p, "antigravity") == before


def test_account_change_changes_identity(tmp_path):
    p = tmp_path / "auth.json"
    _write(p, [{"id": "a1", "source": "env:X"}])
    before = m._auth_store_identity(p, "antigravity")
    _write(p, [{"id": "a2", "source": "manual"}])
    assert m._auth_store_identity(p, "antigravity") != before
    assert m._auth_store_identity(tmp_path / "nope.json", "antigravity") == "missing"

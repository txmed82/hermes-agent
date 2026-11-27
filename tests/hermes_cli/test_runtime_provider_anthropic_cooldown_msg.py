"""A rate-limited Anthropic subscription login must not be reported as a missing API key."""
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp


class _Pool:
    def __init__(self, statuses, reset_at):
        self._entries = [SimpleNamespace(last_status=s) for s in statuses]
        self._reset_at = reset_at

    def entries(self):
        return list(self._entries)

    def next_available_at(self, *, model=None):
        return self._reset_at


def _patch(monkeypatch, pool):
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda **_: None)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _p: pool)


def test_exhausted_subscription_reports_usage_limit_not_missing_key(monkeypatch):
    _patch(monkeypatch, _Pool(["exhausted", "dead"], 1_900_000_000.0))
    with pytest.raises(rp.AuthError) as exc:
        rp._anthropic_token_or_raise(model="claude-opus-5-5")
    msg = str(exc.value)
    assert "usage limit" in msg and "no API key is needed" in msg
    assert "ANTHROPIC_API_KEY" not in msg


def test_all_dead_or_empty_pool_keeps_missing_credentials_message(monkeypatch):
    _patch(monkeypatch, _Pool(["dead"], None))
    with pytest.raises(rp.AuthError) as exc:
        rp._anthropic_token_or_raise(model="claude-opus-5-5")
    assert str(exc.value) == rp._NO_ANTHROPIC_CREDENTIALS_MSG

    _patch(monkeypatch, _Pool([], None))
    with pytest.raises(rp.AuthError) as exc:
        rp._anthropic_token_or_raise()
    assert str(exc.value) == rp._NO_ANTHROPIC_CREDENTIALS_MSG

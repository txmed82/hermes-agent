"""A chat switched to another provider must resume on THAT provider's endpoint.

The messaging gateway's /model wrote only ``model``/``provider`` over a row the Desktop had persisted
with the Nous Portal route, so resuming a chat switched to openai-codex sent the Codex slug to the
Portal's chat/completions ("Model 'gpt-6-luna-900k' isn't available on ChatGPT or Codex").
"""

import json
from types import SimpleNamespace

import pytest

from hermes_cli.cli_model_switch_mixin import stored_session_route
from hermes_state import SessionDB
from tui_gateway.server import _stored_session_runtime_overrides

NOUS_ROUTE = {"base_url": "https://inference-api.nousresearch.com/v1", "api_mode": "chat_completions"}


@pytest.mark.parametrize("row_origin", ["gateway_model_switch", "row_written_by_older_build"])
def test_resume_drops_previous_providers_endpoint(tmp_path, row_origin):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="telegram", model="openai/gpt-6-luna")
    if row_origin == "gateway_model_switch":
        db.update_session_meta("s1", json.dumps({"model": "openai/gpt-6-luna", "provider": "nous", **NOUS_ROUTE}),
                               "openai/gpt-6-luna")
        db.update_session_model("s1", "gpt-6-luna-900k", provider="openai-codex")
    else:
        db.update_session_meta("s1", json.dumps({"model": "gpt-6-luna-900k", "provider": "openai-codex", **NOUS_ROUTE}),
                               "gpt-6-luna-900k")
    row = db.get_session("s1")

    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, None)
    assert stored_session_route(row, current_model="openai/gpt-6-luna", current_provider="nous") == (
        "gpt-6-luna-900k", "openai-codex", None, None, True)


def test_resume_honors_gateway_runtime_over_first_call_billing_provider():
    """The messaging gateway persists the route it last ran only in ``model_config.gateway_runtime``;
    ``billing_provider`` is frozen at the first accounted call. Desktop resume must read the nested
    route like CLI ``--resume`` does, not recombine the current model with the stale bucket (#125942)."""
    row = {"model": "gpt-6-sol", "billing_provider": "anthropic", "model_config": json.dumps(
        {"gateway_runtime": {"provider": "openai-codex", "base_url": None, "api_mode": "codex_responses"}})}
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, "codex_responses")
    assert stored_session_route(row, current_model="gpt-6-sol", current_provider="anthropic")[1:4] == (
        "openai-codex", None, "codex_responses")


@pytest.mark.parametrize("provider,model,base_url,api_mode", [
    ("openai-codex", "gpt-6.1-sol", "https://chatgpt.com/backend-api/codex", "codex_responses"),
    ("anthropic", "claude-sonnet-4.6", "https://api.anthropic.com", "anthropic_messages"),
])
def test_desktop_switch_replaces_older_gateway_route_on_resume(tmp_path, provider, model, base_url, api_mode):
    """A later Desktop selection supersedes the gateway route stored before that selection."""
    from tui_gateway.server import _persist_live_session_runtime

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="desktop", model="z-ai/glm-5.3-flash")
        db.update_session_meta("s1", json.dumps({
            "provider": "nous", **NOUS_ROUTE,
            "gateway_runtime": {"provider": "nous", **NOUS_ROUTE, "fallback_active": False},
            "_branched_from": "parent-session",
        }))
        agent = SimpleNamespace(model=model, provider=provider, base_url=base_url,
                                api_mode=api_mode, reasoning_config=None, service_tier=None, _session_db=db)
        _persist_live_session_runtime({"agent": agent, "session_key": "s1"})
        row = db.get_session("s1")
        assert row is not None
        assert json.loads(row["model_config"])["_branched_from"] == "parent-session"
        desktop = _stored_session_runtime_overrides(row)["model_override"]
        assert (desktop["model"], desktop["provider"], desktop["base_url"], desktop["api_mode"]) == (
            model, provider, base_url, api_mode)
        assert stored_session_route(row, current_model="z-ai/glm-5.3-flash", current_provider="nous") == (
            model, provider, base_url, api_mode, True)
    finally:
        db.close()


def test_resume_prefers_gateway_runtime_over_stale_top_level_keys():
    """A gateway turn after a Desktop pick rewrites only ``gateway_runtime``, so the nested route is the
    newest one — same precedence as ``SessionDB.session_gateway_runtime``."""
    row = {"model": "gpt-6-sol", "model_config": json.dumps(
        {"provider": "anthropic", **NOUS_ROUTE, "gateway_runtime": {"provider": "openai-codex", "api_mode": "codex_responses"}})}
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, "codex_responses")

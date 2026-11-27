"""A per-turn provider fallback must not be saved as the chat's model."""
from types import SimpleNamespace

from tui_gateway.agent_factory import _runtime_model_config


def _agent(**kw):
    base = dict(model="grok-4.7", provider="xai-oauth", requested_provider="xai-oauth",
                base_url="https://api.x.ai/v1", api_mode="chat_completions",
                reasoning_config=None, service_tier=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fallback_turn_persists_primary_route():
    agent = _agent(_fallback_activated=True, reasoning_config={"effort": "low"}, _primary_runtime={
        "model": "claude-opus-5-5", "provider": "anthropic", "requested_provider": "anthropic",
        "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages",
        "reasoning_config": {"effort": "high"}})
    cfg = _runtime_model_config(agent, {})
    assert cfg["model"] == "claude-opus-5-5"
    assert cfg["provider"] == "anthropic"
    assert cfg["api_mode"] == "anthropic_messages"
    assert cfg["reasoning_config"] == {"effort": "high"}


def test_normal_turn_persists_live_route():
    cfg = _runtime_model_config(_agent(_fallback_activated=False), {})
    assert cfg["model"] == "grok-4.7" and cfg["provider"] == "xai-oauth"

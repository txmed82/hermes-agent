"""cron.fallback_providers gives pinned jobs a backup chain."""
from cron.scheduler import _job_fallback_chain


def test_pinned_job_uses_cron_fallback_chain():
    cfg = {"fallback_providers": [{"provider": "nous", "model": "z-ai/glm-5.3-flash"}],
           "cron": {"fallback_providers": [{"provider": "nous", "model": "deepseek/deepseek-v4-flash"}]}}
    chain = _job_fallback_chain({"id": "j", "provider": "anthropic", "model": "claude-opus-5-5"}, cfg)
    assert chain and chain[0]["model"] == "deepseek/deepseek-v4-flash"


def test_pinned_job_without_cron_chain_keeps_no_fallback():
    cfg = {"fallback_providers": [{"provider": "nous", "model": "z-ai/glm-5.3-flash"}]}
    assert not _job_fallback_chain({"id": "j", "provider": "anthropic", "model": "claude-opus-5-5"}, cfg)

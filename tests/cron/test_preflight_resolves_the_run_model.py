"""The provider-key preflight must judge the same model the run will use.

An unpinned job runs on ``cron.model`` (then the main ``model.default``). The preflight used to
pass only ``job.model``/``HERMES_MODEL`` to provider resolution, so it asked for a credential with
no model at all. A credential pool that benches one model (a per-model cooldown) cannot vouch for
an unknown model and returns nothing, and the preflight then reported "No Anthropic credentials
found" for a profile whose login works for the model the job actually runs on.
"""

from unittest.mock import patch

import pytest

from cron.scheduler_preflight import _preflight_check_provider_key, cron_job_model

JOB = {"id": "6522ca5511f1", "name": "researcher"}


@pytest.mark.parametrize(
    ("job", "cfg", "expected"),
    [
        ({}, {"cron": {"model": "claude-opus-5-5"}, "model": {"default": "gpt-6.1-sol"}}, "claude-opus-5-5"),
        ({}, {"cron": {}, "model": {"default": "claude-opus-5-5"}}, "claude-opus-5-5"),
        ({}, {"model": "claude-opus-5-5"}, "claude-opus-5-5"),
        ({"model": "pinned"}, {"cron": {"model": "claude-opus-5-5"}}, "pinned"),
        ({}, {}, ""),
    ],
)
def test_cron_job_model_follows_the_run_precedence(job, cfg, expected, monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    assert cron_job_model(job, cfg) == expected


def test_hermes_model_env_applies_only_when_config_names_no_model(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "env-model")
    assert cron_job_model({}, {}) == "env-model"
    assert cron_job_model({}, {"cron": {"model": "claude-opus-5-5"}}) == "claude-opus-5-5"


def test_preflight_resolves_the_provider_for_the_model_the_run_uses(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {"cron": {"model": "claude-opus-5-5", "model_provider": "anthropic"}, "model": {"default": "x"}}
    seen = {}

    def resolve(**kwargs):
        seen.update(kwargs)
        return {"provider": "anthropic"}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
        assert _preflight_check_provider_key(JOB, cfg) is None

    assert seen == {"requested": "anthropic", "target_model": "claude-opus-5-5"}


def test_preflight_passes_a_model_benched_credential_for_another_model(monkeypatch):
    """A pool whose only usable login is benched for a different model still serves the run's model."""
    from hermes_cli.auth import AuthError

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {"cron": {"model": "claude-opus-5-5", "model_provider": "anthropic"}}

    def resolve(*, requested, target_model, **_):
        if not target_model:
            raise AuthError("No Anthropic credentials found.")
        return {"provider": requested}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
        assert _preflight_check_provider_key(JOB, cfg) is None

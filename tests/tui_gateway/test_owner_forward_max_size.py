"""Regression: ``connect_owner`` must accept frames larger than websockets' 1 MiB
default. A ``session.resume`` answer carries the whole chat history; long Desktop
threads died with close code 1009 (message too big) after a gateway restart.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace


def test_connect_owner_default_max_size_is_64_mib():
    from tui_gateway.owner_forward import connect_owner

    default = inspect.signature(connect_owner).parameters["max_size"].default
    assert default == 64 * 2 ** 20
    assert default > 2 ** 20, "websockets' 1 MiB default breaks long session.resume answers"


def test_connect_owner_passes_the_large_max_size_to_the_dial(tmp_path, monkeypatch):
    from hermes_cli import gateway_client as gc
    from tui_gateway import owner_forward

    captured: dict = {}

    class _StubWS:
        subprotocol = gc.GATEWAY_WS_PROTOCOL

        async def recv(self):
            await asyncio.sleep(3600)

    class _StubCtx:
        async def __aenter__(self):
            return _StubWS()

        async def __aexit__(self, *exc):
            return False

    def fake_connect(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _StubCtx()

    import websockets.asyncio.client as wsc
    monkeypatch.setattr(wsc, "connect", fake_connect)
    monkeypatch.setattr(gc, "_session_ticket", lambda home, endpoint, *, scope=None: "tkt")

    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1")

    async def _run():
        async with owner_forward.connect_owner(tmp_path, endpoint) as client:
            assert client.websocket.subprotocol == gc.GATEWAY_WS_PROTOCOL

    asyncio.run(_run())
    assert captured["max_size"] == 64 * 2 ** 20
    assert captured["url"] == "ws://127.0.0.1:1/api/ws"

"""A profile stop must be able to interrupt a managed worker through ``request_hard_interrupt``.

Before the fix ``ManagedWorker.interrupt()`` took no message, so ``stop_authority_turns`` raised
TypeError mid-retire and the profile runtime stayed draining (all its chats and cron jobs failed
with ``runtime_draining``) until a gateway restart.
"""
from agent.interrupt_compat import request_hard_interrupt
from gateway.session_managed_worker import ManagedWorker


def test_request_hard_interrupt_with_message_stops_managed_worker():
    sent = []
    worker = ManagedWorker.__new__(ManagedWorker)
    worker.control = sent.append
    assert request_hard_interrupt(worker, "Profile stopping", tool_reason="gateway shutdown") is True
    assert sent == [{"type": "stop"}]


def test_plain_interrupt_still_stops():
    sent = []
    worker = ManagedWorker.__new__(ManagedWorker)
    worker.control = sent.append
    worker.interrupt()
    assert sent == [{"type": "stop"}]

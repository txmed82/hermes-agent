"""A profile stop that meets a managed worker through the adopted-agent table must stop it, not crash.

Live incident (2026-11-26 22:07): a backend restart unserved a profile while one of its turns had
adopted a ManagedWorker that was no longer in ``_managed_workers``. ``stop_authority_turns`` hard-
interrupted the adopted agent with a message and a tool reason; ManagedWorker only had a zero-
argument ``interrupt()``, so the TypeError aborted the unserve and left the profile draining (every
cron run refused with ``runtime_draining``) until the next manual restart eight hours later.

#15 fixed the signature. The same unwind still happens when the adopted worker has already closed
(``control`` raises ``managed_worker_lost``) or its control queue is full, or when any other agent's
interrupt raises: one turn must never abort the drain for the whole profile.
"""
import threading
from types import SimpleNamespace

from agent.interrupt_compat import request_hard_interrupt
from gateway.session_managed_worker import ManagedWorker


def _worker():
    return ManagedWorker(SimpleNamespace(pid=None, stdin=None, stdout=None))


def test_profile_stop_arguments_latch_the_managed_worker_stop():
    worker = _worker()
    assert request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown') is True
    assert worker.stop.is_set()
    assert worker.commands.get_nowait() == {'type': 'stop'}


def test_hard_interrupt_of_a_closed_worker_is_a_no_op():
    worker = _worker()
    worker.closed.set()
    request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown')
    assert not worker.stop.is_set()


def test_hard_interrupt_with_a_full_control_queue_keeps_the_latch():
    worker = _worker()
    while not worker.commands.full():
        worker.commands.put_nowait({'type': 'noop'})
    request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown')
    assert worker.stop.is_set(), 'the latch escalates the stop even when the queue is full'


def _authority(adopted, running=None):
    sessions = {sid: SimpleNamespace(event_stream=SimpleNamespace(execution={'execution_generation': 1}),
                                     route=f'route-{sid}') for sid in adopted}
    return SimpleNamespace(sessions=sessions, adopted={sid: (1, a) for sid, a in adopted.items()},
                           runner=SimpleNamespace(_running_agents=running or {}), _managed_workers={},
                           _managed_spawning={}, pending_stops={})


def test_stop_authority_turns_stops_an_adopted_worker_outside_the_worker_table():
    from gateway.run_runtime import stop_authority_turns
    worker = _worker()
    authority = _authority({'s': worker})
    assert stop_authority_turns(authority, in_process=True) == 1
    assert worker.stop.is_set()


def test_one_failing_agent_does_not_abort_the_profile_stop():
    from gateway.run_runtime import stop_authority_turns

    class Broken:
        def hard_interrupt(self, message=None, *, tool_reason=None):
            raise RuntimeError('boom')

    class Agent:
        def __init__(self):
            self.stopped = threading.Event()

        def hard_interrupt(self, message=None, *, tool_reason=None):
            self.stopped.set()

    ok = Agent()
    authority = _authority({'a': Broken(), 'b': ok})
    assert stop_authority_turns(authority, in_process=True) == 2
    assert ok.stopped.is_set(), 'the profile stop must reach every turn after one agent fails'

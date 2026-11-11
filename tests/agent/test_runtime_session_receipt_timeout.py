"""Receipt-backed timeout retry for the worker outbox.

The owner's ``apply_worker_receipt`` makes an exact (execution, sequence, digest)
retry return the stored receipt instead of re-running the mutation, so a raw
``TimeoutError('worker_receipt_timeout')`` after the owner committed is safe to
retry once with the identical journaled entry. These tests use the real
``hermes_state_runtime.mutate_worker_execution`` against a real SQLite
``SessionDB``: the first ``worker.persist`` commits and then loses its reply,
the store's bounded retry recovers the receipt.
"""
from __future__ import annotations

import json

import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB
from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError


def setup_receipt_owner(tmp_path, *, commit_then_timeout=True):
    """A real worker execution and an RPC that commits before losing its first reply."""
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = rt.begin_runtime_epoch(db, instance_id='boot')
    assignment = dict(execution_id='worker', session_id='s', generation=0)
    rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute',
                                 adoption_secret='private-fixture')
    scope = dict(session_id='s', execution_id='worker', generation=0, epoch=epoch)
    calls = []

    def rpc(method, **params):
        assert method == 'worker.persist'
        calls.append(json.loads(json.dumps(params)))
        result = rt.mutate_worker_execution(
            db, epoch=params['epoch'], execution_id=params['execution_id'],
            session_id=params['session_id'], generation=params['generation'],
            sequence=params['sequence'], operation=params['operation'], payload=params['payload'])
        if commit_then_timeout and len(calls) == 1:
            raise TimeoutError('worker_receipt_timeout')  # committed, reply lost
        return result

    def worker_rpc(method, **params):
        # WorkerRPC surfaces owner rejections as WorkerPersistenceError, never raw.
        try:
            return rpc(method, **params)
        except rt.RuntimeStoreError as exc:
            raise WorkerPersistenceError(str(exc)) from None

    return db, scope, worker_rpc, calls


def test_timeout_after_commit_recovers_the_stored_receipt(tmp_path):
    db, scope, rpc, calls = setup_receipt_owner(tmp_path)
    store = RuntimeSessionStore(rpc, scope, tmp_path / 'private')
    rows = [{'role': 'assistant', 'content': 'kept output'}]
    try:
        assert store.append_messages_batch('s', rows) == 1
        # Exactly two attempts: the lost reply, then the identical bounded retry.
        assert len(calls) == 2
        assert calls[0] == calls[1]
        assert calls[0]['operation'] == 'transcript.append'
        assert calls[0]['sequence'] == 1
        assert calls[0]['payload']['messages'] == [{'role': 'assistant', 'content': 'kept output'}]
        # The row annotation from the stored receipt landed on the caller's dicts.
        assert rows[0].get('_row_id'), 'receipt annotations must reach the caller'
        # Exactly one persisted message and one receipt per sequence; the journal drained.
        messages = db.get_messages('s')
        assert len(messages) == 1 and messages[0]['content'] == 'kept output'
        assert store.journal['pending'] == []
        assert json.loads(store.path.read_text())['pending'] == []
        assert store.failure is None
        assert store.flush_token_counts() is True
        # The next write proceeds normally on sequence 2.
        assert store.append_messages_batch('s', [{'role': 'assistant', 'content': 'next'}]) == 1
        assert len(db.get_messages('s')) == 2
        receipts = db._conn.execute(
            'SELECT sequence FROM worker_receipts WHERE execution_id=? ORDER BY sequence',
            (scope['execution_id'],)).fetchall()
        assert [r['sequence'] for r in receipts] == [1, 2]
    finally:
        store.close()
        db.close()


def test_timeout_retry_uses_a_new_worker_rpc_link(tmp_path, monkeypatch):
    from agent.runtime_session_store import WorkerRPC, _OwnerLink

    db, scope, owner_rpc, calls = setup_receipt_owner(tmp_path, commit_then_timeout=False)
    rpc = WorkerRPC(tmp_path)
    links = []

    class Link(_OwnerLink):
        def __init__(self):
            self.ws = self
            self.closed = False
            self.request = {}
            self.response = {}

        def send(self, request_id, message):
            self.request = json.loads(message)
            assert self.request['id'] == request_id
            result = owner_rpc(self.request['method'], **self.request['params'])
            self.response = {'id': request_id, 'result': result}

        def receive(self, timeout):
            assert timeout == 20
            if self is links[0]:
                raise TimeoutError('worker_receipt_timeout')
            return self.response

        def close(self):
            self.closed = True

    def connection():
        if rpc.link is None:
            rpc.link = Link()
            links.append(rpc.link)
        return rpc.link

    monkeypatch.setattr(rpc, '_connection', connection)
    store = RuntimeSessionStore(rpc, scope, tmp_path / 'private')
    try:
        assert store.append_messages_batch('s', [{'role': 'assistant', 'content': 'once'}]) == 1
        assert len(links) == 2 and links[0] is not links[1]
        assert links[0].closed and not links[1].closed
        assert links[0].request['id'] != links[1].request['id']
        assert len(calls) == 2 and calls[0] == calls[1]
        assert len(db.get_messages('s')) == 1
        assert store.journal['pending'] == [] and store.failure is None
    finally:
        if store.failure is None:
            store.close()
        else:
            store._outbox_owner.close()
        rpc.close()
        db.close()


def test_publish_timeout_replays_before_rotating_the_outbox_scope(tmp_path):
    db, scope, rpc, calls = setup_receipt_owner(tmp_path, commit_then_timeout=False)
    store = RuntimeSessionStore(rpc, scope, tmp_path / 'private')
    lost_reply = False

    def publication_rpc(method, **params):
        nonlocal lost_reply
        result = rpc(method, **params)
        if params['operation'] == 'compression.publish' and not lost_reply:
            lost_reply = True
            assert store.scope == scope  # owner moved; client has not received the receipt
            raise TimeoutError('worker_receipt_timeout')
        return result

    try:
        assert store.try_acquire_compression_lock('s', 'holder')
        store.rpc = publication_rpc
        rows = [{'role': 'assistant', 'content': 'retained summary'}]
        store.publish_compression_child(
            parent_session_id='s', child_session_id='child', source='test',
            messages=rows, compression_lock_holder='holder')
        publications = [c for c in calls if c['operation'] == 'compression.publish']
        assert len(publications) == 2 and publications[0] == publications[1]
        assert publications[0]['session_id'] == 's' and publications[0]['sequence'] == 2
        assert rows[0].get('_row_id')
        assert store.scope == dict(scope, session_id='child')
        persisted = json.loads(store.path.read_text(encoding='utf-8'))
        assert persisted['scope'] == store.scope and persisted['pending'] == []
        assert store.failure is None
        child_row = db._read_one('SELECT COUNT(*) FROM sessions WHERE id=?', ('child',))
        assert child_row is not None and child_row[0] == 1
        assert [m['content'] for m in db.get_messages('child')] == ['retained summary']
        receipt_row = db._read_one('SELECT COUNT(*) FROM worker_receipts')
        assert receipt_row is not None and receipt_row[0] == 2
        assert store.append_messages_batch('child', [{'role': 'user', 'content': 'followup'}]) == 1
        assert calls[-1]['session_id'] == 'child' and calls[-1]['sequence'] == 3
        assert [m['content'] for m in db.get_messages('child')] == ['retained summary', 'followup']
    finally:
        if store.failure is None:
            store.close()
        else:
            store._outbox_owner.close()
        db.close()


def test_repeated_timeouts_stay_bounded_and_recover_after_the_rpc_heals(tmp_path):
    db, scope, rpc, calls = setup_receipt_owner(tmp_path)
    flaky = [True]

    def flaky_rpc(method, **params):
        try:
            result = rpc(method, **params)  # first call raises; later calls replay the receipt
        except TimeoutError:
            raise
        if flaky[0]:
            flaky[0] = False  # heal only after this second, bounded attempt
            raise TimeoutError('worker_receipt_timeout')
        return result

    store = RuntimeSessionStore(flaky_rpc, scope, tmp_path / 'private')
    try:
        with pytest.raises(TimeoutError):
            store.append_messages_batch('s', [{'role': 'assistant', 'content': 'kept output'}])
        # Bounded: after the second timed-out attempt the failure is sticky.
        assert len(calls) == 2
        assert store.failure == 'worker_receipt_timeout'
        assert len(store.journal['pending']) == 1
        assert store.journal['pending'][0]['sequence'] == 1
        # No duplicate effects while the journal holds the pending entry.
        assert len(db.get_messages('s')) == 1
        # Writes still refuse until an explicit retry or adoption.
        with pytest.raises(WorkerPersistenceError, match='worker_receipt_timeout'):
            store.queue_token_counts('s', input_tokens=1)
        # The RPC healed: the explicit retry resolves the stored receipt, not a rerun.
        receipt = store.retry_pending()[0]
        assert receipt['count'] == 1
        assert len(calls) == 3 and calls[2] == calls[1] == calls[0]
        assert store.journal['pending'] == [] and store.failure is None
        assert len(db.get_messages('s')) == 1
    finally:
        store.close()
        db.close()


def test_non_timeout_rejection_is_not_retried_and_stays_sticky(tmp_path):
    db, scope, rpc, calls = setup_receipt_owner(tmp_path, commit_then_timeout=False)

    def rejecting_rpc(method, **params):
        result = rpc(method, **params)  # commits and returns the receipt
        if len(calls) == 1:
            raise WorkerPersistenceError('persistence_failed')  # first reply lost as an owner rejection
        return result

    store = RuntimeSessionStore(rejecting_rpc, scope, tmp_path / 'private')
    try:
        with pytest.raises(WorkerPersistenceError, match='persistence_failed'):
            store.append_messages_batch('s', [{'role': 'assistant', 'content': 'kept output'}])
        assert len(calls) == 1, 'a non-timeout rejection must not be retried'
        assert store.failure == 'persistence_failed'
        assert len(store.journal['pending']) == 1
        assert len(db.get_messages('s')) == 1  # the commit landed; the reply did not
        # The identical retry is still available through the explicit path.
        assert store.retry_pending()[0]['count'] == 1
        assert len(calls) == 2 and calls[1] == calls[0]
        assert store.journal['pending'] == [] and store.failure is None
    finally:
        store.close()
        db.close()


def test_retry_without_adoption_hits_the_epoch_fence_and_stays_fail_closed(tmp_path):
    """No epoch/generation bypass: the bounded retry resends the journaled scope, so a
    successor epoch that never adopted this execution rejects it and nothing replays."""
    db, scope, rpc, calls = setup_receipt_owner(tmp_path)
    flaky = [True]

    def flaky_rpc(method, **params):
        result = rpc(method, **params)
        if flaky[0]:
            flaky[0] = False
            raise TimeoutError('worker_receipt_timeout')
        # Successor epoch began, but this execution was never adopted into it.
        assert params['epoch'] == scope['epoch'], 'retry must not adopt a new epoch'
        return result

    store = RuntimeSessionStore(flaky_rpc, scope, tmp_path / 'private')
    try:
        with pytest.raises(TimeoutError):
            store.append_messages_batch('s', [{'role': 'assistant', 'content': 'kept output'}])
        assert len(calls) == 2 and len(db.get_messages('s')) == 1
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=new_epoch)
        with pytest.raises(WorkerPersistenceError, match='stale_epoch'):
            store.retry_pending()
        # Fail closed: pending preserved, no duplicate, no replay under the new epoch.
        assert len(calls) == 3
        assert calls[2]['epoch'] == scope['epoch'] and calls[2] == calls[1]
        assert len(store.journal['pending']) == 1
        assert len(db.get_messages('s')) == 1
        assert store.failure == 'stale_epoch'
    finally:
        store._outbox_owner.close()
        db.close()


def test_conflicting_retry_payload_is_rejected_without_duplicate_effects(tmp_path):
    db, scope, rpc, calls = setup_receipt_owner(tmp_path)
    flaky = [True]

    def flaky_rpc(method, **params):
        result = rpc(method, **params)  # first call commits; the retry replays the receipt
        if flaky[0]:
            flaky[0] = False  # time out once more so the bounded retry goes sticky
            raise TimeoutError('worker_receipt_timeout')
        return result

    store = RuntimeSessionStore(flaky_rpc, scope, tmp_path / 'private')
    try:
        with pytest.raises(TimeoutError):
            store.append_messages_batch('s', [{'role': 'assistant', 'content': 'kept output'}])
        assert len(calls) == 2 and len(db.get_messages('s')) == 1
        # Tamper with the journaled payload before the explicit retry: the owner's digest
        # fence must reject it, keeping the original effect exactly once.
        tampered = json.loads(json.dumps(store.journal))
        tampered['pending'][0]['payload']['messages'][0]['content'] = 'rewritten'
        store.journal = tampered
        with pytest.raises(WorkerPersistenceError, match='admission_conflict'):
            store.retry_pending()
        assert len(db.get_messages('s')) == 1
        assert db.get_messages('s')[0]['content'] == 'kept output'
        assert store.failure == 'admission_conflict'
        assert len(store.journal['pending']) == 1
        receipts = db._conn.execute(
            'SELECT sequence FROM worker_receipts WHERE execution_id=?', (scope['execution_id'],)
        ).fetchall()
        assert len(receipts) == 1
    finally:
        store._outbox_owner.close()
        db.close()

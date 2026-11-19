"""A store tree a live process executes from is set aside, never deleted.

Regression: a same-version interpreter repair renamed the live Python entry to ``.previous-*``,
published the replacement and deleted the old tree while the gateway still executed it. On macOS
``proc_pidpath`` then fails with ENOENT for the gateway and every worker it launched, tccd cannot
build the attribution chain, and the kernel denied Desktop access to the whole tree until a
restart. A rename keeps the executable nameable; only deletion breaks it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pm import install, process_holders
from pm.lock import Facts
from pm.store import Store
from tests.pm.live_executable import running_from


class _Package:
    name = "tool"

    def verify(self, entry, target):
        return None if (entry / "marker").is_file() else "marker is missing"


def _entry(path: Path, marker: bytes) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker").write_bytes(marker)
    return path


@pytest.mark.platforms("linux", "macos")
def test_holders_follow_a_rename_and_lose_an_unlinked_executable(tmp_path):
    import shutil

    tree = tmp_path / "tool-1"
    with running_from(tree) as process:
        assert process_holders.executable_holders(tree) == [process.pid]
        path, unlinked = process_holders.executable_state(process.pid)
        assert Path(path).parent.parent.resolve() == tree.resolve() and not unlinked

        aside = tmp_path / ".previous-tool-1"
        tree.rename(aside)
        assert process_holders.executable_holders(aside) == [process.pid]
        path, unlinked = process_holders.executable_state(process.pid)
        assert Path(path).parent.parent.resolve() == aside.resolve() and not unlinked

        shutil.rmtree(aside)
        # The state tccd trips over: alive, but its executable has no name left.
        assert process_holders.executable_state(process.pid)[1] is True
        assert process_holders.executable_holders(tmp_path) == []


@pytest.mark.platforms("linux", "macos")
def test_publish_keeps_the_replaced_tree_while_a_process_executes_it(tmp_path):
    store = Store(tmp_path / "store")
    entry = store.entry("tool-1")
    previous = store.entry(".previous-tool-1")
    with running_from(_entry(entry, b"old")) as process:
        with install._publish_entry(_Package(), store, _entry(tmp_path / "staged", b"new"),
                                    entry, previous, "target"):
            pass

        assert (entry / "marker").read_bytes() == b"new"
        assert not previous.exists()
        path, unlinked = process_holders.executable_state(process.pid)
        assert not unlinked, "the replaced interpreter was deleted under its running process"
        aside = Path(path).parent.parent
        assert aside.parent.resolve() == store.root.resolve()
        assert aside.name.startswith(".reclaim-") and (aside / "marker").read_bytes() == b"old"

        assert install._reclaim_set_aside(store) == 0
        assert aside.is_dir(), "reclaim deleted a tree its process still executes"
    assert install._reclaim_set_aside(store) == 1
    assert not list(store.root.glob(".reclaim-*"))


@pytest.mark.platforms("linux", "macos")
def test_gc_keeps_an_orphaned_entry_a_process_executes(tmp_path):
    from pm.cli import _gc_store

    store = Store(tmp_path / "store")
    facts = Facts(store.root / "facts.json")
    orphan = store.entry("tool-0.9")
    with running_from(_entry(orphan, b"old")):
        _gc_store(store, facts)
        assert orphan.is_dir(), "gc deleted an orphaned entry a running process executes"
    _gc_store(store, facts)
    assert not orphan.exists()


@pytest.mark.platforms("linux", "macos")
def test_discard_keeps_an_executing_tree_it_could_not_rename(tmp_path, monkeypatch):
    # Without the set-aside rename the race window is back, but deleting a tree a process
    # already executes from is the failure this whole path exists to prevent.
    store = Store(tmp_path / "store")
    tree = store.entry(".previous-tool-1")
    real_rename = Path.rename

    def refuse(self, target):
        if self == tree:
            raise PermissionError(13, "rename refused")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", refuse)
    with running_from(_entry(tree, b"old")) as process:
        install._discard_entry(store, ".previous-tool-1")
        assert (tree / "marker").read_bytes() == b"old"
        assert process_holders.executable_state(process.pid)[1] is False


def test_discard_keeps_a_tree_when_its_holders_cannot_be_read(tmp_path, monkeypatch):
    def unreadable(tree):
        raise OSError("process table unavailable")

    monkeypatch.setattr(process_holders, "executable_holders", unreadable)
    store = Store(tmp_path / "store")
    _entry(store.entry(".previous-tool-1"), b"old")

    install._discard_entry(store, ".previous-tool-1")
    [aside] = store.root.glob(".reclaim-*")
    assert (aside / "marker").read_bytes() == b"old"
    assert install._reclaim_set_aside(store) == 0 and aside.is_dir()


def test_discard_checks_holders_only_after_the_tree_left_its_name(tmp_path, monkeypatch):
    # A launcher can exec the entry's old name between a holder check and the delete; nothing
    # execs the fresh `.reclaim-*` name, so the check has to run on that one.
    checked = []
    monkeypatch.setattr(process_holders, "executable_holders",
                        lambda tree: checked.append((tree.name, tree.is_dir())) or [])
    store = Store(tmp_path / "store")
    _entry(store.entry(".previous-tool-1"), b"old")

    install._discard_entry(store, ".previous-tool-1")
    [(name, existed)] = checked
    assert name.startswith(".reclaim-") and existed
    assert not list(store.root.iterdir())


@pytest.mark.platforms("macos")
def test_darwin_status_reads_pbi_status():
    import signal
    import subprocess
    import sys
    import time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert process_holders._darwin_status(child.pid) == 2  # SRUN
        child.send_signal(signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while process_holders._darwin_status(child.pid) != 4:  # SSTOP; pbi_xstatus would be 17
            assert time.monotonic() < deadline, process_holders._darwin_status(child.pid)
            time.sleep(0.05)
        assert process_holders._darwin_running(child.pid)
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.mark.platforms("linux", "macos")
def test_an_exited_unreaped_child_is_not_an_unlinked_executable():
    import subprocess
    import sys
    import time

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # wait for exit without reaping: a zombie
            if process_holders.executable_state(child.pid) == (None, False):
                break
            time.sleep(0.05)
        assert process_holders.executable_state(child.pid) == (None, False)
    finally:
        child.wait(timeout=10)


def test_discard_still_removes_a_tree_nothing_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(process_holders, "executable_holders", lambda tree: [])
    store = Store(tmp_path / "store")
    _entry(store.entry(".previous-tool-1"), b"old")

    install._discard_entry(store, ".previous-tool-1")
    assert not list(store.root.iterdir())

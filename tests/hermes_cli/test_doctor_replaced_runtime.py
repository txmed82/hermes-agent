"""``hermes doctor`` flags a gateway that runs a replaced or deleted interpreter.

On macOS a gateway whose interpreter was deleted under it fails every privacy (TCC) decision for
itself and the workers it spawns, because tccd cannot name its executable; doctor is where an
operator finds out a restart is owed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from hermes_cli import doctor_replaced_runtime as doctor
from tests.pm.live_executable import running_from


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "tools"
    root.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(root))
    return root


def _gateway(monkeypatch, pid):
    import gateway.status

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda **kw: pid)
    monkeypatch.setattr(doctor, "_gateway_process_tree", lambda gateway_pid: [(gateway_pid, "gateway")])


@pytest.mark.platforms("linux", "macos")
def test_flags_a_gateway_whose_interpreter_was_deleted(store, tmp_path, monkeypatch, capsys):
    tree = tmp_path / "elsewhere" / "python-1"
    with running_from(tree) as process:
        shutil.rmtree(tree)
        _gateway(monkeypatch, process.pid)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    out = capsys.readouterr().out
    assert f"gateway PID {process.pid} runs an executable that was deleted" in out
    assert len(issues) == 1 and "hermes gateway restart" in issues[0]


@pytest.mark.platforms("linux", "macos")
def test_a_failed_holder_scan_keeps_the_gateway_verdict(store, tmp_path, monkeypatch, capsys):
    import pm.process_holders

    def unreadable(tree):
        raise OSError("proc_listpids failed")

    (store / ".previous-python-0").mkdir()
    tree = tmp_path / "elsewhere" / "python-1"
    with running_from(tree) as process:
        shutil.rmtree(tree)
        _gateway(monkeypatch, process.pid)
        monkeypatch.setattr(pm.process_holders, "executable_holders", unreadable)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    out = capsys.readouterr().out
    assert f"gateway PID {process.pid} runs an executable that was deleted" in out
    assert "Could not read the process table" in out
    assert len(issues) == 1 and "hermes gateway restart" in issues[0]


@pytest.mark.platforms("linux", "macos")
def test_flags_a_gateway_running_a_set_aside_toolchain(store, monkeypatch, capsys):
    with running_from(store / "python-1") as process:
        (store / "python-1").rename(store / ".previous-python-1")
        _gateway(monkeypatch, process.pid)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    out = capsys.readouterr().out
    assert f"gateway PID {process.pid} runs a replaced toolchain" in out
    assert ".previous-python-1" in out
    assert len(issues) == 1 and "hermes gateway restart" in issues[0]


@pytest.mark.platforms("macos")
def test_set_aside_tree_matches_a_differently_cased_spelling(store):
    # APFS is case-insensitive and libproc reports the launcher's spelling of the path.
    executable = str(store / ".previous-python-1" / "bin" / "python3").upper()
    assert doctor._set_aside_tree(executable, [store]) is not None
    assert doctor._set_aside_tree(str(store / "python-1" / "bin" / "python3").upper(), [store]) is None


@pytest.mark.platforms("linux", "macos")
def test_flags_a_gateway_launched_through_a_symlinked_store(store, tmp_path, monkeypatch, capsys):
    (tmp_path / "linked").symlink_to(store, target_is_directory=True)
    with running_from(tmp_path / "linked" / "python-1") as process:
        (store / "python-1").rename(store / ".previous-python-1")
        _gateway(monkeypatch, process.pid)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    assert f"gateway PID {process.pid} runs a replaced toolchain" in capsys.readouterr().out
    assert len(issues) == 1


@pytest.mark.platforms("linux", "macos")
def test_flags_other_processes_still_running_a_set_aside_toolchain(store, monkeypatch, capsys):
    import gateway.status

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda **kw: None)
    with running_from(store / ".reclaim-abc") as process:
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    out = capsys.readouterr().out
    assert f"PID {process.pid} still runs the replaced toolchain .reclaim-abc" in out
    assert "Gateway runs the current interpreter" not in out  # no gateway, no claim about one
    assert len(issues) == 1 and "replaced toolchain" in issues[0]


@pytest.mark.platforms("linux", "macos")
def test_a_healthy_gateway_is_reported_alongside_other_holders(store, monkeypatch, capsys):
    with running_from(store / "python-1") as gateway, running_from(store / ".reclaim-abc") as other:
        _gateway(monkeypatch, gateway.pid)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    out = capsys.readouterr().out
    assert "Gateway runs the current interpreter" in out
    assert f"PID {other.pid} still runs the replaced toolchain .reclaim-abc" in out
    assert len(issues) == 1 and "hermes gateway restart" not in issues[0]


@pytest.mark.platforms("linux", "macos")
def test_a_gateway_on_its_live_interpreter_is_ok(store, monkeypatch, capsys):
    with running_from(store / "python-1") as process:
        _gateway(monkeypatch, process.pid)
        issues: list[str] = []
        doctor.check_replaced_runtime(issues)

    assert "Gateway runs the current interpreter" in capsys.readouterr().out
    assert issues == []


def test_a_platform_that_cannot_inspect_processes_claims_nothing(store, monkeypatch, capsys):
    from pm import process_holders

    monkeypatch.setattr(process_holders, "_platform", lambda: None)
    _gateway(monkeypatch, 4242)
    issues: list[str] = []
    doctor.check_replaced_runtime(issues)
    assert capsys.readouterr().out == "" and issues == []


@pytest.mark.platforms("linux", "macos")
def test_gateway_process_tree_skips_exited_unreaped_children():
    import os
    import time

    import psutil

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "child never exited"
            time.sleep(0.05)
        assert child.pid not in dict(doctor._gateway_process_tree(os.getpid()))
    finally:
        child.wait(timeout=10)


def test_no_gateway_and_no_holders_prints_nothing(store, monkeypatch, capsys):
    import gateway.status

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda **kw: None)
    issues: list[str] = []
    doctor.check_replaced_runtime(issues)
    assert capsys.readouterr().out == "" and issues == []


@pytest.mark.platforms("linux", "macos")
def test_gateway_process_tree_reaches_spawned_workers():
    import os

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        tree = dict(doctor._gateway_process_tree(os.getpid()))
        assert tree[os.getpid()] == "gateway"
        assert tree[child.pid] == "gateway child"
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.mark.platforms("linux", "macos")
def test_launchers_are_ancestors_running_the_gateway_program_and_stop_at_anything_else():
    import psutil

    # sh -> python (launcher) -> python (gateway): the shell above them is not a launcher.
    script = ("import subprocess, sys; "
              "subprocess.run([sys.executable, '-c', 'import time; print(flush=True); time.sleep(60)'])")
    shell = subprocess.Popen(["/bin/sh", "-c", f'"{sys.executable}" -c "{script}"; :'],
                             stdout=subprocess.PIPE)
    try:
        shell.stdout.readline()
        [launcher] = psutil.Process(shell.pid).children()
        [gateway] = launcher.children()
        tree = dict(doctor._gateway_process_tree(gateway.pid))
        assert tree[launcher.pid] == "gateway launcher"
        assert shell.pid not in tree
    finally:
        for process in psutil.Process(shell.pid).children(recursive=True):
            process.kill()
        shell.kill()
        shell.wait(timeout=10)
        shell.stdout.close()

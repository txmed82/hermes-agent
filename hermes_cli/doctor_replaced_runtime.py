"""Doctor check: processes still running a replaced PM toolchain.

PM sets a replaced store tree aside (``.previous-*``, ``.reclaim-*``, ``.displaced-*``) while a
live process executes from it, and deletes it once nothing does (``pm.process_holders``). A process
in that state keeps the old interpreter until it restarts; one whose executable was deleted
outright (by an older PM, or by hand) is worse on macOS: tccd cannot name it, so every privacy
decision for its responsibility chain (Desktop, Documents, ...) is denied until a restart.

Checked: the gateway and its launchers for a deleted or set-aside executable, the processes the
gateway spawned for a set-aside one or a deleted one that lay in a store tree, and every process for
a set-aside store tree. Any other deleted executable, inside or outside the gateway's tree, is
usually some program's own self-update (browsers do this routinely) and says nothing about Hermes.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes_cli.doctor_report import _section, check_ok, check_warn

_SET_ASIDE_PREFIXES = (".previous-", ".reclaim-", ".displaced-")
_RESTART_HINT = "`hermes gateway restart`"


def _store_roots() -> list[Path]:
    from pm import paths

    roots = []
    for resolve in (paths.store_root, paths.writable_store_root):
        root = Path(os.path.realpath(resolve()))
        if root not in roots:
            roots.append(root)
    return roots


def _set_aside_tree(executable: str, roots: list[Path]) -> Path | None:
    """The set-aside store tree ``executable`` lies in, or None.

    Compared the way ``pm.process_holders`` compares (case-folded on macOS). ``executable`` is the
    kernel's name for the process executable, which already has symlinks resolved, as do
    ``roots``.
    """
    from pm.process_holders import fold_path

    folded = Path(fold_path(executable))
    for root in roots:
        try:
            depth = len(folded.relative_to(fold_path(str(root))).parts)
        except ValueError:
            continue
        parts = Path(executable).parts
        first = parts[len(parts) - depth] if depth else ""
        if fold_path(first).startswith(_SET_ASIDE_PREFIXES):
            return root / first
    return None


def _in_store(executable: str, roots: list[Path]) -> bool:
    from pm.process_holders import fold_path

    folded = Path(fold_path(executable))
    return any(folded.is_relative_to(fold_path(str(root))) for root in roots)


def replaced_runtime_reason(pid: int, roots: list[Path], *, any_deleted: bool = True) -> str | None:
    """Why ``pid`` runs a replaced interpreter, or None when it does not (or cannot be told).

    With ``any_deleted`` False a deleted executable counts only when it lay in a store tree, which
    only Linux can still tell (macOS has no name left for it).
    """
    from pm.process_holders import executable_state

    state = executable_state(pid)
    if state is None:
        return None
    executable, unlinked = state
    if unlinked:
        if any_deleted or executable is not None and _in_store(executable, roots):
            return "runs an executable that was deleted"
        return None
    if executable is not None and _set_aside_tree(executable, roots) is not None:
        return f"runs a replaced toolchain ({executable})"
    return None


def _gateway_process_tree(pid: int) -> list[tuple[int, str]]:
    """``(pid, role)`` for the gateway, its launcher ancestors and every process it spawned.

    Launchers are the unbroken chain of ancestors whose own command line is a gateway run (the
    supervising interpreter above a re-exec'd gateway), matched the way ``gateway.status``
    identifies a gateway. A shell, IDE or any other interpreter further up is not Hermes, and its
    own self-update says nothing about the gateway.
    """
    tree = [(pid, "gateway")]
    try:
        import psutil
    except ImportError:
        return tree
    from gateway.status import _read_process_cmdline, looks_like_gateway_runtime_command_line

    try:
        process = psutil.Process(pid)
        for parent in process.parents():
            if parent.pid <= 1 or not looks_like_gateway_runtime_command_line(
                    _read_process_cmdline(parent.pid)):
                break
            tree.append((parent.pid, "gateway launcher"))
        for child in process.children(recursive=True):
            try:
                if child.status() != psutil.STATUS_ZOMBIE:  # exited, unreaped: nothing to restart
                    tree.append((child.pid, "gateway child"))
            except psutil.Error:
                continue
    except psutil.Error:
        pass  # exited mid-scan: whatever was collected is still worth checking
    return tree


def _set_aside_holders(roots: list[Path]) -> list[tuple[int, Path]] | None:
    """``(pid, tree)`` for every process running a set-aside tree; None when the scan failed."""
    from pm.process_holders import executable_holders

    holders = []
    for root in roots:
        if not root.is_dir():
            continue
        for tree in sorted(root.iterdir()):
            if tree.name.startswith(_SET_ASIDE_PREFIXES) and tree.is_dir() and not tree.is_symlink():
                try:
                    holders += [(pid, tree) for pid in executable_holders(tree)]
                except OSError:
                    return None  # the gateway verdict above it still stands
    return holders


def check_replaced_runtime(manual_issues: list[str]) -> None:
    """Warn about the gateway tree, or any other process, running a replaced PM interpreter."""
    from gateway.status import get_running_pid
    from pm.process_holders import can_inspect

    roots = _store_roots()
    gateway_pid = get_running_pid(cleanup_stale=False)
    gateway_findings: dict[int, str] = {}
    if gateway_pid is not None:
        for pid, role in _gateway_process_tree(gateway_pid):
            reason = replaced_runtime_reason(pid, roots, any_deleted=role != "gateway child")
            if reason is not None:
                gateway_findings.setdefault(pid, f"{role} PID {pid} {reason}")
    set_aside = _set_aside_holders(roots)
    other_findings = {pid: f"PID {pid} still runs the replaced toolchain {tree.name}"
                      for pid, tree in set_aside or () if pid not in gateway_findings}
    if set_aside is not None and not other_findings and (
            gateway_pid is None or not gateway_findings and not can_inspect()):
        return  # nothing to report, or a platform that cannot tell what the gateway runs
    _section("Gateway Runtime")
    if gateway_pid is not None and not gateway_findings and can_inspect():
        check_ok("Gateway runs the current interpreter")
    for detail in (*gateway_findings.values(), *other_findings.values()):
        check_warn(detail)
    if set_aside is None:
        check_warn("Could not read the process table to check other processes for a replaced toolchain")
    if gateway_findings:
        manual_issues.append(
            f"Restart the gateway ({_RESTART_HINT}): it runs an interpreter that was replaced or "
            "deleted, and on macOS a deleted one fails every privacy check (Desktop, Documents) "
            "for the gateway and the workers it spawns")
    if other_findings:
        manual_issues.append(
            "Restart the processes listed under Gateway Runtime: they still run a replaced "
            "toolchain, which PM deletes once they exit")

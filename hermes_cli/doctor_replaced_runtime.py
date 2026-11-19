"""Doctor check: processes still running a replaced PM toolchain.

PM sets a replaced store tree aside (``.previous-*``, ``.reclaim-*``, ``.displaced-*``) while a
live process executes from it, and deletes it once nothing does (``pm.process_holders``). A process
in that state keeps the old interpreter until it restarts; one whose executable was deleted
outright (by an older PM, or by hand) is worse on macOS: tccd cannot name it, so every privacy
decision for its responsibility chain (Desktop, Documents, ...) is denied until a restart.

Checked: the gateway, its launchers and its children for a deleted or set-aside executable, and
every process for a set-aside store tree. A deleted executable is looked for only in the gateway's
tree: outside it, an unlinked executable is usually some other app's self-update (browsers do this
routinely) and says nothing about Hermes.
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

    Compared the way ``pm.process_holders`` compares (case-folded on macOS): libproc reports the
    launcher's spelling of the path, which need not match the store root's.
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


def replaced_runtime_reason(pid: int, roots: list[Path]) -> str | None:
    """Why ``pid`` runs a replaced interpreter, or None when it does not (or cannot be told)."""
    from pm.process_holders import executable_state

    state = executable_state(pid)
    if state is None:
        return None
    executable, unlinked = state
    if unlinked:
        return "runs an executable that was deleted"
    if executable is not None and _set_aside_tree(executable, roots) is not None:
        return f"runs a replaced toolchain ({executable})"
    return None


def _gateway_process_tree(pid: int) -> list[tuple[int, str]]:
    """``(pid, role)`` for the gateway, its launcher ancestors and every process it spawned."""
    tree = [(pid, "gateway")]
    try:
        import psutil
    except ImportError:
        return tree
    try:
        process = psutil.Process(pid)
        tree += [(parent.pid, "gateway launcher") for parent in process.parents() if parent.pid > 1]
        for child in process.children(recursive=True):
            try:
                if child.status() != psutil.STATUS_ZOMBIE:  # exited, unreaped: nothing to restart
                    tree.append((child.pid, "gateway child"))
            except psutil.Error:
                continue
    except psutil.Error:
        pass  # exited mid-scan: whatever was collected is still worth checking
    return tree


def _set_aside_holders(roots: list[Path]) -> list[tuple[int, Path]]:
    from pm.process_holders import executable_holders

    holders = []
    for root in roots:
        if not root.is_dir():
            continue
        for tree in sorted(root.iterdir()):
            if tree.name.startswith(_SET_ASIDE_PREFIXES) and tree.is_dir() and not tree.is_symlink():
                holders += [(pid, tree) for pid in executable_holders(tree)]
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
            reason = replaced_runtime_reason(pid, roots)
            if reason is not None:
                gateway_findings.setdefault(pid, f"{role} PID {pid} {reason}")
    other_findings = {pid: f"PID {pid} still runs the replaced toolchain {tree.name}"
                      for pid, tree in _set_aside_holders(roots) if pid not in gateway_findings}
    if not other_findings and (gateway_pid is None or not gateway_findings and not can_inspect()):
        return  # nothing to report, or a platform that cannot tell what the gateway runs
    _section("Gateway Runtime")
    if not gateway_findings and not other_findings:
        return check_ok("Gateway runs the current interpreter")
    for detail in (*gateway_findings.values(), *other_findings.values()):
        check_warn(detail)
    if gateway_findings:
        manual_issues.append(
            f"Restart the gateway ({_RESTART_HINT}): it runs an interpreter that was replaced or "
            "deleted, and on macOS a deleted one fails every privacy check (Desktop, Documents) "
            "for the gateway and the workers it spawns")
    if other_findings:
        manual_issues.append(
            "Restart the processes listed under Gateway Runtime: they still run a replaced "
            "toolchain, which PM deletes once they exit")

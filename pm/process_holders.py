"""Which live processes execute a binary from a store tree.

POSIX lets an installer delete a tree whose interpreter a running process still executes: the
process keeps running on the unlinked inode. On macOS that breaks every later privacy (TCC)
decision for the process's whole responsibility chain. ``tccd`` resolves each requester through
``proc_pidpath``, which fails with ENOENT once the executable is unlinked ("Failed to create
attribution chain"), and the kernel then keeps denying, say, Desktop access to a gateway and every
worker it spawns until they restart. A rename keeps the executable resolvable (the kernel reports
the new name), so PM sets a replaced tree aside and deletes it only once nothing executes from it.

Stdlib only: PM runs before the rest of the tree exists.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys

_DARWIN_ALL_PIDS = 1  # PROC_ALL_PIDS
_DARWIN_PATH_MAX = 4096  # PROC_PIDPATHINFO_MAXSIZE
_LINUX_DELETED = " (deleted)"
_DARWIN_LIBPROC = None


def _darwin_libproc():
    global _DARWIN_LIBPROC
    if _DARWIN_LIBPROC is None:
        import ctypes

        lib = ctypes.CDLL(None, use_errno=True)
        lib.proc_listpids.restype = ctypes.c_int
        lib.proc_listpids.argtypes = (ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int)
        lib.proc_pidpath.restype = ctypes.c_int
        lib.proc_pidpath.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
        _DARWIN_LIBPROC = lib
    return _DARWIN_LIBPROC


def _darwin_pids() -> list[int]:
    import ctypes

    lib = _darwin_libproc()
    size = 4096 * 4
    while True:
        buffer = ctypes.create_string_buffer(size)
        used = lib.proc_listpids(_DARWIN_ALL_PIDS, 0, buffer, size)
        if used <= 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
        if used < size:
            return [pid for pid in struct.unpack_from(f"<{used // 4}i", buffer.raw) if pid > 0]
        size *= 2


def _darwin_executable(pid: int) -> tuple[str | None, bool]:
    """``(path, unlinked)``; ``(None, False)`` when the process is gone or not ours."""
    import ctypes
    import errno

    lib = _darwin_libproc()
    buffer = ctypes.create_string_buffer(_DARWIN_PATH_MAX)
    ctypes.set_errno(0)
    if lib.proc_pidpath(pid, buffer, _DARWIN_PATH_MAX) > 0:
        return os.fsdecode(buffer.value), False
    # ENOENT is the kernel saying the process is alive but its executable has no name left,
    # the same failure tccd logs as ``proc_pidpath_audittoken() failed``.
    return None, ctypes.get_errno() == errno.ENOENT


def _linux_pids() -> list[int]:
    return [int(name) for name in os.listdir("/proc") if name.isdigit()]


def _linux_executable(pid: int) -> tuple[str | None, bool]:
    try:
        target = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None, False  # gone, a kernel thread, or not ours
    if target.endswith(_LINUX_DELETED) and not os.path.exists(target):
        return target[: -len(_LINUX_DELETED)], True
    return target, False


def _platform():
    if sys.platform == "darwin":
        return _darwin_pids, _darwin_executable
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        return _linux_pids, _linux_executable
    return None


def executable_state(pid: int) -> tuple[str | None, bool] | None:
    """``(executable path, unlinked)`` for ``pid``; None where this platform cannot tell.

    The path is the kernel's current name for the executable, so it follows renames. ``unlinked``
    is True when the process runs an executable that no longer has any name (macOS reports no
    path; Linux reports the last name with a `` (deleted)`` suffix, returned here without it).
    """
    platform = _platform()
    if platform is None:
        return None
    return platform[1](pid)


def _fold(path: str) -> str:
    path = os.path.normcase(path)
    # APFS defaults to case-insensitive and libproc reports the opener's spelling; folding on a
    # case-sensitive volume can only over-report a holder, which keeps a tree, never loses one.
    return path.casefold() if sys.platform == "darwin" else path


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    candidate = _fold(path)
    return any(candidate == root or candidate.startswith(root + os.sep) for root in roots)


def executable_holders(tree: Path) -> list[int]:
    """PIDs of live processes whose executable lies inside ``tree``.

    Only processes this user may inspect are visible, which covers every Hermes process the
    installer's own user started. Platforms without a process table (Windows refuses to delete a
    mapped image instead) report none. Raises OSError when the process table cannot be read.
    """
    platform = _platform()
    if platform is None:
        return []
    list_pids, executable = platform
    roots = tuple({_fold(os.path.abspath(tree)), _fold(os.path.realpath(tree))})
    holders = []
    for pid in list_pids():
        path, unlinked = executable(pid)
        if path is not None and not unlinked and _inside(path, roots):
            holders.append(pid)
    return holders

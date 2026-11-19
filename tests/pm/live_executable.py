"""Start a real process whose executable lives inside a given directory tree.

The running interpreter's own binary is copied in and pointed back at its standard library, so the
kernel reports ``<tree>/bin/<name>`` as the process executable, exactly like a gateway started from
a PM store entry. PM's managed Python is relocatable this way; an interpreter that loads a sibling
framework library (python.org / Xcode builds) cannot start from a copy and the test is skipped.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@contextmanager
def running_from(tree: Path):
    binary = Path(sys._base_executable).resolve()
    copy = tree / "bin" / binary.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(binary, copy)
    process = subprocess.Popen(
        [str(copy), "-S", "-c", "import time; print('ready', flush=True); time.sleep(120)"],
        env={**os.environ, "PYTHONHOME": sys.base_prefix}, stdout=subprocess.PIPE, text=True,
    )
    try:
        if process.stdout.readline().strip() != "ready":
            pytest.skip(f"{binary} does not run from a copied location")
        yield process
    finally:
        process.kill()
        process.wait(timeout=10)
        process.stdout.close()

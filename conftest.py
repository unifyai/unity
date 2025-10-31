import sys
import pytest
from pathlib import Path
from typing import Optional
from datetime import datetime
from unity.constants import PYTEST_LOG_TO_FILE


_TEE_FILE_HANDLE: Optional[object] = None
_TEE_ORIG_STREAM: Optional[object] = None
_TEE_STREAM_ATTR: Optional[str] = None
_TEE_LOG_PATH: Optional[Path] = None


class _TeeStream:
    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, s):
        self._mirror.write(s)
        self._mirror.flush()
        return self._primary.write(s)

    def flush(self):
        self._mirror.flush()
        return self._primary.flush()

    def isatty(self):
        fn = getattr(self._primary, "isatty", None)
        return bool(fn and fn())

    def fileno(self):
        fn = getattr(self._primary, "fileno", None)
        return fn() if fn else -1

    def writable(self):
        fn = getattr(self._primary, "writable", None)
        return bool(fn and fn())

    @property
    def closed(self):
        return getattr(self._primary, "closed", False)


def pytest_sessionstart(session):
    if not PYTEST_LOG_TO_FILE:
        return
    config = session.config
    if getattr(config.option, "collectonly", False):
        # Skip logging for test discovery runs to avoid huge collection trees
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return

    root_path = Path(config.rootpath)
    logs_dir = root_path / ".pytest_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build a readable, second-precision timestamp (e.g., 2025-10-31_14-05-23)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Open a unique file using exclusive creation to avoid races across concurrent runs
    def _open_unique(path_no_suffix: Path):
        if path_no_suffix.exists():
            # We still use exclusive open below; the check is just a fast path to pick a suffix
            pass
        suffix_index = 0
        while True:
            name = f"{path_no_suffix.stem}{'' if suffix_index == 0 else f' ({suffix_index})'}{path_no_suffix.suffix}"
            candidate = path_no_suffix.with_name(name)
            try:
                fh = open(candidate, mode="x", encoding="utf-8")
                return candidate, fh
            except FileExistsError:
                suffix_index += 1

    base_path = logs_dir / f"{ts}.txt"
    log_path, fh = _open_unique(base_path)

    global _TEE_FILE_HANDLE, _TEE_ORIG_STREAM, _TEE_STREAM_ATTR
    _TEE_FILE_HANDLE = fh
    global _TEE_LOG_PATH
    _TEE_LOG_PATH = log_path

    # Mirror the IDE runner's pre-launch banner into the file (file-only, no terminal dup).
    _TEE_FILE_HANDLE.write(f"Running pytest with args: {sys.argv[1:]}\n")
    _TEE_FILE_HANDLE.flush()

    # Identify the TerminalWriter's underlying stream attribute.
    if hasattr(tr._tw, "_file"):
        _TEE_STREAM_ATTR = "_file"
    elif hasattr(tr._tw, "file"):
        _TEE_STREAM_ATTR = "file"
    else:
        _TEE_STREAM_ATTR = None

    if _TEE_STREAM_ATTR is None:
        return

    _TEE_ORIG_STREAM = getattr(tr._tw, _TEE_STREAM_ATTR)
    setattr(tr._tw, _TEE_STREAM_ATTR, _TeeStream(_TEE_ORIG_STREAM, _TEE_FILE_HANDLE))

    # Test body prints are captured by pytest; we mirror them via pytest_runtest_logreport.


def pytest_unconfigure(config):
    """Print the log file path after pytest's own terminal summary has been emitted."""
    if not PYTEST_LOG_TO_FILE:
        return
    global _TEE_FILE_HANDLE, _TEE_ORIG_STREAM, _TEE_STREAM_ATTR, _TEE_LOG_PATH
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None and _TEE_FILE_HANDLE is not None:
        log_file = (
            _TEE_LOG_PATH or (Path(config.rootpath) / ".pytest_logs" / "unknown.txt")
        ).resolve()
        tr.write_line(f"Test logs saved here: {log_file}")
    # Append a file-only trailer to match the IDE runner's banner.
    if _TEE_FILE_HANDLE is not None:
        _TEE_FILE_HANDLE.write("Finished running tests!\n")
        _TEE_FILE_HANDLE.flush()
    if _TEE_FILE_HANDLE is not None:
        _TEE_FILE_HANDLE.flush()
        _TEE_FILE_HANDLE.close()
        _TEE_FILE_HANDLE = None
    if _TEE_ORIG_STREAM is not None and _TEE_STREAM_ATTR is not None:
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            setattr(tr._tw, _TEE_STREAM_ATTR, _TEE_ORIG_STREAM)
        _TEE_ORIG_STREAM = None
        _TEE_STREAM_ATTR = None
    _TEE_LOG_PATH = None
    # No sys.stdout/sys.stderr monkeypatch remains; nothing to restore here.


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Ensure captured stdout/stderr from tests is mirrored into the file.

    With --capture=tee-sys, prints are shown live in terminal but may not reach our
    TerminalReporter tee due to capture internals. Here we append the captured
    streams from the test's call phase directly to the file to guarantee inclusion.
    """
    if not PYTEST_LOG_TO_FILE:
        return
    # Only mirror the main call phase where test body runs
    if getattr(report, "when", None) != "call":
        return
    global _TEE_FILE_HANDLE
    if _TEE_FILE_HANDLE is None:
        return
    out = []
    capout = getattr(report, "capstdout", "")
    caperr = getattr(report, "capstderr", "")
    if capout:
        out.append(capout)
    if caperr:
        out.append(caperr)
    if out:
        _TEE_FILE_HANDLE.write("".join(out))
        _TEE_FILE_HANDLE.flush()

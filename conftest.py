import sys
import os
import pytest
from pathlib import Path
from typing import Optional
from datetime import datetime
from unity.constants import PYTEST_LOG_TO_FILE


_TEE_FILE_HANDLE: Optional[object] = None
_TEE_ORIG_STREAM: Optional[object] = None
_TEE_STREAM_ATTR: Optional[str] = None
_TEE_LOG_PATH: Optional[Path] = None


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Log Naming Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _path_to_name(path: str) -> str:
    """Convert a test path to a filename-safe string.

    Examples:
        tests/test_contact_manager/test_ask.py → test_contact_manager-test_ask
        test_foo.py → test_foo
    """
    name = path.rstrip("/\\")
    # Strip common prefixes
    for prefix in ("tests/", "tests\\", "./", ".\\"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # Strip .py suffix
    if name.endswith(".py"):
        name = name[:-3]
    # Replace path separators with dashes
    return name.replace("/", "-").replace("\\", "-")


def _derive_multi_target_name(paths: list) -> str:
    """Derive a meaningful name when multiple test targets are provided.

    Strategy:
    1. If all targets are from the same .py file, use file name + first test + count
    2. If all targets share a common directory, use that + first file/test + count
    3. Otherwise, use first target + count
    """
    # Extract base file (before ::) for each path
    bases = []
    nodes = []
    for p in paths:
        if "::" in p:
            base, node = p.split("::", 1)
            bases.append(base)
            nodes.append(node)
        else:
            bases.append(p)
            nodes.append(None)

    # Check if all from the same file
    unique_bases = set(bases)
    if len(unique_bases) == 1:
        # All targets from the same file
        base_name = _path_to_name(bases[0])
        if nodes[0]:
            # Sanitize node: replace :: with -, remove brackets
            first_node = nodes[0].replace("::", "-").replace("[", "-").replace("]", "")
            # Truncate if too long
            if len(first_node) > 40:
                first_node = first_node[:37] + "..."
            extra = len(paths) - 1
            if extra > 0:
                return f"{base_name}--{first_node}+{extra}more"
            return f"{base_name}--{first_node}"
        extra = len(paths) - 1
        return f"{base_name}+{extra}more" if extra > 0 else base_name

    # Check for common directory prefix
    def get_dir(p: str) -> str:
        # Get directory part of a path
        if "/" in p:
            return p.rsplit("/", 1)[0]
        if "\\" in p:
            return p.rsplit("\\", 1)[0]
        return ""

    dirs = [get_dir(b) for b in bases]
    unique_dirs = set(dirs)
    if len(unique_dirs) == 1 and dirs[0]:
        # All from same directory
        dir_name = _path_to_name(dirs[0])
        first_file = _path_to_name(bases[0])
        # Remove directory prefix from first_file for brevity
        if first_file.startswith(dir_name + "-"):
            first_file = first_file[len(dir_name) + 1 :]
        extra = len(paths) - 1
        if extra > 0:
            return f"{dir_name}--{first_file}+{extra}more"
        return f"{dir_name}--{first_file}"

    # Fallback: use first target + count
    first_name = _path_to_name(paths[0].split("::")[0])
    if "::" in paths[0]:
        node = paths[0].split("::", 1)[1]
        node = node.replace("::", "-").replace("[", "-").replace("]", "")
        if len(node) > 30:
            node = node[:27] + "..."
        first_name = f"{first_name}--{node}"
    extra = len(paths) - 1
    return f"{first_name}+{extra}more" if extra > 0 else first_name


def _derive_log_name_from_args(args: list) -> str:
    """Derive a semantic log filename from pytest command-line args.

    Examples:
        ['tests/test_contact_manager/test_ask.py']
            → 'test_contact_manager-test_ask'
        ['tests/test_contact_manager/test_ask.py::test_foo']
            → 'test_contact_manager-test_ask--test_foo'
        ['tests/test_contact_manager/']
            → 'test_contact_manager'
        ['tests/']
            → 'tests'
        [] (no args)
            → 'all'
        Multiple from same file:
            → 'test_session_behavior--TestA-test_x+1more'
        Multiple from same directory:
            → 'test_contact_manager--test_ask+2more'
    """
    if not args:
        return "all"

    # Filter to actual test paths (ignore flags like -v, --tb=short, etc.)
    paths = []
    for a in args:
        if isinstance(a, str) and not a.startswith("-"):
            # Looks like a path or node id
            if a.endswith(".py") or "::" in a or "/" in a or "\\" in a:
                paths.append(a)
            elif os.path.exists(a):
                paths.append(a)

    if not paths:
        return "all"

    # For single target, use detailed naming
    if len(paths) == 1:
        target = paths[0]
        # Handle pytest node ids (path::test_name or path::Class::test_name)
        if "::" in target:
            base, node = target.split("::", 1)
            base_name = _path_to_name(base)
            # Sanitize node: replace :: with -, remove brackets
            node_name = node.replace("::", "-").replace("[", "-").replace("]", "")
            return f"{base_name}--{node_name}"
        return _path_to_name(target)

    # Multiple targets: find common structure for a meaningful name
    return _derive_multi_target_name(paths)


def _get_log_subdir() -> str:
    """Determine the log subdirectory based on terminal/socket context.

    Returns:
        - Socket name (e.g., 'unity_dev_ttys042') if UNITY_TEST_SOCKET is set
        - 'standalone' for direct pytest invocations
    """
    socket = os.environ.get("UNITY_TEST_SOCKET", "").strip()
    if socket:
        return socket
    return "standalone"


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

    # Determine subdirectory based on terminal context (socket name or 'standalone')
    subdir = _get_log_subdir()
    logs_dir = root_path / ".pytest_logs" / subdir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build timestamp suffix for uniqueness
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Derive semantic name from command-line args
    semantic_name = _derive_log_name_from_args(list(config.args))

    # Open a unique file using exclusive creation to avoid races across concurrent runs
    def _open_unique(path_no_suffix: Path):
        suffix_index = 0
        while True:
            name = f"{path_no_suffix.stem}{'' if suffix_index == 0 else f' ({suffix_index})'}{path_no_suffix.suffix}"
            candidate = path_no_suffix.with_name(name)
            try:
                fh = open(candidate, mode="x", encoding="utf-8")
                return candidate, fh
            except FileExistsError:
                suffix_index += 1

    # Build the log filename: semantic_name_timestamp.txt
    base_path = logs_dir / f"{semantic_name}_{ts}.txt"
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
        root_path = Path(config.rootpath)
        subdir = _get_log_subdir()

        # Print a clear banner showing where logs are
        tr.write_line("")
        tr.write_line("=" * 72)
        tr.write_line(f"📄 Test log: {log_file}")
        tr.write_line(
            f"📁 This terminal's logs: {root_path / '.pytest_logs' / subdir}/",
        )
        tr.write_line(f"📂 All terminals' logs:  {root_path / '.pytest_logs'}/*/")
        tr.write_line("=" * 72)
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

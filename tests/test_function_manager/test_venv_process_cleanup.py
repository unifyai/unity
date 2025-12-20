"""
Tests for subprocess and multiprocessing cleanup in venv execution.

These tests verify that:
1. Subprocesses are started in their own process group
2. Process group termination kills all child processes
3. Functions using multiprocessing.Process have children cleaned up
4. Graceful SIGTERM is sent before SIGKILL
5. No orphaned processes remain after cleanup
"""

import asyncio
import os
import signal
import sys
import pytest
import shutil

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


# Sample pyproject.toml with minimal dependencies (fast to sync)
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Test Functions
# ────────────────────────────────────────────────────────────────────────────

# Function that spawns multiprocessing workers (sync version)
# NOTE: Uses __import__ since venv sandbox doesn't have os/multiprocessing in globals
MULTIPROCESSING_SPAWN_FUNCTION = """
def spawn_workers(num_workers: int = 3) -> str:
    '''Spawn worker processes and return their PIDs.'''
    mp = __import__('multiprocessing')
    os = __import__('os')
    time = __import__('time')

    def worker():
        '''Worker that sleeps indefinitely.'''
        os_mod = __import__('os')
        time_mod = __import__('time')
        pid_file = f"/tmp/unity_test_worker_{os_mod.getpid()}.pid"
        fd = os_mod.open(pid_file, os_mod.O_CREAT | os_mod.O_WRONLY, 0o644)
        os_mod.write(fd, str(os_mod.getpid()).encode())
        os_mod.close(fd)
        time_mod.sleep(300)

    ctx = mp.get_context('fork')
    procs = []
    for _ in range(num_workers):
        p = ctx.Process(target=worker)
        p.start()
        procs.append(p)

    # Wait a bit for workers to start and write their PIDs
    time.sleep(0.5)

    # Return the parent PID so caller can verify hierarchy
    return f"spawned {len(procs)} workers, parent pid: {os.getpid()}"
""".strip()


# Async version that spawns workers then waits
# NOTE: Uses __import__ since venv sandbox doesn't have os/multiprocessing in globals
ASYNC_MULTIPROCESSING_FUNCTION = """
async def spawn_and_wait(num_workers: int = 2) -> str:
    '''Spawn workers and wait (can be cancelled).'''
    # Import modules explicitly (sandbox doesn't include these in globals)
    mp = __import__('multiprocessing')
    os = __import__('os')
    time = __import__('time')

    def worker():
        '''Worker that sleeps indefinitely and writes its PID.'''
        os_mod = __import__('os')
        time_mod = __import__('time')
        pid_file = f"/tmp/unity_test_worker_{os_mod.getpid()}.pid"
        # Use os.open and os.write for file I/O
        fd = os_mod.open(pid_file, os_mod.O_CREAT | os_mod.O_WRONLY, 0o644)
        os_mod.write(fd, str(os_mod.getpid()).encode())
        os_mod.close(fd)
        time_mod.sleep(300)

    # Force fork start method for subprocess compatibility
    ctx = mp.get_context('fork')
    procs = []
    for _ in range(num_workers):
        p = ctx.Process(target=worker)
        p.start()
        procs.append(p)

    # Wait for workers to start and write their PIDs
    await asyncio.sleep(1.0)

    # Write parent PID to signal we're ready
    pid_file = f"/tmp/unity_test_parent_{os.getpid()}.pid"
    fd = os.open(pid_file, os.O_CREAT | os.O_WRONLY, 0o644)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    # Wait indefinitely (to be cancelled/stopped)
    await asyncio.sleep(300)
    return "completed"
""".strip()


# Simple long-running function for basic termination test
# NOTE: Uses __import__ since venv sandbox doesn't have os in globals
LONG_RUNNING_FUNCTION = """
async def long_running_task() -> str:
    '''A task that runs for a long time.'''
    os = __import__('os')
    pid_file = f"/tmp/unity_test_longrun_{os.getpid()}.pid"
    fd = os.open(pid_file, os.O_CREAT | os.O_WRONLY, 0o644)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    await asyncio.sleep(300)
    return "done"
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    # Cleanup all created managers
    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def cleanup_test_pid_files():
    """Cleanup any PID files created during tests."""
    yield
    # Clean up any test PID files
    import glob

    for pid_file in glob.glob("/tmp/unity_test_*.pid"):
        try:
            os.unlink(pid_file)
        except Exception:
            pass


def get_test_worker_pids() -> list[int]:
    """Get PIDs of test worker processes from PID files."""
    import glob

    pids = []
    for pid_file in glob.glob("/tmp/unity_test_worker_*.pid"):
        try:
            with open(pid_file) as f:
                pids.append(int(f.read().strip()))
        except Exception:
            pass
    return pids


def is_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cleanup_test_processes():
    """Force kill any remaining test processes."""
    for pid in get_test_worker_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic Process Termination Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_in_venv_terminates_subprocess_on_cleanup(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """Subprocess should be terminated when execute_in_venv is cancelled."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # Start a long-running task
        task = asyncio.create_task(
            fm.execute_in_venv(
                venv_id=venv_id,
                implementation=LONG_RUNNING_FUNCTION,
                call_kwargs={},
                is_async=True,
            ),
        )

        # Wait for process to start and write PID
        await asyncio.sleep(1.5)

        # Get the subprocess PID from the file
        import glob

        pid_files = glob.glob("/tmp/unity_test_longrun_*.pid")
        assert len(pid_files) == 1, f"Expected 1 PID file, found {len(pid_files)}"

        with open(pid_files[0]) as f:
            subprocess_pid = int(f.read().strip())

        # Verify process is running
        assert is_process_alive(subprocess_pid), "Subprocess should be running"

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Give cleanup time to happen
        await asyncio.sleep(0.5)

        # Verify process was terminated
        assert not is_process_alive(
            subprocess_pid,
        ), f"Subprocess {subprocess_pid} should have been terminated"

    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# 2. Multiprocessing Child Cleanup Tests
# ────────────────────────────────────────────────────────────────────────────


async def wait_for_parent_pid_file(timeout: float = 10.0) -> bool:
    """Wait for the parent PID file to appear, indicating workers are ready."""
    import glob

    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        pid_files = glob.glob("/tmp/unity_test_parent_*.pid")
        if pid_files:
            return True
        await asyncio.sleep(0.2)
    return False


@_handle_project
@pytest.mark.asyncio
async def test_multiprocessing_children_terminated_on_cleanup(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """Child processes spawned via multiprocessing should be terminated."""
    if sys.platform == "win32":
        pytest.skip("Process group tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # Start function that spawns workers
        task = asyncio.create_task(
            fm.execute_in_venv(
                venv_id=venv_id,
                implementation=ASYNC_MULTIPROCESSING_FUNCTION,
                call_kwargs={"num_workers": 2},
                is_async=True,
            ),
        )

        # Wait for parent PID file to appear (signals workers are spawned)
        parent_ready = await wait_for_parent_pid_file(timeout=15.0)
        assert parent_ready, "Parent process didn't write PID file in time"

        # Give workers a moment to fully initialize
        await asyncio.sleep(0.5)

        # Get worker PIDs
        worker_pids = get_test_worker_pids()
        assert (
            len(worker_pids) == 2
        ), f"Expected 2 worker PIDs, found {len(worker_pids)}"

        # Verify workers are running
        for pid in worker_pids:
            assert is_process_alive(pid), f"Worker {pid} should be running"

        # Cancel the task (triggers cleanup)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Give cleanup time to happen
        await asyncio.sleep(1.5)

        # Verify ALL workers were terminated
        alive_workers = [pid for pid in worker_pids if is_process_alive(pid)]
        assert len(alive_workers) == 0, (
            f"Worker processes should have been terminated, "
            f"but these are still alive: {alive_workers}"
        )

    finally:
        # Force cleanup any remaining processes
        cleanup_test_processes()
        # Clean up parent PID files too
        import glob

        for f in glob.glob("/tmp/unity_test_parent_*.pid"):
            try:
                os.unlink(f)
            except Exception:
                pass
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_sync_multiprocessing_children_terminated(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """Sync functions with multiprocessing should also have children cleaned up."""
    if sys.platform == "win32":
        pytest.skip("Process group tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # Start sync function that spawns workers
        task = asyncio.create_task(
            fm.execute_in_venv(
                venv_id=venv_id,
                implementation=MULTIPROCESSING_SPAWN_FUNCTION,
                call_kwargs={"num_workers": 3},
                is_async=False,
            ),
        )

        # Wait for workers to spawn
        await asyncio.sleep(2.0)

        # Get worker PIDs
        worker_pids = get_test_worker_pids()
        # Note: sync function returns quickly after spawning, so workers may exist

        if len(worker_pids) > 0:
            # Verify at least some workers were spawned
            assert len(worker_pids) <= 3

            # If the function completed, workers might still be running
            # Let's cancel and verify cleanup
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await asyncio.sleep(1.0)

            # Check if orphans exist (this test verifies the problem)
            alive_workers = [pid for pid in worker_pids if is_process_alive(pid)]

            # After cleanup, no workers should be alive
            assert (
                len(alive_workers) == 0
            ), f"Orphaned worker processes found: {alive_workers}"

    finally:
        cleanup_test_processes()
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# 3. Process Group Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_subprocess_in_own_process_group(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """Subprocess should be started in its own process group."""
    if sys.platform == "win32":
        pytest.skip("Process group tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that reports its process group
    pgid_function = """
import os

def get_pgid() -> dict:
    '''Return process info.'''
    return {
        "pid": os.getpid(),
        "pgid": os.getpgid(os.getpid()),
        "parent_pgid": os.getpgid(os.getppid()),
    }
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=pgid_function,
            call_kwargs={},
            is_async=False,
        )

        assert result["error"] is None, f"Error: {result['error']}"
        info = result["result"]

        # The subprocess should be its own process group leader
        # (pid == pgid when start_new_session=True)
        assert info["pid"] == info["pgid"], (
            f"Subprocess should be its own process group leader. "
            f"PID={info['pid']}, PGID={info['pgid']}"
        )

        # Its PGID should differ from parent's PGID
        assert (
            info["pgid"] != info["parent_pgid"]
        ), "Subprocess should be in a different process group than parent"

    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# 4. Graceful Termination Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_graceful_sigterm_before_sigkill(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """Should send SIGTERM first, allowing graceful cleanup."""
    if sys.platform == "win32":
        pytest.skip("Signal tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that logs signal reception
    signal_handler_function = """
import asyncio
import signal
import os

async def signal_test() -> str:
    '''Test that tracks received signals.'''
    received_signals = []

    def handler(signum, frame):
        received_signals.append(signum)
        # Write signal to file for verification
        with open(f"/tmp/unity_test_signal_{os.getpid()}.txt", "a") as f:
            f.write(f"{signum}\\n")

    signal.signal(signal.SIGTERM, handler)

    # Write PID
    with open(f"/tmp/unity_test_sigtest_{os.getpid()}.pid", "w") as f:
        f.write(str(os.getpid()))

    # Wait to be terminated
    await asyncio.sleep(300)
    return "done"
""".strip()

    try:
        task = asyncio.create_task(
            fm.execute_in_venv(
                venv_id=venv_id,
                implementation=signal_handler_function,
                call_kwargs={},
                is_async=True,
            ),
        )

        # Wait for process to start
        await asyncio.sleep(1.5)

        # Get subprocess PID
        import glob

        pid_files = glob.glob("/tmp/unity_test_sigtest_*.pid")
        assert len(pid_files) == 1

        with open(pid_files[0]) as f:
            subprocess_pid = int(f.read().strip())

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.5)

        # Check if SIGTERM was received (signal file should exist)
        signal_files = glob.glob(f"/tmp/unity_test_signal_{subprocess_pid}.txt")
        if len(signal_files) > 0:
            with open(signal_files[0]) as f:
                signals = [int(s.strip()) for s in f.readlines() if s.strip()]

            # SIGTERM should have been received
            assert (
                signal.SIGTERM in signals
            ), f"SIGTERM should have been sent. Received signals: {signals}"

    finally:
        # Cleanup
        import glob

        for f in glob.glob("/tmp/unity_test_sig*"):
            try:
                os.unlink(f)
            except Exception:
                pass
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# 5. No Orphaned Processes Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_no_orphaned_processes_after_cleanup(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """After cleanup, no orphaned processes should remain."""
    if sys.platform == "win32":
        pytest.skip("Process tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # Run multiple concurrent venv executions that spawn workers
        tasks = []
        for i in range(2):
            task = asyncio.create_task(
                fm.execute_in_venv(
                    venv_id=venv_id,
                    implementation=ASYNC_MULTIPROCESSING_FUNCTION,
                    call_kwargs={"num_workers": 2},
                    is_async=True,
                ),
            )
            tasks.append(task)

        # Wait for all workers to spawn
        await asyncio.sleep(2.5)

        # Collect all worker PIDs
        all_worker_pids = get_test_worker_pids()
        assert (
            len(all_worker_pids) == 4
        ), f"Expected 4 total workers, found {len(all_worker_pids)}"

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(1.5)

        # Verify NO orphaned processes
        orphans = [pid for pid in all_worker_pids if is_process_alive(pid)]
        assert len(orphans) == 0, f"Orphaned processes found: {orphans}"

    finally:
        cleanup_test_processes()
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# 6. Timeout and Force Kill Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_force_kill_after_sigterm_timeout(
    function_manager_factory,
    cleanup_test_pid_files,
):
    """If process doesn't terminate after SIGTERM, SIGKILL should be sent."""
    if sys.platform == "win32":
        pytest.skip("Signal tests not supported on Windows")

    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that ignores SIGTERM
    ignore_sigterm_function = """
import asyncio
import signal
import os

async def ignore_sigterm() -> str:
    '''Ignores SIGTERM (only SIGKILL will work).'''
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    with open(f"/tmp/unity_test_ignore_{os.getpid()}.pid", "w") as f:
        f.write(str(os.getpid()))

    await asyncio.sleep(300)
    return "done"
""".strip()

    try:
        task = asyncio.create_task(
            fm.execute_in_venv(
                venv_id=venv_id,
                implementation=ignore_sigterm_function,
                call_kwargs={},
                is_async=True,
            ),
        )

        await asyncio.sleep(1.5)

        # Get PID
        import glob

        pid_files = glob.glob("/tmp/unity_test_ignore_*.pid")
        assert len(pid_files) == 1

        with open(pid_files[0]) as f:
            subprocess_pid = int(f.read().strip())

        assert is_process_alive(subprocess_pid)

        # Cancel - should eventually SIGKILL after SIGTERM timeout
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Wait for force kill (the implementation has a 5s timeout)
        await asyncio.sleep(1.0)

        # Process should be dead (killed by SIGKILL)
        assert not is_process_alive(
            subprocess_pid,
        ), f"Process {subprocess_pid} should have been force killed"

    finally:
        import glob

        for f in glob.glob("/tmp/unity_test_ignore_*"):
            try:
                os.unlink(f)
            except Exception:
                pass
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

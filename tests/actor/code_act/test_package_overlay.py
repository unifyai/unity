"""Tests for the install_python_packages tool and PackageOverlay cleanup.

These are symbolic/infrastructure tests verifying that:
- Packages installed via PackageOverlay become importable in-process.
- PackageOverlay.cleanup() removes packages from sys.path and sys.modules.
- Child overlay cleanup does not affect parent overlay packages.
- The full actor act() loop cleans up installed packages on completion.
"""

import asyncio
import importlib
import os
import re
import sys

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution.package_overlay import (
    PackageOverlay,
    _CURRENT_PACKAGE_OVERLAY,
)

_UV_TIMING_RE = re.compile(r" in \d+(\.\d+)?(ms|s|m)\b")
_original_install = PackageOverlay.install


def _install_without_timings(self, packages, timeout=120):
    result = _original_install(self, packages, timeout=timeout)
    result["stderr"] = _UV_TIMING_RE.sub("", result["stderr"])
    return result


@pytest.fixture(autouse=True)
def _strip_uv_timings(monkeypatch):
    """Strip non-deterministic timing values from uv stderr so that LLM tool
    results are identical across runs, preventing cache busting."""
    monkeypatch.setattr(PackageOverlay, "install", _install_without_timings)


# ---------------------------------------------------------------------------
# Direct PackageOverlay unit tests (no LLM, no actor)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_overlay_install_makes_package_importable():
    """PackageOverlay.install() makes a package importable in-process."""
    overlay = PackageOverlay(agent_id="test_importable")

    try:
        result = overlay.install(["cowsay"])
        assert result["success"], f"Install failed: {result['stderr']}"

        # The package should now be importable.
        mod = importlib.import_module("cowsay")
        assert mod is not None
        assert "cowsay" in sys.modules
    finally:
        overlay.cleanup()


@pytest.mark.timeout(60)
def test_overlay_cleanup_removes_packages():
    """PackageOverlay.cleanup() purges packages from sys.path and sys.modules."""
    overlay = PackageOverlay(agent_id="test_cleanup")

    try:
        result = overlay.install(["cowsay"])
        assert result["success"], f"Install failed: {result['stderr']}"

        # Force the import so it lands in sys.modules.
        importlib.import_module("cowsay")
        assert "cowsay" in sys.modules
    finally:
        overlay.cleanup()

    # After cleanup: sys.modules entry should be gone.
    assert "cowsay" not in sys.modules

    # The overlay directory should no longer exist on disk.
    assert not overlay.active


@pytest.mark.timeout(60)
def test_overlay_cleanup_removes_directory_from_disk():
    """The overlay's temp directory is deleted on cleanup."""
    overlay = PackageOverlay(agent_id="test_disk_cleanup")

    try:
        result = overlay.install(["cowsay"])
        assert result["success"], f"Install failed: {result['stderr']}"
        target_dir = overlay._dir
        assert target_dir is not None
        assert os.path.isdir(target_dir)
    finally:
        overlay.cleanup()

    assert not os.path.exists(target_dir)


@pytest.mark.timeout(60)
def test_overlay_noop_cleanup_when_nothing_installed():
    """Cleanup on an overlay that never installed anything is a safe no-op."""
    overlay = PackageOverlay(agent_id="test_noop")
    overlay.cleanup()  # Should not raise.
    assert not overlay.active


# ---------------------------------------------------------------------------
# Hierarchical overlay tests (parent/child nesting)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_child_overlay_cleanup_preserves_parent_packages():
    """When a child overlay cleans up, the parent's packages stay importable."""
    parent_overlay = PackageOverlay(agent_id="parent")
    parent_token = _CURRENT_PACKAGE_OVERLAY.set(parent_overlay)

    try:
        parent_result = parent_overlay.install(["cowsay"])
        assert parent_result[
            "success"
        ], f"Parent install failed: {parent_result['stderr']}"

        # Child overlay discovers parent via the ContextVar.
        child_overlay = PackageOverlay(agent_id="child")
        child_token = _CURRENT_PACKAGE_OVERLAY.set(child_overlay)

        try:
            child_result = child_overlay.install(["pyfiglet"])
            assert child_result[
                "success"
            ], f"Child install failed: {child_result['stderr']}"

            # Both should be importable.
            importlib.import_module("cowsay")
            importlib.import_module("pyfiglet")
            assert "cowsay" in sys.modules
            assert "pyfiglet" in sys.modules

            # Child cleans up.
            child_overlay.cleanup()

            # pyfiglet should be gone from sys.modules.
            assert "pyfiglet" not in sys.modules

            # cowsay should STILL be in sys.modules (parent's package).
            assert "cowsay" in sys.modules
        finally:
            try:
                _CURRENT_PACKAGE_OVERLAY.reset(child_token)
            except Exception:
                pass
    finally:
        parent_overlay.cleanup()
        try:
            _CURRENT_PACKAGE_OVERLAY.reset(parent_token)
        except Exception:
            pass

    # After parent cleanup, cowsay should also be gone.
    assert "cowsay" not in sys.modules


@pytest.mark.timeout(90)
def test_child_overlay_creates_subdirectory_of_parent():
    """The child overlay's directory is nested under the parent's directory."""
    parent_overlay = PackageOverlay(agent_id="parent_dir_test")
    parent_token = _CURRENT_PACKAGE_OVERLAY.set(parent_overlay)

    try:
        parent_overlay.install(["cowsay"])
        parent_dir = parent_overlay._dir
        assert parent_dir is not None

        child_overlay = PackageOverlay(agent_id="child_dir_test")
        child_token = _CURRENT_PACKAGE_OVERLAY.set(child_overlay)

        try:
            child_overlay.install(["pyfiglet"])
            child_dir = child_overlay._dir
            assert child_dir is not None

            # Child directory should be a subdirectory of parent.
            assert child_dir.startswith(parent_dir + os.sep), (
                f"Expected child dir {child_dir!r} to be under parent dir "
                f"{parent_dir!r}"
            )
        finally:
            child_overlay.cleanup()
            try:
                _CURRENT_PACKAGE_OVERLAY.reset(child_token)
            except Exception:
                pass
    finally:
        parent_overlay.cleanup()
        try:
            _CURRENT_PACKAGE_OVERLAY.reset(parent_token)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Integration test: full actor act() loop with package cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_packages_not_importable_after_act_completes():
    """Packages installed during act() are cleaned up when the trajectory ends."""
    actor = CodeActActor(timeout=60)
    handle = None

    try:
        handle = await actor.act(
            "Use the install_python_packages tool to install the package "
            "'cowsay'. Then use execute_code with language='python' to run: "
            "import cowsay; print('INSTALL_OK'). "
            "Report 'done' when finished.",
        )
        result = await asyncio.wait_for(handle.result(), timeout=90)
        # The actor should have completed (we don't assert on exact wording,
        # just that it didn't error out).
        assert isinstance(result, str)
    finally:
        if handle and not handle.done():
            try:
                await handle.stop("test cleanup")
            except Exception:
                pass
        await actor.close()

    # After act() cleanup, cowsay should NOT be importable.
    assert (
        "cowsay" not in sys.modules
    ), "cowsay should have been purged from sys.modules after act() cleanup"

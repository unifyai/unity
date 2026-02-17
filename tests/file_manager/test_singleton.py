from __future__ import annotations

from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager import simulated as sim_mod


def test_local_is_singleton(fm_root):
    a = LocalFileManager(fm_root)
    b = LocalFileManager(fm_root)
    assert a is b, "LocalFileManager should be a singleton per domain/root"


def test_simulated_is_not_singleton():
    f1 = sim_mod.SimulatedFileManager()
    f2 = sim_mod.SimulatedFileManager()
    assert f1 is not f2, "SimulatedFileManager should allow multiple instances"

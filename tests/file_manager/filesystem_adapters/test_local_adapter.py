from __future__ import annotations

from pathlib import Path

import pytest

from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter


@pytest.mark.asyncio
async def test_basic_iter_get_open(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (root / "b.txt").write_text("world", encoding="utf-8")

    ad = LocalFileSystemAdapter(root.as_posix())
    files = list(ad.iter_files("."))
    names = {f.name for f in files}
    assert {"a.txt", "b.txt"}.issubset(names)

    ref = ad.get_file("a.txt")
    assert ref.name == "a.txt"
    assert ref.path.endswith("/a.txt")
    assert ref.size_bytes == len("hello".encode("utf-8"))
    assert ad.open_bytes("a.txt") == b"hello"


@pytest.mark.asyncio
async def test_rename_and_move(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "x.txt").write_text("x", encoding="utf-8")
    ad = LocalFileSystemAdapter(root.as_posix())

    # rename
    out = ad.rename("x.txt", "y.txt")
    assert out.name == "y.txt"
    assert out.path.endswith("/y.txt")

    # move
    (root / "sub").mkdir()
    out2 = ad.move("y.txt", "sub")
    assert out2.path.endswith("/sub/y.txt")
    # Ensure original no longer exists and new does
    with pytest.raises(FileNotFoundError):
        ad.get_file("y.txt")
    assert ad.get_file("sub/y.txt").name == "y.txt"


@pytest.mark.asyncio
async def test_import_and_register(tmp_path):
    root = tmp_path / "root"
    src = tmp_path / "src"
    root.mkdir()
    src.mkdir()
    (src / "d.txt").write_text("data", encoding="utf-8")
    ad = LocalFileSystemAdapter(root.as_posix())

    name = ad.import_file((src / "d.txt").as_posix())
    assert name == "d.txt"
    assert (root / name).exists()
    assert (root / name).read_text(encoding="utf-8") == "data"

    name2 = ad.register_existing_file((src / "d.txt").as_posix())
    assert name2 in {"d.txt", "d (1).txt"}
    assert (root / name2).exists()


@pytest.mark.asyncio
async def test_downloads(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ad = LocalFileSystemAdapter(root.as_posix())
    disp = ad.save_file_to_downloads("z.bin", b"\x00\x01")
    assert disp.startswith("Downloads/")
    resolved = ad.resolve_display_name(disp)
    assert resolved is not None and Path(resolved).exists()
    assert Path(resolved).read_bytes() == b"\x00\x01"
    # Idempotent resolution for plain names under root
    p = root / "plain.txt"
    p.write_text("ok", encoding="utf-8")
    assert ad.resolve_display_name("plain.txt").endswith("/plain.txt")


@pytest.mark.asyncio
async def test_delete(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ad = LocalFileSystemAdapter(root.as_posix())

    # Create a file to delete
    p = root / "delete_me.txt"
    p.write_text("some content")

    # Delete the file
    ad.delete("delete_me.txt")

    # Verify
    assert not p.exists()

    # Test deleting a non-existent file
    with pytest.raises(FileNotFoundError):
        ad.delete("non_existent_file.txt")

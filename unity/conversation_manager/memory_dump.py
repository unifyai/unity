"""Exhaustive memory dump for OOM diagnostics.

Called when the memory watchdog triggers graceful shutdown.  Writes a
comprehensive report to ``{UNITY_LOG_DIR}/oom_memory_dump.txt`` covering:

- Cgroup memory breakdown (RSS, cache, swap, kernel, …)
- Per-process RSS/VmSize for every process in the container
- Python GC stats and object counts by type
- Top Python objects by estimated size
- Loaded module count and largest modules by attribute footprint
- Open file descriptors
- /proc/self/status and /proc/self/smaps_rollup

The report is deliberately over-verbose — the goal is to pinpoint exactly
what is consuming memory when an OOM event occurs.
"""

from __future__ import annotations

import gc
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _section(out: io.StringIO, title: str) -> None:
    out.write(f"\n{'=' * 80}\n")
    out.write(f"  {title}\n")
    out.write(f"{'=' * 80}\n\n")


def _read_proc_file(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _dump_cgroup_memory_stat(out: io.StringIO) -> None:
    """Cgroup v2 memory.stat or v1 equivalent — kernel-level breakdown."""
    _section(out, "CGROUP MEMORY BREAKDOWN")

    for path in ("/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat"):
        content = _read_proc_file(path)
        if content:
            out.write(f"Source: {path}\n\n")
            for line in content.strip().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    key, val = parts
                    try:
                        mib = int(val) / 1048576
                        out.write(f"  {key:40s} {val:>15s}  ({mib:,.1f} MiB)\n")
                    except ValueError:
                        out.write(f"  {key:40s} {val:>15s}\n")
                else:
                    out.write(f"  {line}\n")
            return

    # Current usage + limit
    for cur, lim in [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ]:
        cur_val = _read_proc_file(cur)
        lim_val = _read_proc_file(lim)
        if cur_val:
            out.write(
                f"  Current: {cur_val.strip()} ({int(cur_val) / 1048576:,.1f} MiB)\n",
            )
            if lim_val and lim_val.strip() != "max":
                out.write(
                    f"  Limit:   {lim_val.strip()} ({int(lim_val) / 1048576:,.1f} MiB)\n",
                )
            return

    out.write("  (cgroup memory files not found)\n")


def _dump_process_tree(out: io.StringIO) -> None:
    """RSS and VmSize for every process visible in /proc."""
    _section(out, "PER-PROCESS MEMORY (all PIDs in container)")

    procs: list[tuple[int, str, int, int]] = []  # (pid, name, rss_kb, vm_kb)
    proc_path = Path("/proc")
    if not proc_path.exists():
        out.write("  (/proc not available)\n")
        return

    for entry in sorted(proc_path.iterdir()):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        status = _read_proc_file(f"/proc/{pid}/status")
        if not status:
            continue

        name = "?"
        rss_kb = 0
        vm_kb = 0
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("VmRSS:"):
                try:
                    rss_kb = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("VmSize:"):
                try:
                    vm_kb = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
        procs.append((pid, name, rss_kb, vm_kb))

    total_rss = sum(p[2] for p in procs)
    procs.sort(key=lambda p: p[2], reverse=True)

    out.write(f"  {'PID':>7s}  {'NAME':25s}  {'RSS':>12s}  {'VmSize':>12s}\n")
    out.write(f"  {'-' * 7}  {'-' * 25}  {'-' * 12}  {'-' * 12}\n")
    for pid, name, rss_kb, vm_kb in procs:
        out.write(
            f"  {pid:7d}  {name:25s}" f"  {rss_kb:>9,d} kB" f"  {vm_kb:>9,d} kB\n",
        )
    out.write(
        f"\n  Total RSS across all processes: {total_rss:,d} kB ({total_rss / 1024:,.1f} MiB)\n",
    )
    out.write(
        "  NOTE: Shared pages (libc, Python, etc.) are counted in each\n"
        "  process's RSS but only use physical memory once.  The sum\n"
        "  therefore overstates true usage.  Cgroup memory.stat is the\n"
        "  authoritative figure for container-level consumption.\n",
    )


def _dump_self_status(out: io.StringIO) -> None:
    """/proc/self/status for the main Python process."""
    _section(out, "MAIN PROCESS /proc/self/status")
    content = _read_proc_file("/proc/self/status")
    if content:
        out.write(f"  PID: {os.getpid()}\n\n")
        for line in content.strip().splitlines():
            out.write(f"  {line}\n")
    else:
        out.write("  (not available)\n")


def _dump_smaps_rollup(out: io.StringIO) -> None:
    """/proc/self/smaps_rollup — aggregated memory map summary."""
    _section(out, "MAIN PROCESS /proc/self/smaps_rollup")
    content = _read_proc_file("/proc/self/smaps_rollup")
    if content:
        for line in content.strip().splitlines():
            out.write(f"  {line}\n")
    else:
        out.write("  (not available — requires kernel 4.14+)\n")


def _dump_gc_stats(out: io.StringIO) -> None:
    """Python garbage collector statistics."""
    _section(out, "PYTHON GC STATISTICS")

    stats = gc.get_stats()
    for i, gen in enumerate(stats):
        out.write(f"  Generation {i}: {gen}\n")

    out.write(f"\n  gc.get_count(): {gc.get_count()}\n")
    out.write(f"  gc.isenabled(): {gc.isenabled()}\n")


def _dump_object_census(out: io.StringIO) -> None:
    """Count of live Python objects by type, sorted by count."""
    _section(out, "PYTHON OBJECT CENSUS (by type, top 60)")

    type_counts: dict[str, int] = {}
    type_sizes: dict[str, int] = {}

    for obj in gc.get_objects():
        t = type(obj).__name__
        type_counts[t] = type_counts.get(t, 0) + 1
        try:
            type_sizes[t] = type_sizes.get(t, 0) + sys.getsizeof(obj)
        except (TypeError, ReferenceError):
            pass

    out.write(f"  Total tracked objects: {sum(type_counts.values()):,d}\n\n")
    out.write(f"  {'TYPE':40s}  {'COUNT':>10s}  {'SHALLOW SIZE':>14s}\n")
    out.write(f"  {'-' * 40}  {'-' * 10}  {'-' * 14}\n")

    for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:60]:
        size = type_sizes.get(t, 0)
        if size >= 1048576:
            out.write(f"  {t:40s}  {count:>10,d}  {size / 1048576:>10,.1f} MiB\n")
        elif size >= 1024:
            out.write(f"  {t:40s}  {count:>10,d}  {size / 1024:>10,.1f} kiB\n")
        else:
            out.write(f"  {t:40s}  {count:>10,d}  {size:>10,d} B\n")


def _dump_largest_objects(out: io.StringIO) -> None:
    """Top 40 individual objects by sys.getsizeof()."""
    _section(out, "LARGEST INDIVIDUAL PYTHON OBJECTS (top 40)")

    sized: list[tuple[int, str, str]] = []
    for obj in gc.get_objects():
        try:
            s = sys.getsizeof(obj)
        except (TypeError, ReferenceError):
            continue
        if s < 65536:  # only care about objects > 64 kB
            continue
        t = type(obj).__name__
        rep = (
            repr(obj)[:120]
            if not isinstance(obj, (bytes, bytearray))
            else f"<{t} len={len(obj)}>"
        )
        sized.append((s, t, rep))

    sized.sort(reverse=True)
    out.write(f"  {'SIZE':>12s}  {'TYPE':20s}  REPR\n")
    out.write(f"  {'-' * 12}  {'-' * 20}  {'-' * 40}\n")
    for s, t, rep in sized[:40]:
        if s >= 1048576:
            out.write(f"  {s / 1048576:>9,.1f} MiB  {t:20s}  {rep}\n")
        else:
            out.write(f"  {s / 1024:>9,.1f} kiB  {t:20s}  {rep}\n")

    if not sized:
        out.write("  (no objects > 64 kB found)\n")


def _dump_modules(out: io.StringIO) -> None:
    """Loaded Python modules sorted by rough attribute footprint."""
    _section(out, "LOADED PYTHON MODULES (top 60 by attribute footprint)")

    mod_sizes: list[tuple[int, str, int]] = []  # (size, name, attr_count)
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        total = 0
        attr_count = 0
        try:
            for attr in dir(mod):
                attr_count += 1
                try:
                    total += sys.getsizeof(getattr(mod, attr))
                except (TypeError, ReferenceError, AttributeError):
                    pass
        except Exception:
            pass
        mod_sizes.append((total, name, attr_count))

    mod_sizes.sort(reverse=True)
    out.write(f"  Total modules loaded: {len(sys.modules):,d}\n\n")
    out.write(f"  {'ATTR SIZE':>12s}  {'ATTRS':>6s}  MODULE\n")
    out.write(f"  {'-' * 12}  {'-' * 6}  {'-' * 40}\n")
    for size, name, attr_count in mod_sizes[:60]:
        if size >= 1048576:
            out.write(f"  {size / 1048576:>9,.1f} MB  {attr_count:>6d}  {name}\n")
        elif size >= 1024:
            out.write(f"  {size / 1024:>9,.1f} kB  {attr_count:>6d}  {name}\n")
        else:
            out.write(f"  {size:>10d} B  {attr_count:>6d}  {name}\n")


def _dump_open_fds(out: io.StringIO) -> None:
    """Open file descriptors for the main process."""
    _section(out, "OPEN FILE DESCRIPTORS (main process)")

    fd_dir = Path(f"/proc/{os.getpid()}/fd")
    if not fd_dir.exists():
        out.write("  (/proc/self/fd not available)\n")
        return

    fds: list[tuple[int, str]] = []
    for entry in sorted(fd_dir.iterdir(), key=lambda e: int(e.name)):
        try:
            target = os.readlink(str(entry))
        except OSError:
            target = "?"
        fds.append((int(entry.name), target))

    out.write(f"  Total open FDs: {len(fds)}\n\n")
    for fd, target in fds:
        out.write(f"  fd {fd:4d} → {target}\n")


def write_oom_memory_dump(log_dir: str | Path | None = None) -> Path | None:
    """Write an exhaustive memory dump and return the file path.

    Args:
        log_dir: Directory to write the dump file.  Falls back to
                 UNITY_LOG_DIR, then /var/log/unity, then /tmp.

    Returns:
        Path to the dump file, or None if writing failed entirely.
    """
    # Resolve output directory
    if log_dir is None:
        log_dir = os.environ.get("UNITY_LOG_DIR", "").strip()
    if not log_dir:
        for fallback in ("/var/log/unity", "/tmp"):
            if os.path.isdir(fallback):
                log_dir = fallback
                break
    if not log_dir:
        log_dir = "/tmp"

    dump_path = Path(log_dir) / "oom_memory_dump.txt"

    out = io.StringIO()
    ts = datetime.now(timezone.utc).isoformat()
    out.write(f"OOM PREVENTION MEMORY DUMP — {ts}\n")
    out.write(f"PID: {os.getpid()}  Python: {sys.version}\n")

    collectors = [
        _dump_cgroup_memory_stat,
        _dump_process_tree,
        _dump_self_status,
        _dump_smaps_rollup,
        _dump_gc_stats,
        _dump_object_census,
        _dump_largest_objects,
        _dump_modules,
        _dump_open_fds,
    ]

    for collector in collectors:
        try:
            collector(out)
        except Exception as exc:
            out.write(f"\n  *** {collector.__name__} failed: {exc} ***\n")

    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(out.getvalue(), encoding="utf-8")
        return dump_path
    except Exception:
        # Last resort — try /tmp
        fallback = Path("/tmp/oom_memory_dump.txt")
        try:
            fallback.write_text(out.getvalue(), encoding="utf-8")
            return fallback
        except Exception:
            return None

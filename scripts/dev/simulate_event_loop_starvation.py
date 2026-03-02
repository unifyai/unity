#!/usr/bin/env python3
"""
Reproduce & diagnose event-loop starvation in the CM → Actor pipeline.

Models the actual async architecture where:

  1. CM._run_llm processes screenshots (sync I/O + backend calls)
  2. CM calls single_shot_tool_decision → LLM → asyncio.gather(act_tool)
  3. act_tool spawns Actor setup as a CONCURRENT background task, then returns
  4. asyncio.gather tries to resume single_shot, but Actor setup's callbacks
     keep the event loop busy → starvation gap
  5. Actor setup: append_msgs (includes synchronous blocking from
     unillm.append_messages), tool policy, asyncio.sleep(0) yield, LLM call
  6. FunctionManager: sync_primitives + sandbox setup

The key insight is that the Actor setup runs CONCURRENTLY with the gather
resumption, creating real callback contention on the single asyncio event loop.
Synchronous blocking operations (modelling unillm.append_messages, GCP Pub/Sub
client internals) block the event loop thread entirely during the "silent gaps"
observed in production logs.

Usage:
    # Default (reproduces production config)
    python3 scripts/dev/simulate_event_loop_starvation.py

    # No background load or sync blocking (ideal baseline)
    python3 scripts/dev/simulate_event_loop_starvation.py --bg-load 0 --sync-block-dispatch-ms 0 --sync-block-append-ms 0

    # Heavy background load
    python3 scripts/dev/simulate_event_loop_starvation.py --bg-load 3.0

    # Sweep across load levels
    python3 scripts/dev/simulate_event_loop_starvation.py --sweep

    # See all parameters
    python3 scripts/dev/simulate_event_loop_starvation.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Latencies for each pipeline stage (ms). Defaults from production logs."""

    # ── Pre-LLM screenshot processing ─────────────────────────────────────
    screenshot_count: int = 8
    screenshot_save_ms: float = 50  # per-screenshot sync disk write
    image_manager_add_ms: float = 2000  # to_thread → backend
    tm_update_per_msg_ms: float = 500  # to_thread → backend per msg
    prompt_build_ms: float = 1000  # snapshot + build_brain_spec

    # ── CM slow-brain LLM call ────────────────────────────────────────────
    cm_llm_ms: float = 14500

    # ── Synchronous blocking during tool dispatch ─────────────────────────
    # In production, 4.77s of zero log output between asyncio.gather
    # scheduling the Task and the Task body starting. Hypothesised to be
    # synchronous JSON serialisation of the ~200K-token context in unillm,
    # or GCP Pub/Sub client flow-control.
    sync_block_dispatch_ms: float = 4770

    # ── CM.act tool body ──────────────────────────────────────────────────
    actor_handle_setup_ms: float = 50
    watcher_setup_ms: float = 360
    event_publish_ms: float = 360

    # ── Actor setup (concurrent background task) ──────────────────────────
    # Synchronous blocking portion of _msg_dispatcher.append_msgs
    # (6.57s silent gap → likely unillm.append_messages serialising context)
    sync_block_append_ms: float = 6570
    # Async portion of system msg append (event bus publish via Pub/Sub)
    event_bus_publish_ms: float = 500
    tool_policy_ms: float = 100
    preflight_ms: float = 3600

    # ── Actor LLM call ────────────────────────────────────────────────────
    actor_llm_ms: float = 16800

    # ── FunctionManager execute_function overhead ─────────────────────────
    fn_manager_overhead_ms: float = 7200


@dataclass
class BackgroundLoadConfig:
    """Background tasks competing for the event loop."""

    load_multiplier: float = 1.0

    # Async tasks that yield frequently (Pub/Sub, IPC, event handlers, etc.)
    n_yielding_tasks: int = 20
    yield_work_ms: float = 2  # CPU spin per iteration
    yield_sleep_ms: float = 30  # async sleep per iteration

    # Tasks that periodically block the event loop synchronously
    # (simulates GCP Pub/Sub client internals, JSON serialisation, etc.)
    n_sync_block_tasks: int = 3
    sync_block_ms: float = 150
    sync_block_interval_ms: float = 600

    # Thread pool workers (shared with pipeline to_thread calls)
    thread_pool_size: int = 4
    n_thread_pool_tasks: int = 3
    thread_pool_work_ms: float = 400

    @property
    def effective_counts(self) -> dict[str, int]:
        m = self.load_multiplier
        return {
            "yielding": max(0, int(self.n_yielding_tasks * m)),
            "sync_block": max(0, int(self.n_sync_block_tasks * m)),
            "thread_pool": max(0, int(self.n_thread_pool_tasks * m)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_T0 = 0.0


def wall() -> float:
    return time.perf_counter() - _T0


def wall_ms() -> str:
    return f"{wall() * 1000:.0f}ms"


def spin_ms(ms: float):
    """Burn CPU for *ms* milliseconds."""
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


def block_ms(ms: float):
    """Block the current thread for *ms* milliseconds."""
    time.sleep(ms / 1000)


# ──────────────────────────────────────────────────────────────────────────────
# Background load
# ──────────────────────────────────────────────────────────────────────────────

_stop = asyncio.Event()


async def _bg_yielding(work_ms: float, sleep_ms: float):
    while not _stop.is_set():
        spin_ms(work_ms)
        await asyncio.sleep(sleep_ms / 1000)


async def _bg_sync_block(block_dur_ms: float, interval_ms: float):
    while not _stop.is_set():
        block_ms(block_dur_ms)
        await asyncio.sleep(interval_ms / 1000)


async def _bg_thread_pool(pool: ThreadPoolExecutor, work_ms: float):
    loop = asyncio.get_event_loop()
    while not _stop.is_set():
        await loop.run_in_executor(pool, block_ms, work_ms)
        await asyncio.sleep(0.01)


def start_bg(cfg: BackgroundLoadConfig, pool: ThreadPoolExecutor) -> list[asyncio.Task]:
    counts = cfg.effective_counts
    tasks: list[asyncio.Task] = []
    for _ in range(counts["yielding"]):
        tasks.append(
            asyncio.create_task(
                _bg_yielding(cfg.yield_work_ms, cfg.yield_sleep_ms),
            ),
        )
    for _ in range(counts["sync_block"]):
        tasks.append(
            asyncio.create_task(
                _bg_sync_block(cfg.sync_block_ms, cfg.sync_block_interval_ms),
            ),
        )
    for _ in range(counts["thread_pool"]):
        tasks.append(
            asyncio.create_task(
                _bg_thread_pool(pool, cfg.thread_pool_work_ms),
            ),
        )
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Timing collection
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Timings:
    user_utterance: float = 0.0
    screenshot_end: float = 0.0
    cm_llm_start: float = 0.0
    cm_llm_end: float = 0.0
    dispatch_start: float = 0.0
    tool_body_start: float = 0.0
    act_tool_end: float = 0.0
    gather_resume: float = 0.0
    actor_msg_append_end: float = 0.0
    actor_yield_start: float = 0.0
    actor_yield_end: float = 0.0
    actor_llm_start: float = 0.0
    actor_llm_end: float = 0.0
    fn_start: float = 0.0
    fn_end: float = 0.0

    def _ms(self, start: float, end: float) -> float:
        return (end - start) * 1000

    def report(self) -> str:
        lines = [
            "",
            "=" * 80,
            "PIPELINE TIMING REPORT",
            "=" * 80,
            "",
            "Phase-by-phase:",
            "-" * 80,
        ]

        phases = [
            ("1. Screenshot processing", self.user_utterance, self.screenshot_end),
            ("2. CM LLM call", self.cm_llm_start, self.cm_llm_end),
            (
                "3. Tool dispatch gap  ← starvation",
                self.dispatch_start,
                self.tool_body_start,
            ),
            ("4. CM.act tool body", self.tool_body_start, self.act_tool_end),
            (
                "5. Gather resume gap  ← starvation",
                self.act_tool_end,
                self.gather_resume,
            ),
            (
                "6a. Actor: system msg append",
                self.gather_resume,
                self.actor_msg_append_end,
            ),
            (
                "6b. Actor: sleep(0) yield  ← starvation",
                self.actor_yield_start,
                self.actor_yield_end,
            ),
            ("7. Actor LLM call", self.actor_llm_start, self.actor_llm_end),
            ("8. FunctionManager overhead", self.fn_start, self.fn_end),
        ]
        for label, s, e in phases:
            lines.append(f"  {label:<48s} {self._ms(s, e):>8.0f}ms")

        total = self._ms(self.user_utterance, self.fn_end)
        cm_to_actor = self._ms(self.cm_llm_start, self.gather_resume)
        cm_llm = self._ms(self.cm_llm_start, self.cm_llm_end)
        actor_llm = self._ms(self.actor_llm_start, self.actor_llm_end)

        lines += [
            "",
            "-" * 80,
            f"  {'TOTAL (user → fn execution starts)':<48s} {total:>8.0f}ms",
            f"  {'CM thinking → Actor request (info log 107→113)':<48s} {cm_to_actor:>8.0f}ms",
            "",
            "Breakdown of CM → Actor gap:",
            f"  CM LLM inference (necessary):    {cm_llm:>8.0f}ms",
            f"  Overhead in CM → Actor gap:      {cm_to_actor - cm_llm:>8.0f}ms",
            "",
            "Full pipeline overhead:",
            f"  LLM inference (CM + Actor):      {cm_llm + actor_llm:>8.0f}ms",
            f"  All other (I/O + starvation):    {total - cm_llm - actor_llm:>8.0f}ms",
            f"  Overhead ratio:                  {(total - cm_llm - actor_llm) / total * 100:>7.1f}%",
            "",
            "Starvation indicators (should be <10ms in healthy system):",
            "-" * 80,
            f"  Tool dispatch gap:     {self._ms(self.dispatch_start, self.tool_body_start):>8.0f}ms",
            f"  Gather resume gap:     {self._ms(self.act_tool_end, self.gather_resume):>8.0f}ms",
            f"  asyncio.sleep(0):      {self._ms(self.actor_yield_start, self.actor_yield_end):>8.0f}ms",
            "",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline simulation
# ──────────────────────────────────────────────────────────────────────────────


async def run_pipeline(cfg: PipelineConfig, pool: ThreadPoolExecutor) -> Timings:
    t = Timings()
    loop = asyncio.get_event_loop()

    # ── User utterance ────────────────────────────────────────────────────
    t.user_utterance = wall()
    print(f"  [{wall_ms():>8s}] 🧑 User: 'open the browser'")

    # ── Phase 1: Screenshot processing ────────────────────────────────────
    for _ in range(cfg.screenshot_count):
        spin_ms(cfg.screenshot_save_ms)
    await loop.run_in_executor(pool, block_ms, cfg.image_manager_add_ms)
    n_msgs = max(1, cfg.screenshot_count // 3)
    for _ in range(n_msgs):
        await loop.run_in_executor(pool, block_ms, cfg.tm_update_per_msg_ms)
    spin_ms(cfg.prompt_build_ms)
    t.screenshot_end = wall()
    print(f"  [{wall_ms():>8s}] 📸 Screenshots processed")

    # ── Phase 2: CM LLM call ──────────────────────────────────────────────
    t.cm_llm_start = wall()
    await asyncio.sleep(cfg.cm_llm_ms / 1000)
    t.cm_llm_end = wall()
    print(f"  [{wall_ms():>8s}] 🧠 CM LLM → tool='act'")

    # ── Actor setup task (spawned by CM.act, runs concurrently) ───────────
    # This is the background task whose callbacks compete with the gather
    # resumption. It starts running as soon as CM.act calls actor.act().
    actor_ready = asyncio.Event()
    actor_timings: dict[str, float] = {}

    async def actor_setup():
        """Simulates the Actor's async tool loop setup phase."""
        # Synchronous blocking: unillm.append_messages serialises the
        # ~200K-token context. This blocks the event loop thread.
        block_ms(cfg.sync_block_append_ms)

        # Async portion: event bus publish (Pub/Sub network I/O)
        await asyncio.sleep(cfg.event_bus_publish_ms / 1000)
        actor_timings["msg_append_end"] = wall()

        # Tool policy eval
        spin_ms(cfg.tool_policy_ms)
        await asyncio.sleep(cfg.preflight_ms / 1000)

        # The critical asyncio.sleep(0) yield
        actor_timings["yield_start"] = wall()
        await asyncio.sleep(0)
        actor_timings["yield_end"] = wall()

        actor_ready.set()

    # ── Phase 3+4+5: Tool dispatch → CM.act → gather resume ──────────────
    t.dispatch_start = wall()

    async def simulated_act_tool():
        """CM.act brain action tool: spawns Actor, returns handle."""
        # Synchronous blocking at dispatch: the silent 4.77s gap
        # observed in production between gather scheduling and tool entry.
        # Likely unillm context serialisation or GCP Pub/Sub flow control.
        block_ms(cfg.sync_block_dispatch_ms)

        t.tool_body_start = wall()
        print(f"  [{wall_ms():>8s}] 🔧 CM.act tool entered")

        # actor.act() → start background loop, return handle
        await asyncio.sleep(cfg.actor_handle_setup_ms / 1000)
        # Spawn the actor setup as a CONCURRENT task (critical for starvation)
        asyncio.create_task(actor_setup())

        # Watcher setup
        spin_ms(cfg.watcher_setup_ms)
        # event_broker.publish
        await asyncio.sleep(cfg.event_publish_ms / 1000)

        t.act_tool_end = wall()
        print(f"  [{wall_ms():>8s}] 🔧 CM.act tool returning")
        return {"status": "acting"}

    # asyncio.gather wraps in a Task; the gather resume is delayed by
    # the Actor setup's callbacks competing for the event loop.
    await asyncio.gather(simulated_act_tool())
    t.gather_resume = wall()
    print(f"  [{wall_ms():>8s}] ✅ single_shot: all tools completed")

    # ── Wait for Actor to be ready (it's been running concurrently) ───────
    await actor_ready.wait()
    t.actor_msg_append_end = actor_timings.get("msg_append_end", wall())
    t.actor_yield_start = actor_timings.get("yield_start", wall())
    t.actor_yield_end = actor_timings.get("yield_end", wall())
    print(
        f"  [{wall_ms():>8s}] 📝 Actor setup complete "
        f"(sleep(0)={t._ms(t.actor_yield_start, t.actor_yield_end):.0f}ms)",
    )

    # ── Phase 7: Actor LLM call ──────────────────────────────────────────
    t.actor_llm_start = wall()
    await asyncio.sleep(cfg.actor_llm_ms / 1000)
    t.actor_llm_end = wall()
    print(f"  [{wall_ms():>8s}] 🧠 Actor LLM → tools scheduled")

    # ── Phase 8: FunctionManager overhead ─────────────────────────────────
    t.fn_start = wall()
    await loop.run_in_executor(pool, block_ms, cfg.fn_manager_overhead_ms)
    t.fn_end = wall()
    print(f"  [{wall_ms():>8s}] 🚀 execute_function: desktop.act starts")

    return t


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


async def _run_once(
    pipe_cfg: PipelineConfig,
    bg_cfg: BackgroundLoadConfig,
    *,
    quiet: bool = False,
) -> Timings:
    global _T0, _stop
    _stop = asyncio.Event()
    pool = ThreadPoolExecutor(max_workers=bg_cfg.thread_pool_size)
    _T0 = time.perf_counter()

    bg_tasks = start_bg(bg_cfg, pool)
    counts = bg_cfg.effective_counts
    total_bg = sum(counts.values())
    if not quiet:
        print(
            f"\n  Background: {total_bg} tasks "
            f"(yield={counts['yielding']}, sync_block={counts['sync_block']}, "
            f"thread_pool={counts['thread_pool']})",
        )
        print()

    timings = await run_pipeline(pipe_cfg, pool)

    _stop.set()
    for task in bg_tasks:
        task.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)
    pool.shutdown(wait=False)
    return timings


def build_configs(args) -> tuple[PipelineConfig, BackgroundLoadConfig]:
    pipe_cfg = PipelineConfig(
        screenshot_count=args.screenshot_count,
        screenshot_save_ms=args.screenshot_save_ms,
        image_manager_add_ms=args.image_manager_add_ms,
        tm_update_per_msg_ms=args.tm_update_per_msg_ms,
        prompt_build_ms=args.prompt_build_ms,
        cm_llm_ms=args.cm_llm_ms,
        sync_block_dispatch_ms=args.sync_block_dispatch_ms,
        actor_handle_setup_ms=args.actor_handle_setup_ms,
        watcher_setup_ms=args.watcher_setup_ms,
        event_publish_ms=args.event_publish_ms,
        sync_block_append_ms=args.sync_block_append_ms,
        event_bus_publish_ms=args.event_bus_publish_ms,
        tool_policy_ms=args.tool_policy_ms,
        preflight_ms=args.preflight_ms,
        actor_llm_ms=args.actor_llm_ms,
        fn_manager_overhead_ms=args.fn_manager_overhead_ms,
    )
    bg_cfg = BackgroundLoadConfig(
        load_multiplier=args.bg_load,
        n_yielding_tasks=args.yielding_tasks,
        n_sync_block_tasks=args.bg_sync_block_tasks,
        sync_block_ms=args.bg_sync_block_ms,
        sync_block_interval_ms=args.bg_sync_block_interval_ms,
        thread_pool_size=args.thread_pool_size,
        n_thread_pool_tasks=args.thread_pool_tasks,
        thread_pool_work_ms=args.thread_pool_work_ms,
    )
    return pipe_cfg, bg_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Simulate event loop starvation in the CM → Actor pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline
    p = parser.add_argument_group("Pipeline latencies (ms)")
    p.add_argument("--screenshot-count", type=int, default=8)
    p.add_argument("--screenshot-save-ms", type=float, default=50)
    p.add_argument("--image-manager-add-ms", type=float, default=2000)
    p.add_argument("--tm-update-per-msg-ms", type=float, default=500)
    p.add_argument("--prompt-build-ms", type=float, default=1000)
    p.add_argument("--cm-llm-ms", type=float, default=14500)
    p.add_argument(
        "--sync-block-dispatch-ms",
        type=float,
        default=4770,
        help="Sync blocking during tool dispatch (silent gap)",
    )
    p.add_argument("--actor-handle-setup-ms", type=float, default=50)
    p.add_argument("--watcher-setup-ms", type=float, default=360)
    p.add_argument("--event-publish-ms", type=float, default=360)
    p.add_argument(
        "--sync-block-append-ms",
        type=float,
        default=6570,
        help="Sync blocking in _msg_dispatcher.append_msgs (silent gap)",
    )
    p.add_argument("--event-bus-publish-ms", type=float, default=500)
    p.add_argument("--tool-policy-ms", type=float, default=100)
    p.add_argument("--preflight-ms", type=float, default=3600)
    p.add_argument("--actor-llm-ms", type=float, default=16800)
    p.add_argument("--fn-manager-overhead-ms", type=float, default=7200)

    # Background load
    b = parser.add_argument_group("Background load")
    b.add_argument(
        "--bg-load",
        type=float,
        default=1.0,
        help="Scale all background task counts (0=none)",
    )
    b.add_argument("--yielding-tasks", type=int, default=20)
    b.add_argument("--bg-sync-block-tasks", type=int, default=3)
    b.add_argument("--bg-sync-block-ms", type=float, default=150)
    b.add_argument("--bg-sync-block-interval-ms", type=float, default=600)
    b.add_argument("--thread-pool-size", type=int, default=4)
    b.add_argument("--thread-pool-tasks", type=int, default=3)
    b.add_argument("--thread-pool-work-ms", type=float, default=400)

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep across configurations",
    )

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
        return

    pipe_cfg, bg_cfg = build_configs(args)

    print("\n" + "=" * 80)
    print("EVENT LOOP STARVATION SIMULATOR")
    print("=" * 80)

    timings = asyncio.run(_run_once(pipe_cfg, bg_cfg))
    print(timings.report())


def run_sweep(args):
    print("\n" + "=" * 80)
    print("SWEEP: varying sync blocking and background load")
    print("=" * 80)

    configs = [
        ("Ideal (no overhead)", 0, 0, 0.0),
        ("Async overhead only (no sync blocking)", 0, 0, 1.0),
        ("Sync blocking only (no bg load)", 4770, 6570, 0.0),
        ("Production (sync block + bg load)", 4770, 6570, 1.0),
        ("Heavy (sync block + 2x bg load)", 4770, 6570, 2.0),
        ("Extreme (sync block + 3x bg load)", 4770, 6570, 3.0),
    ]

    results: list[tuple[str, Timings]] = []
    for label, dispatch_block, append_block, bg_load in configs:
        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(
            f"  sync_block_dispatch={dispatch_block}ms, "
            f"sync_block_append={append_block}ms, bg_load={bg_load}",
        )
        print(f"{'─' * 80}")

        args.sync_block_dispatch_ms = dispatch_block
        args.sync_block_append_ms = append_block
        args.bg_load = bg_load
        pipe_cfg, bg_cfg = build_configs(args)

        t = asyncio.run(_run_once(pipe_cfg, bg_cfg))
        results.append((label, t))

    # Summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(
        f"\n  {'Config':<42s}  {'Total':>8s}  {'CM→Act':>8s}  "
        f"{'CM LLM':>8s}  {'Disp.':>8s}  {'Gather':>8s}",
    )
    print(f"  {'─' * 42}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for label, t in results:
        total = t._ms(t.user_utterance, t.fn_end)
        cm_to_actor = t._ms(t.cm_llm_start, t.gather_resume)
        cm_llm = t._ms(t.cm_llm_start, t.cm_llm_end)
        dispatch = t._ms(t.dispatch_start, t.tool_body_start)
        gather = t._ms(t.act_tool_end, t.gather_resume)

        print(
            f"  {label:<42s}  {total:>7.0f}ms  {cm_to_actor:>7.0f}ms  "
            f"{cm_llm:>7.0f}ms  {dispatch:>7.0f}ms  {gather:>7.0f}ms",
        )

    print(
        "\n  CM→Act = CM LLM thinking → Actor receives request (info-log lines 107→113)",
    )
    print("  Ideal CM→Act = CM LLM time only (~14500ms)")
    print()


if __name__ == "__main__":
    main()

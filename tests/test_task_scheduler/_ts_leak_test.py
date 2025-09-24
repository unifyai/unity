from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import List, Optional

from sandboxes.utils import (
    configure_sandbox_logging,
    activate_project,
    apply_per_task_simulation_patch,
    SimulationParams,
)
from sandboxes.task_scheduler.sandbox import _dispatch_with_context  # type: ignore
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor
from pytest import MonkeyPatch
from unity.task_scheduler.active_task import ActiveTask
from unity.common.async_tool_loop import AsyncToolUseLoopHandle


class _MemoryLogHandler(logging.Handler):
    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)


def _find_execute_label(
    records: List[logging.LogRecord],
    *,
    after_ts: float = 0.0,
) -> Optional[str]:
    pattern = re.compile(r"\[(TaskScheduler\.execute\([0-9a-f]{4}\))\]")
    label: Optional[str] = None
    for rec in records:
        if rec.created < after_ts:
            continue
        msg = rec.getMessage()
        m = pattern.search(msg)
        if m:
            label = m.group(1)
    return label


async def _seed_tasks(ts: TaskScheduler) -> None:
    # Create the four tasks and materialize a queue in one shot to avoid slow LLM seeding
    tasks = [
        {
            "name": "Research competitor: Accounting Limited",
            "description": (
                'Investigate the competitor "Accounting Limited". Capture: company overview, core products/services, '
                "pricing and plans, target customer segments, market positioning and differentiators, integrations/partner "
                "ecosystem, go-to-market motions, strengths/weaknesses, and recent news/announcements. Research sources: "
                "official site (product/pricing/docs/blog), review sites (G2/Capterra), LinkedIn, press releases/news, and "
                "public filings/registries where applicable. Deliverable: structured notes with citations and key takeaways."
            ),
        },
        {
            "name": "Research competitor: Accounting Inc.",
            "description": (
                'Investigate the competitor "Accounting Inc.". Capture: company overview, core products/services, pricing and plans, '
                "target customer segments, market positioning and differentiators, integrations/partner ecosystem, go-to-market motions, "
                "strengths/weaknesses, and recent news/announcements. Research sources: official site (product/pricing/docs/blog), review "
                "sites (G2/Capterra), LinkedIn, press releases/news, and public filings where applicable. Deliverable: structured notes "
                "with citations and key takeaways."
            ),
        },
        {
            "name": "Compile competitor research report",
            "description": (
                "Synthesize findings on Accounting Limited and Accounting Inc. into a concise comparative report. Include: executive summary, "
                "feature and pricing comparison, SWOT for each, positioning overview, key differentiators, notable gaps/opportunities, and "
                "recommendations/next steps. Include links to sources and appendices with raw notes. Output as an editable doc and export to PDF."
            ),
        },
        {
            "name": "Email report to the boss",
            "description": (
                "Draft and send an email to our boss attaching/linking the competitor research report. Include a brief executive summary, top "
                "findings, risks/opportunities, and proposed next steps. Use a clear subject line and ensure the correct recipient(s) per our "
                "distribution list. Confirm the report link/attachment works."
            ),
        },
    ]
    ts._create_tasks(  # noqa: SLF001
        tasks=tasks,
        queue_ordering=[
            {
                "order": [0, 1, 2, 3],
                "queue_head": {"start_at": "2025-09-29T08:00:00Z"},
            },
        ],
    )


async def main() -> None:
    # Environment to keep things snappy and deterministic locally
    os.environ.setdefault("UNIFY_CACHE", "true")
    os.environ.setdefault("UNIFY_TRACED", "true")
    os.environ.setdefault("UNITY_LOG_ONLY_PROJECT", "true")
    os.environ.setdefault("UNITY_LOG_INCLUDE_PREFIXES", "unity,unify_requests")

    # Logging to terminal; avoid file outputs for a quick dev loop
    configure_sandbox_logging(
        log_in_terminal=True,
        log_file=None,
        tcp_port=0,
        http_tcp_port=0,
        unify_requests_log_file=None,
    )

    # Attach in-memory capture AFTER configuring logging
    mem = _MemoryLogHandler()
    mem.addFilter(lambda r: (r.name or "").startswith("unity"))
    logging.getLogger().addHandler(mem)

    # Activate a clean project and reset EventBus contexts
    activate_project("TaskSchedulerLeakRepro", overwrite=True)

    # Install runtime listeners so we only ask AFTER the SimulatedActor has actually started
    actor_started = asyncio.Event()
    try:
        mp = MonkeyPatch()
    except Exception:
        mp = None  # fallback if pytest not available at runtime

    _orig_create_cm = ActiveTask.create
    _orig_create_fn = (
        _orig_create_cm.__func__
        if hasattr(_orig_create_cm, "__func__")
        else _orig_create_cm
    )

    async def _wrapped_create(
        cls,
        actor,
        *,
        task_description: str,
        parent_chat_context: Optional[list[dict]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ):
        handle = await _orig_create_fn(
            cls,
            actor,
            task_description=task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            task_id=task_id,
            instance_id=instance_id,
            scheduler=scheduler,
        )
        try:
            actor_started.set()
        except Exception:
            pass
        return handle

    if mp is not None:
        mp.setattr(ActiveTask, "create", classmethod(_wrapped_create), raising=False)
    else:
        ActiveTask.create = classmethod(_wrapped_create)

    # Also hook the outer loop's delegate adoption – this fires when execute_by_id returns
    _orig_adopt = AsyncToolUseLoopHandle._adopt

    def _wrapped_adopt(self, new_handle):  # type: ignore[no-redef]
        try:
            _orig_adopt(self, new_handle)
        finally:
            try:
                actor_started.set()
            except Exception:
                pass

    if mp is not None:
        mp.setattr(AsyncToolUseLoopHandle, "_adopt", _wrapped_adopt, raising=False)
    else:
        AsyncToolUseLoopHandle._adopt = _wrapped_adopt  # type: ignore[assignment]

    # Build scheduler + actor (explicit duration to ensure status logs are emitted)
    ts = TaskScheduler(actor=SimulatedActor(duration=500.0, log_mode="print"))

    # Seed four tasks exactly as used in the observed session
    await _seed_tasks(ts)

    # Hard-coded scripted scenario utterances (decoupled from scenarios.json)
    scenarios = {
        # First execute: start Accounting Limited research with explicit simulation instructions
        "user command 1": (
            "Actually could you please start the research on accounting limited right now? "
            "As for the simulation, let's set the duration to 500 seconds and if the user asks for a progress update, "
            "say it's taking longer than expected, but it should be done by the end of the day."
        ),
        # Ask once the actor has started and status logs appear
        "user /freeform interjection 0 (after seeing '⏳ SimulatedActor Duration remaining: Xs')": "How is the task coming along?",
        # Stop reason for the first loop
        "user /freeform interjection 1": "Okay let's not bother then, let's do this next week as we originally planned.",
        # Second execute: start Accounting Inc research (no follow-up ask); include simulation hint
        "user command 2": (
            "Let's start the research on Accounting Inc now instead. "
            "For the simulation, set the duration to 500 seconds and if asked for a progress update, say: "
            "the internet is currently down, trying to restart the router but it doesn't seem to be working, will keep trying."
        ),
    }

    chat_history: List[dict] = []

    # 1) Start the first execute request (Accounting Limited)
    # Ensure the simulated actor uses a long duration (500s) for this execute
    apply_per_task_simulation_patch(
        per_call_overrides=SimulationParams(duration_seconds=500.0),
        log_mode="print",
    )
    cmd1 = scenarios["user command 1"]
    _kind1, handle1, _u1, _d1 = await _dispatch_with_context(
        ts,
        cmd1,
        show_steps=False,
        parent_chat_context=list(chat_history),
        clarifications_enabled=False,
        enable_voice=False,
        isolated=None,
    )

    # Allow the loop to spin up and print its first lines so we can record the label
    await asyncio.sleep(0.8)
    label1 = _find_execute_label(mem.records)
    if not label1:
        # Give it one more brief chance in slow environments
        await asyncio.sleep(0.8)
        label1 = _find_execute_label(mem.records)
    if not label1:
        raise RuntimeError(
            "Could not detect label for the first execute loop – aborting repro.",
        )

    # Wait until the actor is definitely running (started via execute_by_id inside execute)
    # Wait up to 3 minutes for adoption/actor start; in CI this may take time
    try:
        await asyncio.wait_for(actor_started.wait(), timeout=180.0)
    except asyncio.TimeoutError:
        # Best-effort fallback: if we see a simulated-actor status log, proceed
        saw_actor_status = any(
            (r.name or "").startswith("unity.simulated_actor")
            and "SimulatedActor Duration remaining" in r.getMessage()
            for r in mem.records
        )
        if not saw_actor_status:
            # Proceed anyway to avoid hanging indefinitely, but we tried to delay sufficiently
            pass
        detected_start = saw_actor_status
    else:
        detected_start = True

    # a) Add a 10s delay after we DETECT the start before asking
    if detected_start:
        await asyncio.sleep(10.0)
    # Freeform ask while running (post-start)
    try:
        interjection0 = scenarios[
            "user /freeform interjection 0 (after seeing '⏳ SimulatedActor Duration remaining: Xs')"
        ]
    except KeyError:
        interjection0 = "How is the task coming along?"
    # b) Send ask, then wait for the ANSWER, then delay 5s before sending stop
    nested0 = await handle1.ask(interjection0)
    try:
        _ans0 = await nested0.result()
    except Exception:
        _ans0 = None
    # Print the answer similar to the real sandbox
    try:
        if _ans0 is not None:
            print(f"[ask] → {_ans0}")
    except Exception:
        pass

    # Request stop of the first loop (5s after receiving the ask answer)
    await asyncio.sleep(5.0)
    stop_reason1 = scenarios.get("user /freeform interjection 1", "Let's stop for now.")
    handle1.stop(stop_reason1)
    stop_t = time.time()

    # 2) Wait 5s, then start a fresh execute request (Accounting Inc)
    await asyncio.sleep(5.0)

    # 2) Start the second execute request (Accounting Inc)
    # Re-assert the long simulated duration for the second execute as well
    apply_per_task_simulation_patch(
        per_call_overrides=SimulationParams(duration_seconds=500.0),
        log_mode="print",
    )
    cmd2 = scenarios["user command 2"]
    _kind2, handle2, _u2, _d2 = await _dispatch_with_context(
        ts,
        cmd2,
        show_steps=False,
        parent_chat_context=list(chat_history),
        clarifications_enabled=False,
        enable_voice=False,
        isolated=None,
    )

    # Let the second execute run for 30 seconds to surface any leakage
    await asyncio.sleep(30.0)

    # Assertion: after the first stop, there must be NO further logs from its loop
    offending = [
        r for r in mem.records if (r.created >= stop_t and label1 in r.getMessage())
    ]
    if offending:
        sample = "\n".join(x.getMessage() for x in offending[:6])
        print("\n=== Detected leakage from the first execute loop after stop ===")
        print(f"Loop label: [{label1}]")
        print("Offending log lines (sample):\n" + sample)
        raise AssertionError(
            f"Leak reproduced: saw {len(offending)} post-stop log(s) from [{label1}]",
        )

    print("No leakage detected (no post-stop logs from the first execute loop).")


if __name__ == "__main__":
    asyncio.run(main())

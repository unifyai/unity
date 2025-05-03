import queue
import threading
import pytest

from planner.primitives import set_queues
from planner.verifier import Verifier, verify, BubbleUp
from planner.code_rewriter import rewrite_function


def test_bubble_up(monkeypatch):
    #  flag that flips after rewrite
    rewrite_done = {"yes": False}

    # Verifier.check returns 'push_up_stack' for low() UNTIL rewrite happens
    def fake_check(src, *_, **__):
        fn_name = src.split("def ")[1].split("(")[0].strip()
        if fn_name == "low" and not rewrite_done["yes"]:
            return "push_up_stack"
        return "ok"

    monkeypatch.setattr(Verifier, "check", fake_check)

    # count rewrites & flip flag
    rewrite_calls = {"count": 0}

    def fake_rewrite(fn):
        rewrite_calls["count"] += 1
        rewrite_done["yes"] = True  # pretend the rewrite fixed things

    monkeypatch.setattr("planner.code_rewriter.rewrite_function", fake_rewrite)

    # immediate‑rewrite queue plumbing
    q = queue.Queue()
    orig_put = q.put

    def put_and_rewrite(item):
        fake_rewrite(item)
        orig_put(item)

    q.put = put_and_rewrite
    monkeypatch.setattr(Verifier, "_reimplement_queue", q)

    # define plan functions
    @verify
    def low():
        pass

    @verify
    def high():
        low()

    # run
    high()  # should finish without error
    assert rewrite_calls["count"] == 1  # exactly one parent rewrite
    assert q.qsize() == 1  # queue contains that one entry

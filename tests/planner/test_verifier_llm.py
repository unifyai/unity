import pytest
from planner.verifier_utils import _hash_dom, dom_diff_summary
import planner.context as context_mod
from planner.verifier import Verifier
from planner.model import Primitive
import planner.verifier as verifier_mod

# Test _hash_dom stability under reordering and recursive hashing


def test_hash_dom_stability_and_recursive():
    node1 = {
        "tag": "div",
        "text": "hello",
        "attributes": {"aria-label": "label", "other": "ignore"},
        "children": [{"tag": "span", "text": "world", "children": []}],
    }
    # Reorder keys and child attributes
    node2 = {
        "children": [{"text": "world", "children": [], "tag": "span"}],
        "attributes": {"other": "ignore", "aria-label": "label"},
        "text": "hello",
        "tag": "div",
    }
    h1 = _hash_dom(node1)
    h2 = _hash_dom(node2)
    assert isinstance(h1, str) and len(h1) == 40
    assert h1 == h2


# Test dom_diff_summary produces unified diff format


def test_dom_diff_summary_basic():
    before = {"a": 1, "b": 2}
    after = {"a": 1, "b": 3}
    diff = dom_diff_summary(before, after)
    assert diff.startswith("--- before")
    assert "+++ after" in diff
    # Check for removal and addition lines
    assert any(
        line.strip().startswith("-") and "2" in line for line in diff.splitlines()
    )
    assert any(
        line.strip().startswith("+") and "3" in line for line in diff.splitlines()
    )


# Test summarise_snapshot strips screenshot, truncates elements, and injects dom_hash


def test_summarise_snapshot_strips_and_truncates_and_hash(monkeypatch):
    # Prepare a large elements list and a dummy dom
    elements = [{"id": i} for i in range(20)]
    snapshot = {
        "url": "http://example.com",
        "screenshot": b"binarydata",
        "elements": elements,
        "dom": {"tag": "p", "text": "paragraph"},
    }
    # Monkey-patch the module-level _hash_dom in context to return a fixed hash
    monkeypatch.setattr(context_mod, "_hash_dom", lambda dom: "fixedhash")
    summary = context_mod.context.summarise_snapshot(snapshot, max_elems=5)
    # 'screenshot' should be removed
    assert "screenshot" not in summary
    # Elements should be truncated to 5 plus a truncated indicator
    assert len(summary["elements"]) == 6
    assert summary["elements"][-1] == {"_truncated": 15}
    # dom_hash should be injected
    assert summary.get("dom_hash") == "fixedhash"


# Test Verifier.check tier1 heuristics for navigation, scrolling, and click_button


def test_verifier_tier1_navigation():
    prim = Primitive("open_url", {"url": "u"}, "")
    before = {"url": "a"}
    after_changed = {"url": "b"}
    after_same = {"url": "a"}
    assert Verifier.check(prim, before, after_changed) == "ok"
    assert Verifier.check(prim, before, after_same) == "reimplement"


@pytest.mark.parametrize(
    "name", ["scroll_up", "scroll_down", "start_scrolling_up", "stop_scrolling"]
)
def test_verifier_tier1_scrolling(name):
    prim = Primitive(name, {}, "")
    # Regardless of snapshots, scrolling is always OK
    assert Verifier.check(prim, {"dom_sha": "x"}, {"dom_sha": "x"}) == "ok"


def test_verifier_tier1_click_button():
    prim = Primitive("click_button", {"target": "btn"}, "")
    before = {"dom_sha": "h1"}
    after_changed = {"dom_sha": "h2"}
    after_same = {"dom_sha": "h1"}
    # Changed DOM: ok
    assert Verifier.check(prim, before, after_changed) == "ok"
    # Same DOM: falls back to hash comparison and interactive rule
    result = Verifier.check(prim, before, after_same)
    # For click_button, tier1 does not auto reimplement; identical hashes -> goes to tier2 and interactive -> reimplement
    assert result == "reimplement"


# Test Verifier.check tier2 fallback for generic interactive and non-interactive


def test_verifier_tier2_interactive():
    prim = Primitive("press_enter", {}, "")
    before = {"dom_sha": "x"}
    after_same = {"dom_sha": "x"}
    after_changed = {"dom_sha": "y"}
    # Changed DOM: ok
    assert Verifier.check(prim, before, after_changed) == "ok"
    # Same DOM: interactive primitive -> reimplement
    assert Verifier.check(prim, before, after_same) == "reimplement"


def test_verifier_tier2_noninteractive():
    prim = Primitive("foo_action", {}, "")
    before = {"dom_sha": "x"}
    after_changed = {"dom_sha": "y"}
    # Changed DOM: ok even if not interactive
    assert Verifier.check(prim, before, after_changed) == "ok"


# Test Verifier.check tier3 LLM fallback by monkey-patching generate_user


def test_verifier_tier3_llm_fallback(monkeypatch):
    prim = Primitive("custom_action", {}, "")
    before = {"dom_sha": "same"}
    after = {"dom_sha": "same"}
    # Stub out summarise_snapshot and dom_diff_summary to avoid real operations
    monkeypatch.setattr(context_mod.context, "summarise_snapshot", lambda snap: snap)
    monkeypatch.setattr(verifier_mod, "dom_diff_summary", lambda b, a: "diff")
    # No-op stateful stub
    monkeypatch.setattr(verifier_mod, "set_stateful", lambda flag: None)

    for llm_response, expected in [
        ("ok", "ok"),
        ("reimplement", "reimplement"),
        ("push_up_stack", "push_up_stack"),
        ("INVALID", "reimplement"),
    ]:
        # Stub generate_user to return varied responses
        monkeypatch.setattr(verifier_mod, "generate_user", lambda prompt: llm_response)
        result = Verifier.check(prim, before, after)
        assert result == expected

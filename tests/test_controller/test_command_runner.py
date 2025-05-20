import types
import sys

# ---------------------------------------------------------------------------
#  Stub heavy deps: redis, playwright, BrowserWorker  (same as previous file)
# ---------------------------------------------------------------------------

# Stub only playwright (CommandRunner depends on it).
plw_mod = types.ModuleType("playwright")
plw_sync = types.ModuleType("playwright.sync_api")


# minimal types used in CommandRunner type hints
class _Stub:  # generic empty class
    pass


plw_sync.BrowserContext = _Stub
plw_sync.Page = _Stub

sys.modules["playwright"] = plw_mod
sys.modules["playwright.sync_api"] = plw_sync

# ---------------------------------------------------------------------------
#  Imports after stubbing
# ---------------------------------------------------------------------------
from unity.controller.playwright import command_runner as cr_mod  # noqa: E402
from unity.controller import commands as cmd_mod  # noqa: E402

# ---------------------------------------------------------------------------
#  Test CommandRunner scroll-speed parsing
# ---------------------------------------------------------------------------


# stub BrowserContext / Page for CommandRunner
class _DummyPage:
    def __init__(self):
        self._scroll_y = 0

        def _wheel(_dx, dy):
            # native wheel scroll affects scrollY too
            self._scroll_y += dy

        self.mouse = types.SimpleNamespace(wheel=_wheel)
        self.keyboard = types.SimpleNamespace(
            press=lambda *a, **k: None,
            type=lambda *a, **k: None,
            down=lambda *a, **k: None,
            up=lambda *a, **k: None,
        )

    # emulate page.evaluate with minimal behaviour for scroll & query
    def evaluate(self, script, *args):
        # querying scrollY
        if script == "scrollY":
            return self._scroll_y
        # handle our injected smooth-scroll by dict arg containing delta
        if args and isinstance(args[0], dict) and "delta" in args[0]:
            self._scroll_y += args[0]["delta"]
            return None
        # default
        return 0

    def bring_to_front(self):
        pass

    def goto(self, url, *_, **__):
        self.url = url
        # simplistic: set title to url
        self._title = url

    def title(self):
        return getattr(self, "_title", "dummy")

    url = "about:blank"


class _DummyCtx:
    def __init__(self):
        self.pages = [_DummyPage()]


# ---------------------------------------------------------------------------
#  Command registry integrity tests
# ---------------------------------------------------------------------------


def test_all_primitives_unique():
    literals = [v for k, v in vars(cmd_mod).items() if k.startswith("CMD_")]
    assert len(literals) == len(set(literals)), "Command literals must be unique"


def test_autoscroll_groups_consistency():
    # every command in AUTOSCROLL_START / ACTIVE must exist in ALL_PRIMITIVES
    base = cmd_mod.ALL_PRIMITIVES
    for g in (cmd_mod.AUTOSCROLL_START, cmd_mod.AUTOSCROLL_ACTIVE):
        assert g <= base


def test_group_subsets():
    """Ensure every group constant is fully included in ALL_PRIMITIVES."""
    groups = [
        cmd_mod.TEXTBOX_COMMANDS,
        cmd_mod.NAV_COMMANDS,
        cmd_mod.BUTTON_PATTERNS,
        cmd_mod.SCROLL_PATTERNS["up"],
        cmd_mod.SCROLL_PATTERNS["down"],
        cmd_mod.AUTOSCROLL_START,
        cmd_mod.AUTOSCROLL_ACTIVE,
        cmd_mod.DIALOG_COMMANDS,
        cmd_mod.POPUP_COMMANDS,
    ]
    master = cmd_mod.ALL_PRIMITIVES
    for grp in groups:
        assert grp <= master


def test_wildcard_trailing_star_patterns():
    """Verify wildcard commands keep the expected '*' suffix where required."""
    # patterns that should contain '*'
    patterns = [
        cmd_mod.CMD_ENTER_TEXT,
        cmd_mod.CMD_SCROLL_DOWN,
        cmd_mod.CMD_SCROLL_UP,
        cmd_mod.CMD_SEARCH,
        cmd_mod.CMD_OPEN_URL,
        cmd_mod.CMD_CLICK_BUTTON,
        cmd_mod.CMD_SELECT_TAB,
        cmd_mod.CMD_CLOSE_TAB,
        cmd_mod.CMD_SELECT_POPUP,
        cmd_mod.CMD_TYPE_DIALOG,
    ]
    for p in patterns:
        assert p.endswith("*"), f"Pattern {p} missing terminal '*'"


# ---------------------------------------------------------------------------
#  Additional state consistency tests
# ---------------------------------------------------------------------------


def test_open_url_updates_state():
    ctx = _DummyCtx()
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.run("open_url example.com")
    assert runner.state.url == "https://example.com"
    assert ctx.pages[0].url == "https://example.com"


def test_start_scrolling_default_speed():
    # when no speed given, default should be 250 px/s
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_down")
    assert runner.state.auto_scroll == "down"
    assert runner.state.scroll_speed == 250


def test_scroll_up_updates_scroll_y():
    ctx = _DummyCtx()
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.run("scroll_up 120")
    # scroll up uses negative delta so scroll_y decreases
    assert runner.state.scroll_y == -120


def test_scroll_speed_parsed():
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_down 600")
    assert runner.state.auto_scroll == "down"
    assert runner.state.scroll_speed == 600


def test_autoscroll_stop():
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_up 300")
    assert runner.state.auto_scroll == "up"
    runner.run("stop_scrolling")
    assert runner.state.auto_scroll is None


def test_click_out_resets_flag():
    # Page that will report "in text box" only on first call
    class _ClickPage(_DummyPage):
        def __init__(self):
            super().__init__()
            self._first = True

        def evaluate(self, script, *_args):
            if "return ['input'" in script:
                if self._first:
                    self._first = False
                    return True  # initially inside textbox
                return False  # after blur, no longer inside
            return 0

    ctx = _DummyCtx()
    ctx.pages = [_ClickPage()]
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.state.in_textbox = True
    runner.run("click_out")
    assert runner.state.in_textbox is False

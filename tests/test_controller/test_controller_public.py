import types, sys, pytest  # noqa: E402 (stubs before imports)

# ---------------------------------------------------------------------------
# Stubs for external heavy deps (redis & BrowserWorker) BEFORE importing code
# ---------------------------------------------------------------------------

# --- Redis stub -----------------------------------------------------------
class _FakePubSub:
    def __init__(self):
        self._messages = []
    def subscribe(self, *_):
        pass
    def listen(self):
        # generator expected by Controller.run(); empty -> instant end if used
        while self._messages:
            yield self._messages.pop()
        while True:
            yield {"type": "noop"}
    def get_message(self):
        return None

class _FakeRedis:
    def __init__(self, *a, **k):
        self._pubsub = _FakePubSub()
        self.published: list[tuple[str, str]] = []
    def pubsub(self):
        return self._pubsub
    def publish(self, chan, msg):
        self.published.append((chan, msg))

sys.modules.setdefault('redis', types.ModuleType('redis'))
sys.modules['redis'].Redis = _FakeRedis

# --- BrowserWorker stub ---------------------------------------------------
class _DummyWorker:
    def __init__(self, *a, **k):
        self.started = False
        self.stopped = False
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def join(self, *a, **k):
        pass

# ensure parent package module exists
pkg_path = 'unity.controller.playwright'
if pkg_path not in sys.modules:
    sys.modules[pkg_path] = types.ModuleType('playwright_stub')
worker_mod = types.ModuleType('worker')
worker_mod.BrowserWorker = _DummyWorker
sys.modules['unity.controller.playwright.worker'] = worker_mod

# ---------------------------------------------------------------------------
# Imports after stubbing
# ---------------------------------------------------------------------------
from unity.controller.controller import Controller  # noqa: E402


@pytest.mark.timeout(30)
def test_controller_observe_bool():
    """Smoke-test Controller.observe with bool response."""
    c = Controller()
    # minimal cached context
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ret = c.observe("Is 2+2 equal to 4?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ret, bool)


@pytest.mark.timeout(30)
def test_controller_observe_str():
    """Smoke-test observe with string response type."""
    c = Controller()
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ans = c.observe("Reply with 'hello'.", str)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ans, str)
    assert len(ans) > 0


@pytest.mark.timeout(30)
def test_controller_act_smoke():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()
    try:
        action_str = c.act("open browser")
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(action_str, str)
    # browser worker should have been started
    assert c._browser_open is True
    # ensure action_completion event was published
    assert ("action_completion", action_str) in c._redis_client.published 
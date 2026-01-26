from __future__ import annotations

import json
from typing import Any, Dict

import pytest


@pytest.fixture()
def csb_http(monkeypatch):
    """Fixture to mock unity.utils.http.get/post for CodeSandbox service tests.

    Usage:
        cfg = csb_http()
        cfg.set_get_json({"items": ["a.txt"]}, status=200)
        cfg.set_post_status(200)
    """

    class _Resp:
        def __init__(
            self,
            status_code: int = 200,
            payload: Any | None = None,
            content: bytes | None = None,
        ):
            self.status_code = status_code
            self._payload = payload
            self.content = content or (
                json.dumps(payload or {}).encode("utf-8")
                if payload is not None
                else b""
            )
            self.text = self.content.decode("utf-8", errors="ignore")

        def json(self) -> Any:
            return self._payload

    state: Dict[str, Any] = {
        "get": _Resp(200, {}),
        "post": _Resp(200, {}),
        "get_calls": [],
        "post_calls": [],
    }

    def _get(
        url: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        timeout: int = 30,
        *,
        raise_for_status: bool | None = None,
    ):
        state["get_calls"].append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
                "raise_for_status": raise_for_status,
            },
        )
        return state["get"]

    def _post(
        url: str,
        json: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        timeout: int = 30,
        *,
        raise_for_status: bool | None = None,
    ):
        state["post_calls"].append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
                "raise_for_status": raise_for_status,
            },
        )
        return state["post"]

    # Patch at both module level and in the adapter module (which imports them directly)
    monkeypatch.setattr("unify.utils.http.get", _get, raising=True)
    monkeypatch.setattr("unify.utils.http.post", _post, raising=True)
    monkeypatch.setattr(
        "unity.file_manager.filesystem_adapters.codesandbox_adapter.http.get",
        _get,
        raising=True,
    )
    monkeypatch.setattr(
        "unity.file_manager.filesystem_adapters.codesandbox_adapter.http.post",
        _post,
        raising=True,
    )

    class Cfg:
        def set_get_json(
            self,
            payload: Any,
            status: int = 200,
            content: bytes | None = None,
        ):
            state["get"] = _Resp(status, payload, content)

        def set_post_json(self, payload: Any, status: int = 200):
            state["post"] = _Resp(status, payload)

        def set_post_status(self, status: int):
            state["post"] = _Resp(status, {})

        @property
        def get_calls(self):
            return list(state["get_calls"])  # copy

        @property
        def post_calls(self):
            return list(state["post_calls"])  # copy

    return Cfg()


@pytest.fixture()
def interact_urlopen(monkeypatch):
    """Fixture to mock urllib.request.urlopen for Interact adapter tests.

    Returns a configurator with .set(path_suffix, payload_dict) mapping.
    """

    class Resp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    mapping: Dict[str, Any] = {}

    def fake_urlopen(req, data=None, timeout=30):
        url = getattr(req, "full_url", getattr(req, "_full_url", ""))
        for key, payload in mapping.items():
            if key in url:
                return Resp(payload)
        return Resp({})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=True)

    class Cfg:
        def set(self, url_contains: str, payload: Any):
            mapping[url_contains] = payload

    return Cfg()

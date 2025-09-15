from __future__ import annotations

import os
import threading
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter


# Single shared HTTP session for this process. We deliberately do not
# customize pool_maxsize or adapters so that the underlying defaults apply.
_SESSION: Optional[requests.Session] = None
_LOCK = threading.Lock()

# Default timeout (seconds) for backend requests issued via this helper.
# Can be overridden per-call and via env UNITY_HTTP_TIMEOUT_SECONDS.
try:
    _DEFAULT_TIMEOUT: float = float(os.environ.get("UNITY_HTTP_TIMEOUT_SECONDS", "60"))
except Exception:
    _DEFAULT_TIMEOUT = 60.0

# Connection pool size (per host). Defaults to 512; overridable via env.
try:
    _POOL_SIZE: int = int(os.environ.get("UNITY_HTTP_POOL_MAXSIZE", "512"))
except Exception:
    _POOL_SIZE = 512


def get_session() -> requests.Session:
    """Return a process-wide requests.Session for connection reuse."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
            # Mount adapters with enlarged connection pools for both HTTP and HTTPS.
            adapter = HTTPAdapter(
                pool_connections=_POOL_SIZE,
                pool_maxsize=_POOL_SIZE,
                pool_block=False,
            )
            _SESSION.mount("http://", adapter)
            _SESSION.mount("https://", adapter)
    return _SESSION


def request(
    method: str,
    url: str,
    *,
    timeout: Union[float, tuple[float, float], None] = None,
    **kwargs,
) -> requests.Response:
    """Dispatch an HTTP request via the shared Session.

    Parameters
    ----------
    method : str
        HTTP method (e.g., "GET", "POST").
    url : str
        Absolute URL.
    timeout : float | (connect, read) | None
        Per-call timeout override. When None, uses default.
    **kwargs
        Forwarded to requests.Session.request.
    """
    sess = get_session()
    eff_timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
    return sess.request(method, url, timeout=eff_timeout, **kwargs)


def get(
    url: str,
    *,
    timeout: Union[float, tuple[float, float], None] = None,
    **kwargs,
) -> requests.Response:
    return request("GET", url, timeout=timeout, **kwargs)


def post(
    url: str,
    *,
    timeout: Union[float, tuple[float, float], None] = None,
    **kwargs,
) -> requests.Response:
    return request("POST", url, timeout=timeout, **kwargs)


def patch(
    url: str,
    *,
    timeout: Union[float, tuple[float, float], None] = None,
    **kwargs,
) -> requests.Response:
    return request("PATCH", url, timeout=timeout, **kwargs)


def delete(
    url: str,
    *,
    timeout: Union[float, tuple[float, float], None] = None,
    **kwargs,
) -> requests.Response:
    return request("DELETE", url, timeout=timeout, **kwargs)

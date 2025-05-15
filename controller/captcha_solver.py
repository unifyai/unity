from __future__ import annotations

"""Helper wrappers around Anti-Captcha Python SDK used by the BrowserWorker.

This module purposefully isolates the dependency on *anticaptchaofficial* so it
can be mocked in unit tests and imported lazily in runtime code that actually
needs to solve a CAPTCHA.
"""

from typing import Optional
import base64
import logging
from urllib.parse import urlparse

from constants import ANTI_CAPTCHA_KEY

try:
    from anticaptchaofficial.recaptchav2proxyless import recaptchaV2Proxyless
    from anticaptchaofficial.hcaptchaproxyless import hCaptchaProxyless
    from anticaptchaofficial.imagecaptcha import imagecaptcha  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – caught during tests
    raise ImportError(
        "The 'anticaptchaofficial' package is required for CAPTCHA solving. "
        "Add it to requirements.txt and pip-install it."
    ) from exc

LOGGER = logging.getLogger("unity.captcha")

if not ANTI_CAPTCHA_KEY:
    LOGGER.warning(
        "ANTI_CAPTCHA_KEY env-var not set; CAPTCHA solving will be disabled."
    )

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

UNSUPPORTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "gmail.com",
    "google.co",  # wildcard for country TLDs handled later
}


def _extract_reg_domain(url: str) -> str:
    """Return eTLD+1 (registered domain) best-effort, e.g. accounts.google.com → google.com."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def solve_recaptcha(sitekey: str, url: str, *, invisible: bool = False, timeout: int = 120) -> Optional[str]:
    """Solve Google reCAPTCHA v2 (checkbox or invisible).

    Returns the solution `g-recaptcha-response` token, or *None* if solving
    failed.  Exceptions from the SDK are swallowed and logged so callers can
    decide what to do next (e.g. fall back to user prompt).
    """
    if not ANTI_CAPTCHA_KEY:
        return None

    # early exit on unsupported domain
    if _extract_reg_domain(url) in UNSUPPORTED_DOMAINS:
        LOGGER.warning("Anti-Captcha not supported on domain: %s", url)
        return None

    solver = recaptchaV2Proxyless()
    solver.set_key(ANTI_CAPTCHA_KEY)
    solver.set_website_url(url)
    solver.set_website_key(sitekey)
    solver.set_is_invisible(True)
    solver.set_soft_id(0)  # optional affiliate id
    # solver.set_task_timeout(timeout)

    try:
        token = solver.solve_and_return_solution()
        if token != 0:
            return token
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Error solving reCAPTCHA: %s", exc)
    return None


def solve_hcaptcha(sitekey: str, url: str, *, timeout: int = 120) -> Optional[str]:
    """Solve hCaptcha and return the token."""
    if not ANTI_CAPTCHA_KEY:
        return None

    # early exit on unsupported domain
    if _extract_reg_domain(url) in UNSUPPORTED_DOMAINS:
        LOGGER.warning("Anti-Captcha not supported on domain: %s", url)
        return None

    solver = hCaptchaProxyless()
    solver.set_key(ANTI_CAPTCHA_KEY)
    solver.set_website_url(url)
    solver.set_website_key(sitekey)
    # solver.set_task_timeout(timeout)

    try:
        token = solver.solve_and_return_solution()
        if token != 0:
            return token
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Error solving hCaptcha: %s", exc)
    return None


def solve_image(base64_png: str, *, timeout: int = 120) -> Optional[str]:
    """Solve image-based CAPTCHA (PNG -> text answer)."""
    if not ANTI_CAPTCHA_KEY:
        return None

    solver = imagecaptcha()
    solver.set_key(ANTI_CAPTCHA_KEY)
    # solver.set_task_timeout(timeout)

    try:
        text = solver.solve_and_return_solution(base64_png)
        if text != 0:
            return text
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Error solving image CAPTCHA: %s", exc)
    return None


# Convenience wrapper --------------------------------------------------------

def solve(captcha: dict, page_url: str) -> Optional[str]:
    """Generic dispatcher used by the worker.

    `captcha` is the payload returned by `detect_captcha` helper:
      {"type": "recaptcha_v2", "sitekey": "..."} etc.

    Returns a token / solution string or *None* if unsolved.
    """
    typ = captcha.get("type")
    if typ == "recaptcha_v2":
        return solve_recaptcha(captcha["sitekey"], page_url, invisible=captcha.get("invisible", False))
    if typ == "hcaptcha":
        return solve_hcaptcha(captcha["sitekey"], page_url)
    if typ == "image":
        handle = captcha.get("handle")
        if handle is None:
            return None
        try:
            png = handle.screenshot(type="png")
        except Exception:  # pragma: no cover
            return None
        b64 = base64.b64encode(png).decode()
        return solve_image(b64)
    return None 
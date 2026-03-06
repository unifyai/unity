"""
Interactive sandbox for testing and debugging computer-use primitives.

Provides direct access to ComputerPrimitives sessions (desktop, web, web-vm)
with visual coordinate accuracy diagnostics: red-dot overlays, side-by-side
screenshot comparisons, and LLM coordinate grounding tests.

No CodeActActor, no ConversationManager, no voice pipeline -- just raw session
control for fast iteration on computer-use capabilities.

Usage:
    .venv/bin/python -m sandboxes.computer_use.sandbox -p TestProject
    .venv/bin/python -m sandboxes.computer_use.sandbox -p TestProject --agent-url http://localhost:3000
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from sandboxes.utils import (
    activate_project,
    build_cli_parser,
    configure_sandbox_logging,
)

LG = logging.getLogger("computer_use_sandbox")

ANTHROPIC_MAX_EDGE = 1568

DEBUG_DIR = ROOT / ".debug_computer_use"


# ---------------------------------------------------------------------------
# Visual diagnostics
# ---------------------------------------------------------------------------


def _ensure_debug_dir() -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_DIR


def draw_click_marker(
    img: "Image.Image",
    x: int,
    y: int,
    color: str = "red",
    radius: int = 10,
    label: str = "",
) -> "Image.Image":
    """Draw a red dot on the image at (x, y). Returns a copy."""
    from PIL import ImageDraw

    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline="white",
        width=2,
    )
    if label:
        draw.text((x + radius + 4, y - radius), label, fill=color)

    w, h = img.size
    dim_text = f"{w}x{h}"
    draw.text((10, 10), dim_text, fill="yellow")

    return marked


def save_debug_image(img: "Image.Image", name: str) -> Path:
    """Save a debug image and return the path."""
    path = _ensure_debug_dir() / f"{name}.png"
    img.save(path)
    print(f"  [saved] {path}")
    return path


def log_dimensions(source: str, img: "Image.Image") -> None:
    """Print image dimensions and warn about API downscaling."""
    w, h = img.size
    print(f"  [screenshot] {source} -> {w}x{h}")
    if max(w, h) > ANTHROPIC_MAX_EDGE:
        scale = ANTHROPIC_MAX_EDGE / max(w, h)
        api_w, api_h = round(w * scale), round(h * scale)
        print(
            f"  [warning] Anthropic API will downscale to ~{api_w}x{api_h} "
            f"(max edge {ANTHROPIC_MAX_EDGE}px). LLM coordinates will be in "
            f"the downscaled space, not {w}x{h}!",
        )


# ---------------------------------------------------------------------------
# Diagnostic helpers (injected into REPL namespace)
# ---------------------------------------------------------------------------


async def _get_screenshot_logged(
    session: Any,
    source_label: str = "session",
) -> "Image.Image":
    """Take a screenshot and log its dimensions."""
    from PIL import Image

    b64 = await session.get_screenshot()
    if isinstance(b64, Image.Image):
        img = b64
    else:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    log_dimensions(source_label, img)
    return img


async def click_debug(
    session: Any,
    x: int,
    y: int,
    *,
    label: str = "click",
) -> dict:
    """Click with full visual diagnostics.

    1. Takes screenshot BEFORE click, draws red dot, saves
    2. Executes session.click(x, y)
    3. Takes screenshot AFTER click, saves
    4. Returns the click result
    """
    ts = time.strftime("%H%M%S")
    name = f"{ts}_{label}"

    before = await _get_screenshot_logged(session, "before_click")
    marked = draw_click_marker(before, x, y, label=f"({x},{y})")
    save_debug_image(marked, f"{name}_before")

    print(f"  [click] ({x}, {y}) ...")
    result = await session.click(x, y)

    await asyncio.sleep(0.3)
    after = await _get_screenshot_logged(session, "after_click")
    save_debug_image(after, f"{name}_after")

    print(f"  [done] Click ({x}, {y}). Debug images: {DEBUG_DIR}/{name}_*.png")
    return result


async def type_debug(
    session: Any,
    text: str,
    *,
    label: str = "type",
) -> dict:
    """Type text with before/after screenshots."""
    ts = time.strftime("%H%M%S")
    name = f"{ts}_{label}"

    print(f"  [type] '{text[:50]}{'...' if len(text) > 50 else ''}'")
    result = await session.type_text(text)

    await asyncio.sleep(0.3)
    after = await _get_screenshot_logged(session, "after_type")
    save_debug_image(after, f"{name}_after")
    return result


async def ask_coords(
    session: Any,
    element_description: str,
    *,
    label: str = "ask_coords",
) -> tuple[int, int]:
    """Ask the LLM for pixel coordinates of an element on the current page.

    Takes a screenshot, sends it to Claude asking for coordinates,
    draws a red dot at the predicted location, and saves the image.
    """
    from unity.common.llm_client import new_llm_client

    img = await _get_screenshot_logged(session, "ask_coords")
    w, h = img.size

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

    client = new_llm_client("claude-4.6-opus@anthropic", origin="computer_use_sandbox")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"This is a screenshot of a web page. The image dimensions are {w}x{h} pixels.\n\n"
                        f"I need the pixel coordinates of: {element_description}\n\n"
                        f"Look at the image carefully. Count pixels from the top-left corner (0,0). "
                        f"Reply with ONLY the coordinates in this exact format: x,y\n"
                        f"No explanation, just the two numbers separated by a comma."
                    ),
                },
            ],
        },
    ]

    response = await client.generate(messages=messages)
    raw = response.strip() if isinstance(response, str) else str(response).strip()

    match = re.search(r"(\d+)\s*,\s*(\d+)", raw)
    if not match:
        print(f"  [error] Could not parse coordinates from LLM response: {raw}")
        return (0, 0)

    cx, cy = int(match.group(1)), int(match.group(2))

    if max(w, h) > ANTHROPIC_MAX_EDGE:
        scale = ANTHROPIC_MAX_EDGE / max(w, h)
        api_w, api_h = round(w * scale), round(h * scale)
        if cx <= api_w and cy <= api_h:
            inv_scale = max(w, h) / ANTHROPIC_MAX_EDGE
            orig_cx, orig_cy = round(cx * inv_scale), round(cy * inv_scale)
            print(
                f"  [coords] LLM returned ({cx}, {cy}) — likely in downscaled "
                f"{api_w}x{api_h} space",
            )
            print(
                f"  [scaled] Mapped to viewport: ({orig_cx}, {orig_cy}) "
                f"(scale factor: {inv_scale:.3f})",
            )
            marked = draw_click_marker(
                img,
                orig_cx,
                orig_cy,
                color="blue",
                label=f"scaled({orig_cx},{orig_cy})",
            )
            marked = draw_click_marker(
                marked,
                cx,
                cy,
                color="red",
                label=f"raw({cx},{cy})",
            )
            ts = time.strftime("%H%M%S")
            save_debug_image(marked, f"{ts}_{label}")
            print(f"  [info] Red dot = raw LLM coords, Blue dot = scaled to viewport")
            return (orig_cx, orig_cy)

    print(f"  [coords] LLM says ({cx}, {cy}) for '{element_description}' on {w}x{h}")
    ts = time.strftime("%H%M%S")
    marked = draw_click_marker(img, cx, cy, label=f"({cx},{cy})")
    save_debug_image(marked, f"{ts}_{label}")
    return (cx, cy)


async def test_click(
    session: Any,
    element_description: str,
    *,
    label: str = "test",
) -> bool:
    """Full click test: ask LLM for coords, click, verify."""
    print(f"\n{'='*60}")
    print(f"TEST: Click on '{element_description}'")
    print(f"{'='*60}")

    coords = await ask_coords(session, element_description, label=f"{label}_coords")
    x, y = coords

    if x == 0 and y == 0:
        print("  [SKIP] No valid coordinates from LLM")
        return False

    await click_debug(session, x, y, label=f"{label}_click")

    print("  [verify] Checking if click landed...")
    try:
        result = await session.observe(
            f"Did the click land on {element_description}? "
            f"Is it focused, highlighted, or activated? Answer YES or NO and briefly explain.",
        )
        result_str = str(result).strip()
        print(f"  [observe] {result_str}")

        passed = "yes" in result_str.lower()[:20]
        print(f"\n  {'PASS' if passed else 'FAIL'}: Click on '{element_description}'")
        print(f"{'='*60}\n")
        return passed
    except Exception as e:
        print(f"  [error] Observe failed: {e}")
        return False


async def compare_screenshots(
    session: Any,
    cp: Any,
) -> None:
    """Compare desktop vs web session screenshots side-by-side."""

    print("\n--- Screenshot Comparison ---")

    web_img = await _get_screenshot_logged(session, "session.get_screenshot()")

    try:
        desktop_img = await _get_screenshot_logged(
            cp.desktop,
            "desktop.get_screenshot()",
        )
    except Exception as e:
        print(f"  [skip] Desktop screenshot unavailable: {e}")
        desktop_img = None

    ts = time.strftime("%H%M%S")
    save_debug_image(web_img, f"{ts}_compare_web")
    if desktop_img:
        save_debug_image(desktop_img, f"{ts}_compare_desktop")

        ww, wh = web_img.size
        dw, dh = desktop_img.size
        print(f"\n  Web session viewport: {ww}x{wh}")
        print(f"  Desktop full display: {dw}x{dh}")
        if (ww, wh) != (dw, dh):
            print(
                f"  DIFFERENT coordinate spaces! A click at (x,y) means "
                f"different things in each.",
            )
        else:
            print(f"  Same dimensions (but desktop includes browser chrome).")
    print("---\n")


def screenshot(
    session_or_img: Any,
    name: str = "screenshot",
) -> None:
    """Synchronous helper to save a PIL Image or take+save a screenshot."""
    from PIL import Image

    if isinstance(session_or_img, Image.Image):
        save_debug_image(session_or_img, name)
    else:
        print(
            "  [hint] Use: img = await session.get_screenshot(); screenshot(img, 'name')",
        )


# ---------------------------------------------------------------------------
# Low-level session helpers for diagnostics
# ---------------------------------------------------------------------------


def _inner_session(session: Any) -> Any:
    """Unwrap a WebSessionHandle or _ComputerNamespace to get the underlying ComputerSession.

    For WebSessionHandle: returns session._session directly.
    For _ComputerNamespace (cp.desktop): resolves via the backend's cached sessions.
    For a raw ComputerSession: returns it as-is.
    """
    # WebSessionHandle has _session
    inner = getattr(session, "_session", None)
    if inner is not None:
        return inner
    # _ComputerNamespace has _owner (a ComputerPrimitives) and _mode
    owner = getattr(session, "_owner", None)
    mode = getattr(session, "_mode", None)
    if owner is not None and mode is not None:
        backend = getattr(owner, "_backend", None)
        if backend is not None:
            sessions = getattr(backend, "_sessions", {})
            cached = sessions.get(mode)
            if cached is not None:
                return cached
    return session


async def js_eval(session: Any, expression: str) -> Any:
    """Run arbitrary JavaScript in the browser page via the /eval endpoint.

    Returns the evaluated result. The expression should be a string that
    JS can evaluate (e.g., a function call, JSON.stringify, etc).
    """
    inner = _inner_session(session)
    response = await inner._request("POST", "/eval", {"expression": expression})
    return response.get("result")


async def viewport_info(session: Any) -> dict:
    """Get comprehensive viewport/coordinate space diagnostic info.

    Returns playwright viewport, JS window metrics, and raw screenshot dims.
    """
    inner = _inner_session(session)
    response = await inner._request("POST", "/viewport-info", {})
    return response


async def diagnose_coordinates(session: Any) -> None:
    """Full coordinate space diagnostic — the definitive mismatch detector.

    Compares screenshot pixel coordinates with CSS viewport coordinates,
    checks DPR, and identifies the Email field in both coordinate spaces.
    """
    print("\n" + "=" * 60)
    print("COORDINATE SPACE DIAGNOSTIC")
    print("=" * 60)

    # 1. Viewport info from the server
    print("\n--- Step 1: Viewport & Screenshot Dimensions ---")
    try:
        info = await viewport_info(session)
        print(f"  Mode:              {info.get('mode')}")
        print(f"  Playwright VP:     {info.get('playwrightViewport')}")
        js = info.get("jsViewport", {})
        print(f"  JS innerWidth:     {js.get('innerWidth')}")
        print(f"  JS innerHeight:    {js.get('innerHeight')}")
        print(f"  JS outerWidth:     {js.get('outerWidth')}")
        print(f"  JS outerHeight:    {js.get('outerHeight')}")
        print(f"  JS DPR:            {js.get('devicePixelRatio')}")
        print(f"  JS clientWidth:    {js.get('clientWidth')}")
        print(f"  JS clientHeight:   {js.get('clientHeight')}")
        print(f"  screen:            {js.get('screenWidth')}x{js.get('screenHeight')}")
        print(
            f"  screenAvail:       {js.get('screenAvailWidth')}x{js.get('screenAvailHeight')}",
        )
        raw = info.get("rawScreenshotDims", {})
        rsc = info.get("rescaledScreenshotDims", {})
        print(f"  Raw screenshot:    {raw.get('width')}x{raw.get('height')}")
        print(f"  Rescaled (÷DPR):   {rsc.get('width')}x{rsc.get('height')}")

        dpr = js.get("devicePixelRatio", 1)
        iw, ih = js.get("innerWidth"), js.get("innerHeight")
        sw, sh = raw.get("width"), raw.get("height")
        if sw and iw and dpr:
            expected_raw_w = round(iw * dpr)
            expected_raw_h = round(ih * dpr)
            match_w = abs(sw - expected_raw_w) <= 2
            match_h = abs(sh - expected_raw_h) <= 2
            print(f"\n  Expected raw (innerW*DPR): {expected_raw_w}x{expected_raw_h}")
            print(f"  Actual raw:                {sw}x{sh}")
            if match_w and match_h:
                print(
                    f"  ✅ Screenshot matches innerWidth×DPR (coordinate spaces aligned)",
                )
            else:
                print(f"  ❌ MISMATCH! Screenshot does NOT match innerWidth×DPR")
                print(
                    f"     Delta: width={sw - expected_raw_w}, height={sh - expected_raw_h}",
                )
    except Exception as e:
        print(f"  [error] viewport_info failed: {e}")
        info = {}

    # 2. Find the Email field's bounding box in CSS coordinates
    print("\n--- Step 2: Email Field CSS Bounding Box ---")
    try:
        email_info = await js_eval(
            session,
            """
            (() => {
                const el = document.querySelector('input[type=email], input[placeholder*=Email], input[placeholder*=email]');
                if (!el) return { found: false };
                const r = el.getBoundingClientRect();
                return {
                    found: true,
                    tag: el.tagName,
                    placeholder: el.placeholder,
                    css_left: Math.round(r.left),
                    css_top: Math.round(r.top),
                    css_right: Math.round(r.right),
                    css_bottom: Math.round(r.bottom),
                    css_width: Math.round(r.width),
                    css_height: Math.round(r.height),
                    css_centerX: Math.round(r.left + r.width / 2),
                    css_centerY: Math.round(r.top + r.height / 2),
                };
            })()
        """,
        )
        if email_info and email_info.get("found"):
            print(
                f"  Found: <{email_info['tag']}> placeholder=\"{email_info.get('placeholder')}\"",
            )
            print(
                f"  CSS bounding box: ({email_info['css_left']},{email_info['css_top']}) to ({email_info['css_right']},{email_info['css_bottom']})",
            )
            print(
                f"  CSS center:       ({email_info['css_centerX']}, {email_info['css_centerY']})",
            )
            print(
                f"  CSS size:         {email_info['css_width']}x{email_info['css_height']}",
            )
        else:
            print("  [warn] No email input field found on page")
            email_info = {}
    except Exception as e:
        print(f"  [error] Email field lookup failed: {e}")
        email_info = {}

    # 3. Check elementFromPoint at LLM coordinates vs CSS coordinates
    print("\n--- Step 3: elementFromPoint Check ---")
    coords_to_check = [(480, 412, "LLM coords"), (520, 336, "Magnitude coords")]
    if email_info and email_info.get("found"):
        cx, cy = email_info["css_centerX"], email_info["css_centerY"]
        coords_to_check.append((cx, cy, "CSS center of Email"))

    for x, y, label in coords_to_check:
        try:
            result = await js_eval(
                session,
                f"""
                (() => {{
                    const el = document.elementFromPoint({x}, {y});
                    if (!el) return {{ found: false, x: {x}, y: {y} }};
                    return {{
                        found: true,
                        x: {x}, y: {y},
                        tag: el.tagName,
                        id: el.id,
                        className: el.className?.toString().substring(0, 60) || '',
                        placeholder: el.placeholder || '',
                        textContent: el.textContent?.substring(0, 40) || '',
                    }};
                }})()
            """,
            )
            tag = result.get("tag", "?")
            placeholder = result.get("placeholder", "")
            text = result.get("textContent", "")
            is_email = (
                "email" in placeholder.lower()
                or "email" in (result.get("id", "") or "").lower()
            )
            marker = "✅ EMAIL FIELD" if is_email else "❌ not email"
            print(
                f'  ({x:>4},{y:>4}) [{label:>20}] → <{tag}> placeholder="{placeholder}" {marker}',
            )
            if not is_email and text:
                print(f'         textContent: "{text[:40]}"')
        except Exception as e:
            print(f"  ({x},{y}) [{label}] → error: {e}")

    # 4. Take screenshot and compare LLM position with CSS position
    print("\n--- Step 4: Screenshot Coordinate Comparison ---")
    img = await _get_screenshot_logged(session, "diagnostic")
    sw, sh = img.size
    if email_info and email_info.get("found"):
        css_cx, css_cy = email_info["css_centerX"], email_info["css_centerY"]
        dpr = info.get("jsViewport", {}).get("devicePixelRatio", 1)

        marked = draw_click_marker(img, 480, 412, color="red", label="LLM(480,412)")
        marked = draw_click_marker(
            marked,
            css_cx,
            css_cy,
            color="green",
            label=f"CSS({css_cx},{css_cy})",
        )

        if dpr != 1:
            raw_cx = round(css_cx * dpr)
            raw_cy = round(css_cy * dpr)
            if raw_cx != css_cx or raw_cy != css_cy:
                marked = draw_click_marker(
                    marked,
                    raw_cx,
                    raw_cy,
                    color="blue",
                    label=f"RAW({raw_cx},{raw_cy})",
                )

        ts = time.strftime("%H%M%S")
        save_debug_image(marked, f"{ts}_diagnostic_comparison")
        print(
            f"  Red dot:   LLM coordinates (480, 412) — where LLM thinks Email is on screenshot",
        )
        print(
            f"  Green dot: CSS coordinates ({css_cx}, {css_cy}) — where Email actually is in CSS",
        )
        if abs(480 - css_cx) > 5 or abs(412 - css_cy) > 5:
            print(f"\n  ❌ COORDINATE MISMATCH CONFIRMED")
            print(f"     ΔX = {css_cx - 480:+d}px, ΔY = {css_cy - 412:+d}px")
            print(f"     Screenshot pixel (480,412) ≠ CSS pixel (480,412)")
        else:
            print(f"\n  ✅ Coordinates are aligned (within 5px)")

    print("\n" + "=" * 60)
    print("END DIAGNOSTIC")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


class AsyncREPL:
    """Simple async-aware REPL for interactive session testing."""

    def __init__(self, namespace: dict):
        self._namespace = namespace
        self._loop = asyncio.get_event_loop()

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("COMPUTER USE SANDBOX")
        print("=" * 60)
        print(f"Debug images saved to: {DEBUG_DIR}/")
        print()
        print("Available objects:")
        print("  cp            - ComputerPrimitives instance")
        print("  agent_url     - Agent service URL")
        print()
        print("--- Web-VM mode (browser on VM desktop) ---")
        print("  session = await cp.web.new_session(visible=True)")
        print("  await session.navigate('https://example.com')")
        print("  await session.click(x, y)")
        print("  img = await session.get_screenshot()")
        print("  screenshot(img, 'my_page')")
        print()
        print("--- Desktop mode (full VM desktop via noVNC) ---")
        print("  await cp.desktop.act('Open the terminal')")
        print("  await cp.desktop.click(x, y)")
        print("  img = await cp.desktop.get_screenshot()")
        print("  screenshot(img, 'desktop')")
        print()
        print("Visual diagnostics (pass session or cp.desktop):")
        print("  await click_debug(session, x, y, label='test')")
        print("  coords = await ask_coords(session, 'the Submit button')")
        print("  await test_click(session, 'the Email input field')")
        print("  await compare_screenshots(session, cp)")
        print()
        print("Coordinate debug (pass session or cp.desktop):")
        print(
            "  await diagnose_coordinates(session)        # full coordinate diagnostic",
        )
        print(
            "  await viewport_info(session)               # viewport/DPR/screenshot dims",
        )
        print(
            "  await js_eval(session, 'document.title')   # run arbitrary JS in browser",
        )
        print()
        print("Type Python expressions. Use 'await' for async calls.")
        print("Ctrl+D to exit.\n")

        import ast

        while True:
            try:
                line = input(">>> ")
            except EOFError:
                print("\nExiting.")
                break
            except KeyboardInterrupt:
                print()
                continue

            if not line.strip():
                continue

            while line.rstrip().endswith(":") or line.rstrip().endswith("\\"):
                try:
                    continuation = input("... ")
                    line += "\n" + continuation
                except EOFError:
                    break
                except KeyboardInterrupt:
                    line = ""
                    break

            if not line.strip():
                continue

            has_await = "await " in line

            try:
                if has_await:
                    stripped = line.strip()
                    is_bare_await = stripped.startswith("await ")
                    if is_bare_await and "=" not in stripped.split("await", 1)[0]:
                        wrapped = (
                            "async def __repl_wrapper__(__ns__):\n"
                            f"    __result__ = {stripped}\n"
                            "    return __result__"
                        )
                        exec(compile(wrapped, "<repl>", "exec"), self._namespace)
                        coro = self._namespace["__repl_wrapper__"](self._namespace)
                        result = self._loop.run_until_complete(coro)
                        if result is not None:
                            print(repr(result))
                    else:
                        wrapped = (
                            "async def __repl_wrapper__(__ns__):\n"
                            + "\n".join("    " + l for l in line.splitlines())
                            + "\n    __ns__.update({k: v for k, v in locals().items() if k != '__ns__' and not k.startswith('__')})"
                        )
                        exec(compile(wrapped, "<repl>", "exec"), self._namespace)
                        coro = self._namespace["__repl_wrapper__"](self._namespace)
                        self._loop.run_until_complete(coro)
                else:
                    try:
                        tree = ast.parse(line, mode="eval")
                        compiled = compile(
                            ast.Expression(body=tree.body),
                            "<repl>",
                            "eval",
                        )
                        result = eval(compiled, self._namespace)
                        if asyncio.iscoroutine(result):
                            result = self._loop.run_until_complete(result)
                        if result is not None:
                            print(repr(result))
                    except SyntaxError:
                        exec(compile(line, "<repl>", "exec"), self._namespace)

            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from unity.function_manager.primitives import ComputerPrimitives

    parser = build_cli_parser("Computer Use Test Sandbox")
    parser.add_argument(
        "--agent-url",
        type=str,
        default="http://localhost:3000",
        help="URL of the Magnitude agent service (Docker container).",
    )
    args = parser.parse_args()

    activate_project(args.project_name, args.overwrite)
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_computer_use_sandbox.txt",
    )

    agent_url = args.agent_url

    from sandboxes.conversation_manager.desktop_bootstrap import (
        try_auto_bootstrap_desktop,
    )

    print(f"\nBootstrapping desktop container...")
    boot_result = try_auto_bootstrap_desktop(
        repo_root=ROOT,
        agent_server_url=agent_url,
        progress=lambda msg: print(f"  {msg}"),
    )
    if not boot_result.ok:
        print(f"\n  Desktop bootstrap failed: {boot_result.summary}")
        print("  Start the container manually or check Docker.")
        sys.exit(1)
    print(f"  {boot_result.summary}")

    import webbrowser

    from sandboxes.conversation_manager.desktop_bootstrap import _desktop_novnc_url

    novnc_url = _desktop_novnc_url()
    print(f"  noVNC: {novnc_url}")
    webbrowser.open(novnc_url)

    print(f"\nAgent service: {agent_url}")
    print(f"Debug output:  {DEBUG_DIR}/")

    cp = ComputerPrimitives(
        computer_mode="magnitude",
        agent_server_url=agent_url,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    namespace = {
        "cp": cp,
        "agent_url": agent_url,
        "asyncio": asyncio,
        "click_debug": click_debug,
        "type_debug": type_debug,
        "ask_coords": ask_coords,
        "test_click": test_click,
        "compare_screenshots": compare_screenshots,
        "screenshot": screenshot,
        "draw_click_marker": draw_click_marker,
        "save_debug_image": save_debug_image,
        "log_dimensions": log_dimensions,
        "js_eval": js_eval,
        "viewport_info": viewport_info,
        "diagnose_coordinates": diagnose_coordinates,
        "DEBUG_DIR": DEBUG_DIR,
    }

    repl = AsyncREPL(namespace)
    try:
        repl.run()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        print("\nCleaning up...")
        web_factory = getattr(cp, "_web_factory", None)
        if web_factory:
            for handle in list(getattr(web_factory, "_handles", [])):
                if getattr(handle, "active", False):
                    try:
                        loop.run_until_complete(handle.stop())
                        print(f"  Stopped web session {handle.session_id}")
                    except Exception:
                        pass
        try:
            desktop_session = getattr(cp, "_desktop_session", None)
            if desktop_session:
                loop.run_until_complete(desktop_session.stop())
                print("  Stopped desktop session")
        except Exception:
            pass

        loop.close()

        from sandboxes.conversation_manager.desktop_bootstrap import (
            stop_desktop_container,
        )

        stop_desktop_container(progress=lambda msg: print(f"  {msg}"))
        print("Done.")


if __name__ == "__main__":
    main()

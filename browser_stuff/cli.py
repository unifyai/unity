"""
`python -m browser`                                   (or `python -m browser --overlay`)
----------------------------------------------------------------------
A tiny demo: opens the given URL in a headed Chromium window and (optionally)
turns on the element overlay until you press <ENTER>.
"""
from __future__ import annotations
import argparse, asyncio, sys

from core import Browser

# ---------------------------------------------------------------------------

async def _demo(url: str, overlay: bool = False) -> None:
    br = Browser()
    await br.start()
    await br.goto(url)
    if overlay:
        await br.enable_overlay()
        await asyncio.get_running_loop().run_in_executor(None, input)
    await br.close()

# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser("browserlib overlay demo")
    p.add_argument("url", nargs="?", default="https://example.com")
    p.add_argument("--overlay", action="store_true")
    args = p.parse_args()

    try:
        asyncio.run(_demo(args.url, overlay=args.overlay))
    except KeyboardInterrupt:
        sys.exit(130)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

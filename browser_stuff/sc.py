"""
state_demo.py
─────────────
Quick smoke-test for Browser.get_state().

• Opens Google, lets the overlay render,
• Grabs the current interactive elements + a screenshot,
• Writes the PNG to disk and pretty-prints the element summary.
"""

import asyncio, textwrap
from pathlib import Path

from core import Browser   # adjust import if running elsewhere


async def main() -> None:
    br = Browser()
    await br.start()
    await br.goto("https://www.google.com")
    await br.enable_overlay()
    await asyncio.sleep(0.2)
    await br.create_new_tab("https://www.google.com")

    elements, png = await br.get_state()
    Path("state_screenshot.png").write_bytes(png)

    # Nicely formatted dump
    print("\nDetected elements:")
    for i, meta in elements.items():
        label = meta["label"] or "(no label)"
        print(f"  {i:>3d}: {meta['tag']:<8} {meta['kind']:<6} {label}")

    print(
        textwrap.dedent(
            """
            -------------------------------------------------
            ✓ Screenshot saved to  state_screenshot.png
            ✓ Total elements found: {}
            -------------------------------------------------
            """.format(
                len(elements)
            )
        )
    )

    await br.close()


if __name__ == "__main__":
    asyncio.run(main())

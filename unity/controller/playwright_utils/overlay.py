"""
overlay.py – builds the JS helper that lives in `js_helper_template.js`

The heavy JavaScript that drives discovery & the on‑page overlay now
lives in its own file so it’s readable and editable without a forest of
escaped quotes.  At runtime we just read that template, drop the current
heuristics JSON into the `__HEURISTICS__` placeholder, and return the
final source string for injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

# path to the template JS (same directory as this file)
_JS_TEMPLATE = Path(__file__).with_name("js_helper_template.js")


def _make_js_helper(heur_list: List[Dict]) -> str:
    """
    Read `js_helper_template.js`, replace the `__HEURISTICS__` token with
    the actual heuristics JSON, and return the ready‑to‑inject script.
    """
    template = _JS_TEMPLATE.read_text(encoding="utf‑8")
    heur_json = json.dumps(heur_list, ensure_ascii=False)
    return template.replace("__HEURISTICS__", heur_json)

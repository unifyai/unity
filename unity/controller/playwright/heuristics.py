# browser/heuristics.py
"""Central registry for discovery *heuristics*.

Edit the ``HEURISTICS`` list to control which elements the Browser treats
as interactive.  Tweaking *only* this file is enough – the JS helper is
rebuilt at runtime.
"""
from __future__ import annotations
from typing import List, TypedDict, Dict


class Heuristic(TypedDict, total=False):
    name: str  # unique key
    selector: str  # CSS selector pre‑filter
    kind: str  # click | input | exclude | other
    filter_js: str  # JS fn body: (el) => boolean   – optional
    label_js: str  # JS fn body: (el) => string    – optional


# ---------------------------------------------------------------------------#
#  HEURISTICS – generic → site‑specific wins by order                        #
# ---------------------------------------------------------------------------#

HEURISTICS: List[Heuristic] = [
    # ─────────────────────────────────────────────
    # Google Docs rich-text editor surface (site-specific)
    # ─────────────────────────────────────────────
    {
        "name": "gdocs-editor-surface",
        "selector": ".kix-appview-editor, .kix-canvas-tile-content, .docs-texteventtarget",
        "kind": "input",
        "filter_js": (
            "return location.hostname.includes('docs.google.com') && "
            "el.offsetWidth > 0 && el.offsetHeight > 0;"
        ),
        "label_js": "return 'Google Docs Input';",
    },
    # ─────────────────────────────────────────────
    # ClickUp SSO Google button (site-specific)
    # ─────────────────────────────────────────────
    {
        "name": "clickup-sso-google",
        "selector": "div#google_auth_button.login-page-new__main-form-google",
        "kind": "click",
        "filter_js": (
            "return location.hostname.includes('clickup.com')"
        ),
        "label_js": "return 'Continue with Google';",
    }, 
    # ─────────────────────────────────────────────
    # 1. TRACKING "BEACON" ANCHORS  (exclude)
    #    Many e‑commerce grids inject a 0×0 or 1×1 <a>
    #    solely for click analytics.  Skip them so they
    #    don't steal the ID.
    # ─────────────────────────────────────────────
    {
        "name": "tiny-beacon",
        "selector": "a[href]",
        "kind": "exclude",
        "filter_js": (
            "const r = el.getBoundingClientRect();"
            "return r.width < 4 && r.height < 4;"  # <= 3 px either side
        ),
    },
    # ─────────────────────────────────────────────
    # 2. PROMOTE THE FIRST VISIBLE CHILD OF THOSE
    #    TINY ANCHORS  (generic card thumbnail)
    # ─────────────────────────────────────────────
    {
        "name": "clickable-child",
        # any element *inside* a link / button
        "selector": "a[href] *, button *",
        "kind": "click",
        "filter_js": (
            # must be visually substantial
            "const r = el.getBoundingClientRect();"
            "if (r.width < 60 || r.height < 60) return false;"
            # parent must be the zero‑rect beacon we just excluded
            "const p = el.parentElement;"
            "if (!p) return false;"
            "const pr = p.getBoundingClientRect();"
            "return pr.width < 4 && pr.height < 4;"
        ),
        "label_js": ("return (el.alt || el.innerText || '').trim().slice(0, 40);"),
    },
    # ─────────────────────────────────────────────
    # 3. TAG‑LEVEL CLICKABLES  (original rule)
    # ─────────────────────────────────────────────
    {
        "name": "tag-clickable",
        "selector": (
            "a[href],button,[role='button'],summary,*[href],"
            "details,summary,label,option,optgroup,fieldset,legend"
        ),
        "kind": "click",
    },
    {
        "name": "heading-in-link",
        # any <h1>–<h6> that sits *inside* a link
        "selector": "a[href] h1, a[href] h2, a[href] h3, a[href] h4, a[href] h5, a[href] h6",
        "kind": "click",
        "label_js": "return (el.innerText || '').trim().slice(0, 80);",
    },
    # ─────────────────────────────────────────────
    # 4. FORM INPUTS
    # ─────────────────────────────────────────────
    {
        "name": "tag-input",
        "selector": "input:not([type='hidden']),textarea,select",
        "kind": "input",
    },
    {
        "name": "tabindex-focusable",
        "selector": "*[tabindex]",  # any element that declares a tabindex
        "kind": "click",
        "filter_js": (
            # // 1. Positive (or zero) tabindex means it's keyboard-focusable
            "const tab = parseInt(el.getAttribute('tabindex') || '-1', 10);"
            "if (Number.isNaN(tab) || tab < 0) return false;"
            # // 2. Skip elements that are already covered by stronger rules
            "const tag = el.tagName.toLowerCase();"
            "if (['a','button','input','select','textarea','label','summary','details'].includes(tag))"
            "  return false;"
            # // 3. Must be visible and occupy some area
            "const r = el.getBoundingClientRect();"
            "if (!r.width || !r.height) return false;"
            # // 4. If it merely wraps a real control, let the child win
            "if (el.querySelector('a[href],button,input,select,textarea,[role=\"button\"]'))"
            "  return false;"
            "return true;"
        ),
        "label_js": "return (el.innerText || '').trim().slice(0, 60);",
    },
    {
        "name": "scrollable-container",
        "selector": "*",
        "kind": "scroll",
        "filter_js": (
            # it must create its own scroll bar
            "const st = getComputedStyle(el);"
            "if (!['scroll','auto'].includes(st.overflowY)) return false;"
            # …and actually be able to scroll
            "if (el.scrollHeight <= el.clientHeight + 4) return false;"
            # visible & not microscopic
            "const r = el.getBoundingClientRect();"
            "return r.width > 60 && r.height > 60;"
        ),
        "label_js": (
            "return (el.getAttribute('aria-label') || el.title || "
            "        el.dataset.testid || '').trim().slice(0,60);"
        ),
    },
    # ─────────────────────────────────────────────
    # 5. CURSOR‑BASED CANDIDATES (allow list)
    # ─────────────────────────────────────────────
    {
        "name": "cursor-allow",
        "selector": "*",
        "kind": "click",
        "filter_js": (
            "const good=new Set(["
            "'pointer','grab','grabbing','move','zoom-in','zoom-out',"
            "'text','vertical-text','cell','copy','alias','crosshair',"
            "'col-resize','row-resize','e-resize','w-resize','n-resize','s-resize',"
            "'ne-resize','nw-resize','se-resize','sw-resize','ns-resize','ew-resize',"
            "'nesw-resize','nwse-resize','all-scroll'"
            "]);"
            "const skip=new Set(['svg','path','image','g','rect','circle','line','polyline','polygon']);"
            "if (skip.has(el.tagName.toLowerCase())) return false;"
            "const clickableAnc = "
            "  'a[href],button,summary,"
            '   [role="button"],[role="option"],[role^="menuitem"],'
            '   [role="tab"],[role="link"],[role="checkbox"],[role="radio"]\';'
            "if (el.closest(clickableAnc) && !el.matches(clickableAnc)) return false;"
            "return good.has(getComputedStyle(el).cursor);"
        ),
    },
    # ─────────────────────────────────────────────
    # 6. CURSOR‑BASED EXCLUDE (block list)
    # ─────────────────────────────────────────────
    {
        "name": "cursor-block",
        "selector": "*",
        "kind": "exclude",
        "filter_js": (
            "return new Set(["
            "'not-allowed','no-drop','wait','progress','initial','inherit'"
            "]).has(getComputedStyle(el).cursor);"
        ),
    },
    # ─────────────────────────────────────────────
    # 7. BASIC ARIA ROLES
    # ─────────────────────────────────────────────
    {
        "name": "role",
        "selector": "[role],[aria-role]",
        "kind": "click",
        "filter_js": (
            "const ok=new Set(["
            "'button','link','menuitem','menuitemradio','menuitemcheckbox','tab',"
            "'switch','slider','spinbutton','combobox','searchbox','textbox',"
            "'radio','checkbox','scrollbar','option'"
            "]);"
            "return ok.has(el.getAttribute('role')||el.getAttribute('aria-role'));"
        ),
    },
    # ─────────────────────────────────────────────
    # 8. CONTENT‑EDITABLE FIELDS
    # ─────────────────────────────────────────────
    {
        "name": "editable",
        "selector": "[contenteditable]:not([contenteditable='false'])",
        "kind": "input",
    },
    # ─────────────────────────────────────────────
    # 9. BOOTSTRAP / DROPDOWN HINTS
    # ─────────────────────────────────────────────
    {
        "name": "class-hint",
        "selector": (
            ".button,.dropdown-toggle,[data-toggle='dropdown'],"
            "*[data-index],[aria-haspopup='true']"
        ),
        "kind": "click",
    },
    # ─────────────────────────────────────────────
    # 10. INLINE HANDLER ATTRIBUTES
    # ─────────────────────────────────────────────
    {
        "name": "onclick-handler",
        "selector": "*[onclick],*[ondblclick],*[onmousedown],*[onmouseup]",
        "kind": "click",
    },
    # ─────────────────────────────────────────────
    # 11. DATA‑TEST IDS
    # ─────────────────────────────────────────────
    # {
    #     "name": "data-test-id",
    #     "selector": "*[data-testid],*[data-cy],*[data-test]",
    #     "kind": "click",
    # },
    # ─────────────────────────────────────────────
    # 12. LABEL TRACKER (BLOCKER)
    # ─────────────────────────────────────────────
    {
        "name": "label-tracker",
        "selector": ".label-tracker",
        "kind": "exclude",
    },
]


# ---------------------------------------------------------------------------#
# Helper – export active list for JS helper                                  #
# ---------------------------------------------------------------------------#
def export_for_js() -> List[Dict]:
    """Return heuristics in plain‑dict form (ready for json.dumps)."""
    return HEURISTICS.copy()

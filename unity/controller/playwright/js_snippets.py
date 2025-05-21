"""
All JavaScript helpers in one place so other modules
can `from .js_snippets import *`.
"""

# === element inspection =====================================================
ELEMENT_INFO_JS = """
el => {
  // ---- find out if element sits in an iframe -------------------------
  function isInsideIframe(node) {
    return node.ownerDocument.defaultView !== window.top;
  }

  // ---- climb ancestors for fixed / sticky / transform ----------------
  function hasFixedAncestor(node){
    while (node && node !== document.body && node !== document.documentElement){
      const p = getComputedStyle(node).position;
      if (p === 'fixed' || p === 'sticky') return true;
      node = node.parentElement;
    }
    return false;
  }

  // ---- compute rect relative to the *top* window ---------------------
  function rectInPage(element){
    const r = element.getBoundingClientRect();
    let left = r.left, top = r.top;
    let win  = element.ownerDocument.defaultView;
    while (win && win !== window.top && win.frameElement){
      const fr = win.frameElement.getBoundingClientRect();
      left += fr.left;
      top  += fr.top;
      win = win.parent;
    }
    return {left, top, width:r.width, height:r.height};
  }

  const rp = rectInPage(el);
  if (!rp.width || !rp.height) return null;        // hidden / 0‑size

  const tag = el.tagName.toLowerCase();
  const label =
        el.innerText.trim()                         ||
        ((tag === 'input' || tag === 'textarea') ?
             (el.value || el.placeholder || '') : '') ||
        el.getAttribute('aria-label')               ||
        el.getAttribute('alt')                      ||
        el.getAttribute('title')                    ||
        el.getAttribute('href')                     ||
        '<no label>';

  return {
    fixed  : hasFixedAncestor(el) || isInsideIframe(el),
    hover  : el.matches(':hover'),
    left   : rp.left,   // absolute page coords (no scroll subtraction)
    top    : rp.top,
    width  : rp.width,
    height : rp.height,
    label  : label,
  };
}
"""

# === smooth one-off scroll ===================================================
HANDLE_SCROLL_JS = r"""
  ({ delta, duration }) => {
    // pick scroll target: focused scrollable element or scrollable under center
    function isScrollable(el) {
      if (!el) return false;
      const cs = getComputedStyle(el);
      return el.scrollHeight > el.clientHeight + 2 &&
            (cs.overflowY === 'auto' || cs.overflowY === 'scroll');
    }
    function findScrollableAncestor() {
      let el = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
      while (el) {
        if (isScrollable(el)) return el;
        el = el.parentElement || el.host;
      }
      return null;
    }
    const focusEl = document.activeElement;
    const target = isScrollable(focusEl)
      ? focusEl
      : (findScrollableAncestor() || window);
    const y0 = target === window ? scrollY : target.scrollTop;
    const y1 = y0 + delta;
    const t0 = performance.now();
    const ease = p => (p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p);
    const step = t => {
      const p = Math.min(1, (t - t0) / duration);
      const pos = y0 + (y1 - y0) * ease(p);
      if (target === window) {
        window.scrollTo(0, pos);
      } else {
        target.scrollTop = pos;
      }
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
"""

# === continuous auto‑scroll ==================================================
AUTO_SCROLL_JS = r"""
  ({ dir, speed }) => {
    // stop any previous auto-scroll
    if (!window.__asStop) {
      window.__asStop = () => cancelAnimationFrame(window.__asId);
    }
    window.__asStop();
    if (dir === 'stop') return;
    // pick scroll target: focused scrollable or under center
    function isScrollable(el) {
      if (!el) return false;
      const cs = getComputedStyle(el);
      return el.scrollHeight > el.clientHeight + 2 &&
            (cs.overflowY === 'auto' || cs.overflowY === 'scroll');
    }
    function findScrollableAncestor() {
      let el = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
      while (el) {
        if (isScrollable(el)) return el;
        el = el.parentElement || el.host;
      }
      return null;
    }
    const focusEl = document.activeElement;
    const target = isScrollable(focusEl)
      ? focusEl
      : (findScrollableAncestor() || window);
    const sign = dir === 'down' ? 1 : -1;
    let last = performance.now();
    const step = t => {
      const dt = t - last;
      last = t;
      const delta = sign * speed * dt;
      if (target === window) {
        window.scrollBy(0, delta);
      } else {
        target.scrollTop += delta;
      }
      window.__asId = requestAnimationFrame(step);
    };
    window.__asId = requestAnimationFrame(step);
  }
"""

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
  const y0 = scrollY;
  const y1 = y0 + delta;
  const t0 = performance.now();

  const ease = (p) => (p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p);

  const step = (t) => {
    const p = Math.min(1, (t - t0) / duration);
    scrollTo(0, y0 + (y1 - y0) * ease(p));
    if (p < 1) requestAnimationFrame(step);
  };

  requestAnimationFrame(step);
}
"""

# === continuous auto‑scroll ==================================================
AUTO_SCROLL_JS = r"""
({ dir, speed }) => {
  if (!window.__asStop) {
    window.__asStop = () => cancelAnimationFrame(window.__asId);
  }
  window.__asStop();
  if (dir === "stop") return;

  const sign = dir === "down" ? 1 : -1;
  let last = performance.now();

  const step = (t) => {
    const dt = t - last;
    last = t;
    scrollBy(0, sign * speed * dt);
    window.__asId = requestAnimationFrame(step);
  };

  window.__asId = requestAnimationFrame(step);
}
"""

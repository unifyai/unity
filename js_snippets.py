"""
All JavaScript helpers in one place so other modules
can `from .js_snippets import *`.
"""

# === element inspection =====================================================
ELEMENT_INFO_JS = r"""
(el) => {
  function hasFixedAncestor(n){
    while (n && n !== document.body && n !== document.documentElement){
      const p = getComputedStyle(n).position;
      if (p === 'fixed' || p === 'sticky') return true;
      n = n.parentElement;
    }
    return false;
  }
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return {
    fixed  : hasFixedAncestor(el),
    hover  : el.matches(':hover'),
    vleft  : r.left,
    vtop   : r.top,
    left   : r.left + scrollX,
    top    : r.top  + scrollY,
    width  : r.width,
    height : r.height,
    label  : (el.innerText.trim()           ||
              el.getAttribute('aria-label') ||
              el.getAttribute('alt')        ||
              el.getAttribute('title')      ||
              el.getAttribute('href')       ||
              '<no label>')
  };
}
"""

# === overlay painter ========================================================
UPDATE_OVERLAY_JS = r"""
(boxes) => {
  const idPage  = "__pw_rootPage__";
  const idFixed = "__pw_rootFixed__";

  let pageRoot  = document.getElementById(idPage);
  let fixedRoot = document.getElementById(idFixed);

  if (!pageRoot) {
    pageRoot  = Object.assign(document.createElement("div"), {
      id: idPage,
      style:
        "position:absolute;left:0;top:0;pointer-events:none;z-index:2147483646",
    });
    fixedRoot = Object.assign(document.createElement("div"), {
      id: idFixed,
      style:
        "position:fixed;inset:0;pointer-events:none;z-index:2147483647",
    });
    document.body.append(pageRoot, fixedRoot);
  } else {
    pageRoot.replaceChildren();
    fixedRoot.replaceChildren();
  }

  boxes.forEach(({ i, fixed, x, y, px, py, w, h }, n, { length }) => {
    const hue = (360 * n) / length;
    const div = document.createElement("div");
    div.style.cssText = `
      position:${fixed ? "fixed" : "absolute"};
      left:${fixed ? px : x}px; top:${fixed ? py : y}px;
      width:${w}px; height:${h}px;
      outline:2px solid hsl(${hue} 100% 50%);
      background:hsl(${hue} 100% 50%/.12);
      font:700 12px/1 sans-serif;
      color:hsl(${hue} 100% 30%);
    `;
    const tag = document.createElement("span");
    tag.textContent = i;
    tag.style.cssText =
      "position:absolute;left:0;top:0;background:#fff;padding:0 2px";
    div.append(tag);
    (fixed ? fixedRoot : pageRoot).append(div);
  });
}
"""

# === smooth one‑off scroll ===================================================
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

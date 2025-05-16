// js_helper_template.js  – injected into every frame by core.py
(() => {
  if (window.__bl) return;         // already injected

  /*──────────── CONFIG ────────────*/
  const REFRESH_MS = 120;
  const VP_PAD     = 500;
  const HEUR       = __HEURISTICS__;           // ← filled-in by Python

  /* compile inline filter/label strings */
  HEUR.forEach(h => {
    if (h.filter_js) h.filter = new Function('el', h.filter_js);
    if (h.label_js)  h.label  = new Function('el', h.label_js);
  });
  const SPEC_SEL = HEUR
      .filter(h => h.selector && h.selector !== '*')
      .map(h => h.selector).join(',');

  /* tiny caches */
  let stCache = new Map(), rectCache = new Map();
  const css  = el => stCache.get(el) ?? (stCache.set(el, getComputedStyle(el)), stCache.get(el));
  const rect = el => rectCache.get(el) ?? (rectCache.set(el, el.getBoundingClientRect()), rectCache.get(el));

  /* helpers */
  const inView = r =>
    !(r.bottom < -VP_PAD || r.top > innerHeight + VP_PAD ||
      r.right  < -VP_PAD || r.left > innerWidth + VP_PAD);

  function isVisible(el) {
    const r = rect(el), cs = css(el);
    return r.width && r.height && cs.visibility !== 'hidden' && cs.display !== 'none';
  }

  function isTopMost(el) {
    const rs = el.getClientRects(); if (!rs.length) return false;
    const c  = rs[0]; if (!inView(c)) return false;
    const root = el.getRootNode();
    const top  = (root instanceof ShadowRoot ? root : document)
                   .elementFromPoint(c.left + c.width / 2, c.top + c.height / 2);
    for (let cur = top; cur; cur = cur.parentElement) if (cur === el) return true;
    return false;
  }

  /* full DOM scan */
  let nextId = 1;
  const giveId = el => el.dataset.blId ? +el.dataset.blId
                                       : +(el.dataset.blId = String(nextId++));
  let ELEMENTS = [];

  // version counter exposed to Python – incremented after every scan
  let VERSION = 0;
  window.__blVersion = VERSION;

  function fullScan(roots) {
    const out = [], work = [...roots];
    const depth = el => { let d = 0; while (el.parentElement) { el = el.parentElement; d++; } return d; };

    /* ───────────────────────── consider(el) ────────────────────────── */
    function consider(el) {
      if (!isVisible(el) || !isTopMost(el)) return;

      // 1️⃣  Does the element satisfy any heuristic?
      let hMatch = null;
      for (const h of HEUR) {
        if (!el.matches(h.selector))          continue;
        if (h.filter && !h.filter(el))        continue;
        if (h.kind === 'exclude')             return;   // blocked
        hMatch = h; break;
      }
      if (!hMatch) return;                    // not interactive

      const rc = rect(el); if (!rc.width || !rc.height) return;
      const areaEl = rc.width * rc.height;

      // 2️⃣  If we already captured a descendant, keep the larger area
      for (let i = 0; i < out.length; i++) {
        const it = out[i];
        if (el.contains(it.el)) {
          const rChild = it.el.getBoundingClientRect();
          const areaChild = rChild.width * rChild.height;
          // parent wins if ≥15 % larger *or* >120 px² larger
          if (areaEl > areaChild * 1.15 || (areaEl - areaChild) > 120) {
            out.splice(i, 1);                 // drop child
            break;
          } else {
            return;                           // child big enough → skip parent
          }
        }
      }

      // 3️⃣  Add element
      const id = giveId(el);
      const selector = `[data-bl-id="${id}"]`;

      let label = ((hMatch.label ? hMatch.label(el) : '') ||
                   el.getAttribute('aria-label') || el.title ||
                   el.placeholder || el.innerText ||
                   el.value || el.name || el.alt || '').trim();
      if (!label) {
        const img = el.querySelector('img[alt]'); if (img) label = img.alt.trim();
      }
      if (!label) {
        const svgTitle = el.querySelector('svg title');
        if (svgTitle && svgTitle.textContent) label = svgTitle.textContent.trim();
      }
      label = (label || el.tagName.toLowerCase()).slice(0, 40);

      out.push({
        id, tag: el.tagName.toLowerCase(), kind: hMatch.kind || 'other',
        label, selector, el
      });
    }
    /* ───────────────────────────────────────────────────────────────── */

    while (work.length) {
      const root = work.pop(); if (!root) continue;
      let nodes = [];
      try { nodes = [...root.querySelectorAll(SPEC_SEL)]
                       .sort((a, b) => depth(b) - depth(a)); }
      catch { /* cross-origin iframe etc. */ }
      nodes.forEach(consider);
      root.querySelectorAll('*').forEach(n =>
        n.shadowRoot && work.push(n.shadowRoot));
    }
    return out.sort((a, b) => a.id - b.id);
  }

  /* canvas overlay */
  const COLORS = ['#ff0000', '#00bfff', '#ffa500', '#7cfc00', '#ff1493'];
  let canvas = document.createElement('canvas');
  canvas.id = 'bl-overlay'; canvas.dataset.blOverlayRoot = 'true';
  canvas.setAttribute('style', `
    position:fixed; inset:0;
    width:100%; height:100%;                /* ← no 100vw/100vh */
    pointer-events:none; z-index:2147483647 !important;`);
  const ctx = canvas.getContext('2d');

  /* where to put the canvas so it’s always on top */
  function attachCanvas() {
    const dlg = document.querySelector('dialog[open]');
    if (dlg) { dlg.appendChild(canvas); return; }

    const topLayer = document.querySelector('#top-layer');
    if (topLayer) { topLayer.appendChild(canvas); return; }

    document.documentElement.appendChild(canvas);
  }

  const clearOverlay = () => { canvas.remove(); };

  function redraw() {
    if (!ELEMENTS.length) return;
    attachCanvas();

    const dpr = devicePixelRatio || 1;
    const bb = canvas.getBoundingClientRect();           // exact on-screen size
    ctx.canvas.width  = bb.width  * dpr;
    ctx.canvas.height = bb.height * dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width / dpr, ctx.canvas.height / dpr);

    ELEMENTS.forEach(it => {
      const rs = it.el.getClientRects();
      const c  = COLORS[it.id % COLORS.length];
      ctx.fillStyle   = c + '22';
      ctx.strokeStyle = c;
      ctx.lineWidth   = 2;
      [...rs].forEach(r => {
        if (!r.width || !r.height) return;
        const PAD = 1;                                     // tiny gap
        ctx.fillRect(r.x - PAD, r.y - PAD, r.width + PAD*2, r.height + PAD*2);
        ctx.strokeRect(r.x - PAD, r.y - PAD, r.width + PAD*2, r.height + PAD*2);
      });
      const r0 = rs[0];
      ctx.font = 'bold 14px sans-serif';
      ctx.lineWidth = 4; ctx.strokeStyle = '#000';
      ctx.strokeText(String(it.id), r0.x + 4, r0.y + 14);
      ctx.fillStyle = '#fff';
      ctx.fillText(String(it.id), r0.x + 4, r0.y + 14);
    });
  }

  /* scheduler */
  let on = false, mo = null, dirty = new Set(), idle = 0, raf = 0;
  const opt = { capture: true, passive: true };

  const idleScan = () => {
    const roots = dirty.size ? [...dirty] : [document];
    dirty.clear(); stCache.clear(); rectCache.clear();
    ELEMENTS = fullScan(roots); redraw();
    VERSION++;
    window.__blVersion = VERSION;
    window.__blScanDone?.(VERSION);
  };
  const reqIdle = () => {
    if (!idle) idle = setTimeout(() => { idle = 0; idleScan(); }, REFRESH_MS);
  };
  const fast = () => {
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; redraw(); });
    dirty.add(document); reqIdle();
  };

  function enable() {
    if (on) return; on = true; idleScan();
    mo ||= new MutationObserver(ms => ms.forEach(m => {
      if (m.target === canvas || m.target.dataset?.blOverlayRoot) return;
      dirty.add(document); reqIdle();
    }));
    mo.observe(document, { childList: true, subtree: true, attributes: true });
    addEventListener('scroll', fast, opt);
    addEventListener('resize', fast, opt);
  }
  function disable() {
    if (!on) return; on = false; mo?.disconnect();
    removeEventListener('scroll', fast, opt);
    removeEventListener('resize', fast, opt);
    clearTimeout(idle); idle = 0;
    cancelAnimationFrame(raf); raf = 0;
    clearOverlay(); ELEMENTS = [];
  }

  /* public API for Python */
  window.__bl = {
    enableOverlay:  enable,
    disableOverlay: disable,
    scan: () => ELEMENTS.map(e => ({
      id:        e.id,
      tag:       e.tag,
      kind:      e.kind,
      label:     e.label,
      selector:  e.selector
    }))
  };
  if (window.__blOverlayWanted) enable();
})();

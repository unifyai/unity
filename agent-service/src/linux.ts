// @ts-nocheck
import express from 'express';
import { execFile, exec, spawn } from 'child_process';
import { promises as fs } from 'fs';
import os from 'os';
import path from 'path';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const execAsync = promisify(exec);

const linux = express.Router();
// In-memory map for screen recordings: recording ID -> process and output file
const recordProcs: Record<string, { proc: any; filepath: string }> = {};

// Linux desktop automation endpoints

// Helpers
async function focusWindowByTitle(title: string): Promise<void> {
  try {
    await execFileAsync('wmctrl', ['-a', title]);
  } catch (_e) {
    await execFileAsync('xdotool', ['search', '--name', title, 'windowactivate', '--sync']);
  }
  await new Promise(r => setTimeout(r, 100));
}

async function focusWindowByClass(className: string): Promise<void> {
  try {
    // wmctrl -x matches WM_CLASS; try to activate by class
    await execFileAsync('wmctrl', ['-x', '-a', className]);
  } catch (_e) {
    // fallback to xdotool search by class
    await execFileAsync('xdotool', ['search', '--class', className, 'windowactivate', '--sync']);
  }
  await new Promise(r => setTimeout(r, 100));
}

function normalizeModifier(mod: string): string | null {
  const m = String(mod || '').toLowerCase().trim();
  if (!m) return null;
  if (['shift'].includes(m)) return 'shift';
  if (['ctrl', 'control', 'ctl'].includes(m)) return 'ctrl';
  if (['alt', 'option'].includes(m)) return 'alt';
  if (['meta', 'super', 'cmd', 'command', 'win', 'windows'].includes(m)) return 'super';
  return m; // let xdotool try
}

async function performClicks(clicks: Array<{ x: number; y: number; button?: number; repeat?: number; delayMs?: number; modifiers?: string[] }>): Promise<void> {
  for (const { x, y, button = 1, repeat = 1, delayMs = 30, modifiers = [] } of clicks) {
    // Move pointer
    await execFileAsync('xdotool', ['mousemove', `${x}`, `${y}`]);
    // Press modifiers (keydown)
    const norm = modifiers.map(normalizeModifier).filter(Boolean) as string[];
    for (const mod of norm) {
      await execFileAsync('xdotool', ['keydown', mod]);
    }
    // Click with repeat/delay
    const args = ['click', '--repeat', `${Math.max(1, repeat)}`, '--delay', `${Math.max(0, delayMs)}`, `${button}`];
    await execFileAsync('xdotool', args);
    // Release modifiers (keyup, reverse order)
    for (const mod of norm.slice().reverse()) {
      await execFileAsync('xdotool', ['keyup', mod]);
    }
  }
}

async function performKeys(keys: string[] | string | Array<string | string[]>): Promise<void> {
  const list: Array<string | string[]> = Array.isArray(keys) ? (keys as Array<string | string[]>) : [keys as string];
  for (const k of list) {
    if (Array.isArray(k)) {
      // key combo chord: hold all but last as modifiers, press last, release all
      const combo = k.map((t) => String(t));
      if (combo.length === 0) continue;
      const mods = combo.slice(0, Math.max(0, combo.length - 1)).map(normalizeModifier).filter(Boolean) as string[];
      const last = combo[combo.length - 1];
      for (const m of mods) await execFileAsync('xdotool', ['keydown', m]);
      // Treat some friendly names
      const keyName = last === 'Enter' ? 'Return' : last;
      await execFileAsync('xdotool', ['key', keyName]);
      for (const m of mods.slice().reverse()) await execFileAsync('xdotool', ['keyup', m]);
    } else {
      // simple string key or text
      const s = String(k);
      if (s === 'Enter') {
        await execFileAsync('xdotool', ['key', 'Return']);
      } else if (s.length === 1 || /\s/.test(s)) {
        // heuristically: short or contains spaces -> type as text
        await execFileAsync('xdotool', ['type', s]);
      } else {
        await execFileAsync('xdotool', ['key', s]);
      }
    }
  }
}

async function windowExistsByTitle(windowTitle: string): Promise<boolean> {
  const { stdout } = await execAsync('wmctrl -l');
  return stdout.split('\n').some((line: string) => line.includes(windowTitle));
}

async function templateExists(templatePath: string, threshold = '0.95'): Promise<{ exists: boolean; score: number }> {
  const cmd = `import -window root png:- | compare -metric NCC -subimage-search - ${templatePath} null:`;
  const { stderr } = await execAsync(cmd, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
  const score = parseFloat(stderr.split(' ')[0]);
  return { exists: score >= parseFloat(threshold), score };
}

async function resolveWindowId(id?: string, title?: string): Promise<string> {
  if (id && String(id).trim()) return String(id).trim();
  if (!title || !title.trim()) throw new Error('id or title required');
  const { stdout } = await execAsync('wmctrl -l');
  const line = stdout.split('\n').find((l: string) => l.includes(title));
  if (!line) throw new Error(`window_not_found: ${title}`);
  const token = line.trim().split(/\s+/)[0]; // hex id like 0x04200008
  return token;
}

async function resolveWindowIdsByClass(className: string): Promise<string[]> {
  try {
    const { stdout } = await execAsync(`xdotool search --class ${className}`, { encoding: 'utf8' });
    return stdout
      .split('\n')
      .map(s => s.trim())
      .filter(s => s.length > 0);
  } catch {
    return [];
  }
}

// GET /linux/screenshot
linux.get('/screenshot', async (req: any, res: any) => {
  try {
    const { save } = req.query as any;
    const shouldSave = String(save || '').toLowerCase() === 'true' || save === '1';
    const { stdout } = await execAsync(
      `import -window root png:-`,
      { encoding: 'buffer', maxBuffer: 10 * 1024 * 1024 }
    );
    const buf: Buffer = stdout as any;
    let filepath: string | undefined;
    if (shouldSave) {
      const iso = new Date().toISOString().replace(/[:.]/g, '-');
      filepath = path.join(os.tmpdir(), `screenshot-${iso}.png`);
      await fs.writeFile(filepath, buf);
    }
    res.json({ screenshot: buf.toString('base64'), saved: shouldSave, filepath });
  } catch (err) {
    res.status(500).json({ error: 'screenshot_failed', message: String(err) });
  }
});

// GET /linux/screenshot/region?x=&y=&w=&h=&save=
linux.get('/screenshot/region', async (req: any, res: any) => {
  const { x, y, w, h, save } = req.query as any;
  const xi = Number(x), yi = Number(y), wi = Number(w), hi = Number(h);
  if (!Number.isFinite(xi) || !Number.isFinite(yi) || !Number.isFinite(wi) || !Number.isFinite(hi) || wi <= 0 || hi <= 0) {
    return res.status(400).json({ error: 'bad_request', message: 'valid x,y,w,h required' });
  }
  try {
    const shouldSave = String(save || '').toLowerCase() === 'true' || save === '1';
    const { stdout } = await execAsync(
      `import -window root png:- | convert png:- -crop ${wi}x${hi}+${xi}+${yi} png:-`,
      { encoding: 'buffer', maxBuffer: 10 * 1024 * 1024 }
    );
    const buf: Buffer = stdout as any;
    let filepath: string | undefined;
    if (shouldSave) {
      const iso = new Date().toISOString().replace(/[:.]/g, '-');
      filepath = path.join(os.tmpdir(), `screenshot-region-${xi}-${yi}-${wi}x${hi}-${iso}.png`);
      await fs.writeFile(filepath, buf);
    }
    res.json({ screenshot: buf.toString('base64'), saved: shouldSave, filepath });
  } catch (err) {
    res.status(500).json({ error: 'screenshot_region_failed', message: String(err) });
  }
});

// Helper to write base64 PNG to a temp file
async function writeTempPNGFromB64(b64: string, prefix = 'img'): Promise<string> {
  const buf = Buffer.from(b64, 'base64');
  const file = path.join(os.tmpdir(), `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);
  await fs.writeFile(file, buf);
  return file;
}

// POST /linux/image/locate { templatePath? | templateB64?, x?, y?, w?, h?, threshold? }
linux.post('/image/locate', async (req: any, res: any) => {
  const { templatePath, templateB64, x, y, w, h, threshold = 0.95 } = req.body || {};
  if (!templatePath && !templateB64) {
    return res.status(400).json({ error: 'bad_request', message: 'templatePath or templateB64 required' });
  }
  let tmpTemplate: string | null = null;
  let tmpScreen: string | null = null;
  try {
    // Prepare template file
    const templateFile = templatePath && String(templatePath).trim().length > 0
      ? String(templatePath)
      : (tmpTemplate = await writeTempPNGFromB64(String(templateB64 || ''), 'tpl'));

    // Prepare screenshot or region file
    const xi = x !== undefined ? Number(x) : null;
    const yi = y !== undefined ? Number(y) : null;
    const wi = w !== undefined ? Number(w) : null;
    const hi = h !== undefined ? Number(h) : null;
    if (xi !== null && yi !== null && wi !== null && hi !== null) {
      tmpScreen = path.join(os.tmpdir(), `screen-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);
      await execAsync(
        `import -window root png:- | convert png:- -crop ${wi}x${hi}+${xi}+${yi} "${tmpScreen}"`,
        { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }
      );
    } else {
      tmpScreen = path.join(os.tmpdir(), `screen-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);
      await execAsync(`import -window root "${tmpScreen}"`, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
    }

    // Identify template size
    const { stdout: idOut } = await execAsync(`identify -format "%w %h" "${templateFile}"`, { encoding: 'utf8' });
    const [twStr, thStr] = idOut.trim().split(/\s+/);
    const tw = Number(twStr), th = Number(thStr);

    // Run subimage search
    const { stderr } = await execAsync(
      `compare -metric NCC -subimage-search "${tmpScreen}" "${templateFile}" null:`,
      { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }
    );
    // stderr typically: "0.987654 @ 123,456"
    const m = stderr.match(/([0-9]*\.?[0-9]+)\s*@\s*(\d+),(\d+)/);
    if (!m) {
      return res.json({ found: false, score: 0 });
    }
    const score = parseFloat(m[1]);
    const px = parseInt(m[2], 10);
    const py = parseInt(m[3], 10);
    const found = score >= Number(threshold);
    return res.json({ found, score, x: px, y: py, w: tw, h: th });
  } catch (err) {
    return res.status(500).json({ error: 'image_locate_failed', message: String(err) });
  } finally {
    try { if (tmpTemplate) await fs.unlink(tmpTemplate); } catch {}
    try { if (tmpScreen) await fs.unlink(tmpScreen); } catch {}
  }
});

// POST /linux/region/changed { beforeB64, afterB64, x, y, w, h, metric?, threshold? }
linux.post('/region/changed', async (req: any, res: any) => {
  const { beforeB64, afterB64, x, y, w, h, metric = 'AE', threshold } = req.body || {};
  if (!beforeB64 || !afterB64 || x === undefined || y === undefined || w === undefined || h === undefined) {
    return res.status(400).json({ error: 'bad_request', message: 'beforeB64, afterB64, x, y, w, h required' });
  }
  const xi = Number(x), yi = Number(y), wi = Number(w), hi = Number(h);
  if (!Number.isFinite(xi) || !Number.isFinite(yi) || !Number.isFinite(wi) || !Number.isFinite(hi) || wi <= 0 || hi <= 0) {
    return res.status(400).json({ error: 'bad_request', message: 'valid x,y,w,h required' });
  }
  let fBefore: string | null = null;
  let fAfter: string | null = null;
  let cBefore: string | null = null;
  let cAfter: string | null = null;
  try {
    fBefore = await writeTempPNGFromB64(String(beforeB64), 'before');
    fAfter = await writeTempPNGFromB64(String(afterB64), 'after');
    cBefore = path.join(os.tmpdir(), `cropb-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);
    cAfter = path.join(os.tmpdir(), `cropa-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);
    await execAsync(`convert "${fBefore}" -crop ${wi}x${hi}+${xi}+${yi} "${cBefore}"`);
    await execAsync(`convert "${fAfter}" -crop ${wi}x${hi}+${xi}+${yi} "${cAfter}"`);
    const met = String(metric || 'AE').toUpperCase();
    if (met === 'AE') {
      const { stderr } = await execAsync(`compare -metric AE "${cBefore}" "${cAfter}" null:`, { encoding: 'utf8' });
      const delta = parseFloat((stderr.trim().split(/\s+/)[0]) || '0');
      const thr = threshold !== undefined ? Number(threshold) : 0; // AE > 0 means any pixel changed
      const changed = delta > thr;
      return res.json({ changed, metric: 'AE', delta });
    } else {
      const { stderr } = await execAsync(`compare -metric NCC "${cBefore}" "${cAfter}" null:`, { encoding: 'utf8' });
      const score = parseFloat((stderr.trim().split(/\s+/)[0]) || '0');
      const thr = threshold !== undefined ? Number(threshold) : 0.995; // lower score = more change
      const changed = score < thr;
      return res.json({ changed, metric: 'NCC', score });
    }
  } catch (err) {
    return res.status(500).json({ error: 'region_changed_failed', message: String(err) });
  } finally {
    try { if (fBefore) await fs.unlink(fBefore); } catch {}
    try { if (fAfter) await fs.unlink(fAfter); } catch {}
    try { if (cBefore) await fs.unlink(cBefore); } catch {}
    try { if (cAfter) await fs.unlink(cAfter); } catch {}
  }
});
// GET /linux/mouse/position
linux.get('/mouse/position', async (_req: any, res: any) => {
  try {
    const { stdout } = await execAsync('xdotool getmouselocation --shell', { encoding: 'utf8' });
    const lines = stdout.trim().split('\n');
    const map: Record<string, string> = {};
    for (const line of lines) {
      const idx = line.indexOf('=');
      if (idx > 0) {
        const key = line.slice(0, idx).trim();
        const value = line.slice(idx + 1).trim();
        map[key] = value;
      }
    }
    const x = Number(map['X'] || 0);
    const y = Number(map['Y'] || 0);
    const screen = Number(map['SCREEN'] || 0);
    const window = map['WINDOW'] || '';
    res.json({ x, y, screen, window });
  } catch (err) {
    res.status(500).json({ error: 'mouse_failed', message: String(err) });
  }
});

// POST /linux/mouse/move { x, y }
linux.post('/mouse/move', async (req: any, res: any) => {
  const { x, y } = req.body as { x?: number; y?: number };
  if (typeof x !== 'number' || typeof y !== 'number') {
    return res.status(400).json({ error: 'bad_request', message: 'x and y are required numbers' });
  }
  try {
    await execFileAsync('xdotool', ['mousemove', `${x}`, `${y}`]);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'mouse_move_failed', message: String(err) });
  }
});

// POST /linux/drag { fromX?, fromY?, toX, toY, button?, steps?, delayMs?, modifiers? }
linux.post('/drag', async (req: any, res: any) => {
  const { fromX, fromY, toX, toY, button = 1, steps = 1, delayMs = 0, modifiers = [] } = req.body as {
    fromX?: number; fromY?: number; toX?: number; toY?: number; button?: number; steps?: number; delayMs?: number; modifiers?: string[];
  };
  if (typeof toX !== 'number' || typeof toY !== 'number') {
    return res.status(400).json({ error: 'bad_request', message: 'toX and toY are required numbers' });
  }
  try {
    let startX = fromX; let startY = fromY;
    if (typeof startX !== 'number' || typeof startY !== 'number') {
      // get current mouse position
      const { stdout } = await execAsync('xdotool getmouselocation --shell', { encoding: 'utf8' });
      const map: Record<string, string> = {};
      for (const line of stdout.trim().split('\n')) {
        const idx = line.indexOf('=');
        if (idx > 0) map[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
      }
      startX = Number(map['X'] || 0);
      startY = Number(map['Y'] || 0);
    }
    // Normalize modifiers
    const normMods = (modifiers as string[]).map(normalizeModifier).filter(Boolean) as string[];
    // Move to start
    await execFileAsync('xdotool', ['mousemove', `${startX}`, `${startY}`]);
    // Press modifiers (keydown)
    for (const mod of normMods) {
      await execFileAsync('xdotool', ['keydown', mod]);
    }
    // Mouse down
    await execFileAsync('xdotool', ['mousedown', `${button}`]);
    // Intermediate moves if steps > 1
    const n = Math.max(1, Number(steps) || 1);
    for (let i = 1; i <= n; i++) {
      const ix = Math.round(startX + (i * (toX - startX)) / n);
      const iy = Math.round(startY + (i * (toY - startY)) / n);
      await execFileAsync('xdotool', ['mousemove', `${ix}`, `${iy}`]);
      if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
    }
    // Mouse up
    await execFileAsync('xdotool', ['mouseup', `${button}`]);
    // Release modifiers (keyup)
    for (const mod of normMods.slice().reverse()) {
      await execFileAsync('xdotool', ['keyup', mod]);
    }
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'drag_failed', message: String(err) });
  }
});

// POST /linux/scroll { direction: 'up'|'down'|'left'|'right', amount?: number, delayMs?: number }
linux.post('/scroll', async (req: any, res: any) => {
  const { direction, amount = 1, delayMs = 0 } = req.body as { direction?: string; amount?: number; delayMs?: number };
  const dir = String(direction || '').toLowerCase();
  const map: Record<string, number> = { up: 4, down: 5, left: 6, right: 7 };
  if (!(dir in map)) {
    return res.status(400).json({ error: 'bad_request', message: "direction must be one of 'up'|'down'|'left'|'right'" });
  }
  const btn = map[dir];
  const n = Math.max(1, Number(amount) || 1);
  const d = Math.max(0, Number(delayMs) || 0);
  try {
    await execFileAsync('xdotool', ['click', '--repeat', `${n}`, '--delay', `${d}`, `${btn}`]);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'scroll_failed', message: String(err) });
  }
});

// POST /linux/app/open  { cmd, args?, cwd?, env?, wait? }
linux.post('/app/open', async (req: any, res: any) => {
  const { cmd, args = [], cwd, env = {}, wait = false } = req.body as {
    cmd?: string; args?: string[]; cwd?: string; env?: Record<string, string>; wait?: boolean;
  };
  if (!cmd || typeof cmd !== 'string' || !cmd.trim()) {
    return res.status(400).json({ error: 'bad_request', message: 'cmd is required' });
  }
  try {
    const child = spawn(cmd, Array.isArray(args) ? args : [], {
      cwd: cwd || process.cwd(),
      env: { ...process.env, ...(env || {}) },
      detached: true,
      stdio: 'ignore',
    });
    const pid = child.pid;
    if (wait) {
      child.on('error', (e) => {});
      child.unref();
      child.on('exit', (code) => {
        // no-op
      });
      return res.json({ pid, status: 'started' });
    } else {
      child.unref();
      return res.json({ pid, status: 'started' });
    }
  } catch (err) {
    return res.status(500).json({ error: 'app_open_failed', message: String(err) });
  }
});

// POST /linux/click
linux.post('/click', async (req: any, res: any) => {
  const { clicks = [], x, y, button, repeat, delayMs, modifiers } = req.body as {
    clicks?: Array<{ x: number; y: number; button?: number; repeat?: number; delayMs?: number; modifiers?: string[] }>;
    x?: number; y?: number; button?: number; repeat?: number; delayMs?: number; modifiers?: string[];
  };
  try {
    const list = Array.isArray(clicks) && clicks.length > 0
      ? clicks
      : (typeof x === 'number' && typeof y === 'number' ? [{ x, y, button, repeat, delayMs, modifiers }] : []);
    if (list.length === 0) return res.status(400).json({ error: 'bad_request', message: 'clicks or (x,y) required' });
    await performClicks(list);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'click_failed', message: String(err) });
  }
});

// POST /linux/type (supports key combos via nested arrays)
linux.post('/type', async (req: any, res: any) => {
  const { keys } = req.body as { keys?: string[] | string | Array<string | string[]> };
  if (keys === undefined) return res.status(400).json({ error: 'bad_request', message: 'keys required' });
  try {
    await performKeys(keys);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'type_failed', message: String(err) });
  }
});

// GET /linux/window (list)
linux.get('/window', async (req: any, res: any) => {
  const { extended } = req.query as any;
  try {
    if (extended) {
      // -lx -p: id, desktop, pid, class, host, title
      const { stdout } = await execAsync('wmctrl -lx -p', { encoding: 'utf8' });
      const windows = stdout
        .split('\n')
        .map((line: string) => line.trim())
        .filter((line: string) => line.length > 0)
        .map((line: string) => {
          const parts = line.split(/\s+/);
          const id = parts[0];
          const desktop = Number(parts[1]);
          const pid = Number(parts[2]);
          const classRaw = parts[3]; // e.g. Xterm.XTerm
          const host = parts[4];
          const title = parts.slice(5).join(' ');
          return { id, desktop, pid, class: classRaw, host, title };
        });
      return res.json({ windows });
    }
    // default: geometry if available
    const { stdout } = await execAsync('wmctrl -lG', { encoding: 'utf8' });
    const windows = stdout
      .split('\n')
      .map((line: string) => line.trim())
      .filter((line: string) => line.length > 0)
      .map((line: string) => {
        const parts = line.split(/\s+/);
        const id = parts[0];
        const desktop = Number(parts[1]);
        const x = Number(parts[2]);
        const y = Number(parts[3]);
        const w = Number(parts[4]);
        const h = Number(parts[5]);
        const host = parts[6];
        const title = parts.slice(7).join(' ');
        return { id, desktop, x, y, w, h, host, title };
      });
    res.json({ windows });
  } catch (err) {
    try {
      const { stdout: alt } = await execAsync('wmctrl -l', { encoding: 'utf8' });
      const windows = alt
        .split('\n')
        .map((line: string) => line.trim())
        .filter((line: string) => line.length > 0)
        .map((line: string) => {
          const id = line.split(/\s+/)[0];
          const rest = line.slice(id.length).trim();
          return { id, title: rest };
        });
      return res.json({ windows });
    } catch (e2) {
      return res.status(500).json({ error: 'window_list_failed', message: String(err) });
    }
  }
});

// POST /linux/window/focus
linux.post('/window/focus', async (req: any, res: any) => {
  const { title, class: klass } = req.body as { title?: string; class?: string };
  if ((!title || !title.trim()) && (!klass || !klass.trim())) {
    return res.status(400).json({ error: 'bad_request', message: 'title or class is required' });
  }
  try {
    if (klass && klass.trim()) {
      await focusWindowByClass(klass.trim());
    } else if (title && title.trim()) {
      await focusWindowByTitle(title.trim());
    }
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(404).json({ error: 'focus_failed', message: String(err) });
  }
});

// GET /linux/window/exist  (migrated from /linux/exists)
linux.get('/window/exist', async (req: any, res: any) => {
  const { windowTitle, templatePath, threshold = '0.95' } = req.query as any;
  try {
    if (windowTitle) {
      const exists = await windowExistsByTitle(windowTitle);
      return res.json({ exists });
    }
    if (templatePath) {
      const { exists, score } = await templateExists(templatePath, threshold);
      return res.json({ exists, score });
    }
    res.status(400).json({ error: 'bad_request', message: 'windowTitle or templatePath required' });
  } catch (err) {
    res.status(500).json({ error: 'exists_failed', message: String(err) });
  }
});

// POST /linux/window/move  { id|title, x, y }
linux.post('/window/move', async (req: any, res: any) => {
  const { id, title, x, y } = req.body as { id?: string; title?: string; x?: number; y?: number };
  if (typeof x !== 'number' || typeof y !== 'number') {
    return res.status(400).json({ error: 'bad_request', message: 'x and y are required numbers' });
  }
  try {
    const win = await resolveWindowId(id, title);
    await execFileAsync('xdotool', ['windowmove', win, `${x}`, `${y}`]);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'move_failed', message: String(err) });
  }
});

// POST /linux/window/resize  { id|title, w, h }
linux.post('/window/resize', async (req: any, res: any) => {
  const { id, title, w, h } = req.body as { id?: string; title?: string; w?: number; h?: number };
  if (typeof w !== 'number' || typeof h !== 'number') {
    return res.status(400).json({ error: 'bad_request', message: 'w and h are required numbers' });
  }
  try {
    const win = await resolveWindowId(id, title);
    await execFileAsync('xdotool', ['windowsize', win, `${w}`, `${h}`]);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'resize_failed', message: String(err) });
  }
});

// POST /linux/window/state  { id|title|class, action: minimize|maximize|restore|close }
linux.post('/window/state', async (req: any, res: any) => {
  const { id, title, class: klass, action } = req.body as { id?: string; title?: string; class?: string; action?: string };
  const valid = new Set(['minimize', 'maximize', 'restore', 'close']);
  if (!action || !valid.has(action)) {
    return res.status(400).json({ error: 'bad_request', message: 'action must be one of minimize|maximize|restore|close' });
  }
  try {
    const doForId = async (winId: string) => {
      if (action === 'minimize') {
        await execFileAsync('xdotool', ['windowminimize', winId]);
      } else if (action === 'maximize') {
        await execFileAsync('wmctrl', ['-ir', winId, '-b', 'add,maximized_vert,maximized_horz']);
      } else if (action === 'restore') {
        await execFileAsync('wmctrl', ['-ir', winId, '-b', 'remove,maximized_vert,maximized_horz']);
        await execFileAsync('xdotool', ['windowmap', winId]);
      } else if (action === 'close') {
        await execFileAsync('xdotool', ['windowclose', winId]);
      }
    };

    if (klass && klass.trim()) {
      const ids = await resolveWindowIdsByClass(klass.trim());
      for (const wid of ids) {
        await doForId(wid);
      }
      return res.json({ status: 'ok', affected: ids.length });
    }

    const win = await resolveWindowId(id, title);
    await doForId(win);
    res.json({ status: 'ok', affected: 1 });
  } catch (err) {
    res.status(500).json({ error: 'state_failed', message: String(err) });
  }
});

// POST /linux/record/start { fps?, region? }
linux.post('/record/start', async (req: any, res: any) => {
  const { fps = 10, region } = req.body as { fps?: number; region?: { x: number; y: number; w: number; h: number } };
  try {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const outfile = path.join(os.tmpdir(), `record-${id}.mp4`);
    const args: string[] = ['-y', '-f', 'x11grab', '-framerate', `${fps}`];
    if (
      region &&
      Number.isFinite(region.x) &&
      Number.isFinite(region.y) &&
      Number.isFinite(region.w) &&
      Number.isFinite(region.h)
    ) {
      args.push('-video_size', `${region.w}x${region.h}`, '-i', `:99.0+${region.x},${region.y}`);
    } else {
      args.push('-video_size', '1920x1080', '-i', process.env.DISPLAY || ':99.0');
    }
    args.push(outfile);
    const child = spawn('ffmpeg', args, { detached: true, stdio: 'ignore' });
    child.unref();
    recordProcs[id] = { proc: child, filepath: outfile };
    res.json({ id, status: 'started', filepath: outfile });
  } catch (err) {
    res.status(500).json({ error: 'record_start_failed', message: String(err) });
  }
});

// POST /linux/record/stop { id }
linux.post('/record/stop', async (req: any, res: any) => {
  const { id } = req.body as { id?: string };
  if (!id || !recordProcs[id]) {
    return res.status(400).json({ error: 'bad_request', message: 'valid id required' });
  }
  try {
    const { proc, filepath } = recordProcs[id];
    proc.kill('SIGINT');
    delete recordProcs[id];
    res.json({ id, status: 'stopped', filepath });
  } catch (err) {
    res.status(500).json({ error: 'record_stop_failed', message: String(err) });
  }
});

// GET /linux/fs/list?path=
linux.get('/fs/list', async (req: any, res: any) => {
  const { path: p } = req.query as any;
  try {
    if (typeof p !== 'string') throw new Error('path is required');
    const target = path.resolve(process.cwd(), p);
    if (!target.startsWith(process.cwd())) throw new Error('path out of range');
    const files = await fs.readdir(target);
    res.json({ files });
  } catch (err) {
    res.status(500).json({ error: 'fs_list_failed', message: String(err) });
  }
});

// POST /linux/fs/write { path, contentBase64 }
linux.post('/fs/write', async (req: any, res: any) => {
  const { path: p, contentBase64 } = req.body as any;
  try {
    if (!p || typeof contentBase64 !== 'string') throw new Error('path and contentBase64 required');
    const target = path.resolve(process.cwd(), p);
    if (!target.startsWith(process.cwd())) throw new Error('path out of range');
    const buf = Buffer.from(contentBase64, 'base64');
    await fs.writeFile(target, buf);
    res.json({ status: 'ok', path: p });
  } catch (err) {
    res.status(500).json({ error: 'fs_write_failed', message: String(err) });
  }
});

export default linux;

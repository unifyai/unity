// @ts-nocheck
import express from 'express';
import { execFile, exec, spawn } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const execAsync = promisify(exec);

const linux = express.Router();

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

async function performClicks(clicks: Array<{ x: number; y: number; button?: number }>): Promise<void> {
  for (const { x, y, button = 1 } of clicks) {
    await execFileAsync('xdotool', ['mousemove', `${x}`, `${y}`, 'click', `${button}`]);
  }
}

async function performKeys(keys: string[] | string): Promise<void> {
  const list = Array.isArray(keys) ? keys : [keys];
  for (const k of list) {
    if (k === 'Enter') {
      await execFileAsync('xdotool', ['key', 'Return']);
    } else {
      await execFileAsync('xdotool', ['type', k]);
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
linux.get('/screenshot', async (_req: any, res: any) => {
  try {
    const { stdout } = await execAsync(
      `import -window root png:-`,
      { encoding: 'buffer', maxBuffer: 10 * 1024 * 1024 }
    );
    res.json({ screenshot: (stdout as any).toString('base64') });
  } catch (err) {
    res.status(500).json({ error: 'screenshot_failed', message: String(err) });
  }
});

// GET /linux/mouse
linux.get('/mouse', async (_req: any, res: any) => {
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
  const { clicks = [], x, y, button } = req.body as {
    clicks?: Array<{ x: number; y: number; button?: number }>;
    x?: number; y?: number; button?: number;
  };
  try {
    const list = Array.isArray(clicks) && clicks.length > 0 ? clicks : (typeof x === 'number' && typeof y === 'number' ? [{ x, y, button }] : []);
    if (list.length === 0) return res.status(400).json({ error: 'bad_request', message: 'clicks or (x,y) required' });
    await performClicks(list);
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'click_failed', message: String(err) });
  }
});

// POST /linux/type
linux.post('/type', async (req: any, res: any) => {
  const { keys } = req.body as { keys?: string[] | string };
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

export default linux;

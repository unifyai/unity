// @ts-nocheck
import express from 'express';
import { execFile, exec } from 'child_process';
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

// POST /linux/window/focus
linux.post('/window/focus', async (req: any, res: any) => {
  const { title } = req.body as { title?: string };
  if (!title || typeof title !== 'string' || !title.trim()) {
    return res.status(400).json({ error: 'bad_request', message: 'title is required' });
  }
  try {
    await focusWindowByTitle(title);
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

export default linux;

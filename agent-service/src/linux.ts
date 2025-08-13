// @ts-nocheck
import express from 'express';
import { execFile, exec } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const execAsync = promisify(exec);

const linux = express.Router();

// Linux desktop automation endpoints

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

// POST /linux/act
linux.post('/act', async (req: any, res: any) => {
  const { clicks = [], keys = [], focusWindowTitle } = req.body as {
    clicks?: Array<{ x: number; y: number; button?: number }>;
    keys?: string[];
    focusWindowTitle?: string;
  };
  try {
    // Optionally bring a window to the foreground by title
    if (focusWindowTitle && typeof focusWindowTitle === 'string' && focusWindowTitle.trim().length > 0) {
      try {
        await execFileAsync('wmctrl', ['-a', focusWindowTitle]);
      } catch (e) {
        // Fallback to xdotool if wmctrl fails
        try {
          await execFileAsync('xdotool', ['search', '--name', focusWindowTitle, 'windowactivate', '--sync']);
        } catch (e2) {
          return res.status(404).json({ error: 'focus_failed', message: `Could not focus window with title: ${focusWindowTitle}` });
        }
      }
      // small settle delay
      await new Promise(r => setTimeout(r, 100));
    }

    for (const { x, y, button = 1 } of clicks) {
      await execFileAsync('xdotool', ['mousemove', `${x}`, `${y}`, 'click', `${button}`]);
    }
    for (const k of keys) {
      if (k === 'Enter') {
        await execFileAsync('xdotool', ['key', 'Return']);
      } else {
        await execFileAsync('xdotool', ['type', k]);
      }
    }
    res.json({ status: 'ok' });
  } catch (err) {
    res.status(500).json({ error: 'act_failed', message: String(err) });
  }
});

// GET /linux/exists
linux.get('/exists', async (req: any, res: any) => {
  const { windowTitle, templatePath, threshold = '0.95' } = req.query as any;
  try {
    if (windowTitle) {
      const { stdout } = await execAsync('wmctrl -l');
      const exists = stdout.split('\n').some((line: string) => line.includes(windowTitle));
      return res.json({ exists });
    }
    if (templatePath) {
      const cmd = `import -window root png:- | compare -metric NCC -subimage-search - ${templatePath} null:`;
      const { stderr } = await execAsync(cmd, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
      const score = parseFloat(stderr.split(' ')[0]);
      return res.json({ exists: score >= parseFloat(threshold), score });
    }
    res.status(400).json({ error: 'bad_request', message: 'windowTitle or templatePath required' });
  } catch (err) {
    res.status(500).json({ error: 'exists_failed', message: String(err) });
  }
});

export default linux;

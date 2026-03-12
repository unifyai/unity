import express, { Request, Response } from 'express';
import https from 'https';
import http from 'http';
import expressWs from 'express-ws';
import WebSocket from 'ws';
import util from 'util';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions, AgentMemory, Observation } from 'magnitude-core';
import { z, ZodTypeAny, ZodAny, ZodType } from 'zod';
import { partitionHtml, serializeToMarkdown, PartitionOptions, MarkdownSerializerOptions } from 'magnitude-extract';
import dotenv from 'dotenv';
dotenv.config();
import os from 'os';
import path from 'path';
import fs from 'fs';
import net from 'net';
import { randomUUID } from 'crypto';
import { ChildProcess, spawn, execSync } from 'child_process';
import multer from 'multer';
import { jsonSchemaToZod } from './jsonSchemaToZod';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// --- Debug logging helpers ---
const MAGNITUDE_DEBUG = process.env.MAGNITUDE_DEBUG === 'true';
const MAGNITUDE_LOG_DIR = process.env.MAGNITUDE_LOG_DIR || '';

function makeActId(task: string): string {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const slug = task.slice(0, 40).replace(/[^a-zA-Z0-9]+/g, '_').replace(/_+$/, '');
  return `${ts}_${slug}`;
}

function debugSaveImage(actId: string, label: string, base64Data: string): void {
  if (!MAGNITUDE_DEBUG || !MAGNITUDE_LOG_DIR) return;
  try {
    const imgPath = path.join(MAGNITUDE_LOG_DIR, 'acts', actId, `${label}.png`);
    fs.mkdirSync(path.dirname(imgPath), { recursive: true });
    fs.writeFileSync(imgPath, Buffer.from(base64Data, 'base64'));
  } catch (err) {
    console.warn(`[debug] Failed to save image ${label}: ${err}`);
  }
}

function debugSaveTrace(actId: string, trace: Record<string, any>): void {
  if (!MAGNITUDE_DEBUG || !MAGNITUDE_LOG_DIR) return;
  try {
    const tracePath = path.join(MAGNITUDE_LOG_DIR, 'acts', actId, 'act_trace.json');
    fs.mkdirSync(path.dirname(tracePath), { recursive: true });
    fs.writeFileSync(tracePath, JSON.stringify(trace, null, 2));
  } catch (err) {
    console.warn(`[debug] Failed to save trace: ${err}`);
  }
}

function debugLog(line: string): void {
  if (!MAGNITUDE_DEBUG || !MAGNITUDE_LOG_DIR) return;
  try {
    fs.mkdirSync(MAGNITUDE_LOG_DIR, { recursive: true });
    fs.appendFileSync(path.join(MAGNITUDE_LOG_DIR, 'magnitude.log'), line + '\n');
  } catch (_) { /* best-effort */ }
}

// --- File System and Command Execution Utilities ---
//
// Workspace root for file operations, command execution, and browser downloads.
// Matches Unity's get_local_root() default of ~/Unity/Local.
// Override via UNITY_LOCAL_ROOT env var.
const LOCAL_ROOT = process.env.UNITY_LOCAL_ROOT || path.join(os.homedir(), 'Unity', 'Local');
try { fs.mkdirSync(LOCAL_ROOT, { recursive: true }); } catch (_e) { /* ignore */ }
const DEFAULT_EXEC_TIMEOUT = 60 * 60 * 1000; // 1 hour


// Multer configuration for multipart file uploads
const uploadTempDir = path.join(os.tmpdir(), 'unity-uploads');
try {
  fs.mkdirSync(uploadTempDir, { recursive: true });
} catch (_e) {
  // ignore
}

const uploadMiddleware = multer({
  dest: uploadTempDir,
  limits: {
    fileSize: 500 * 1024 * 1024, // 500MB per file
    files: 100,
  },
});

function sanitizePath(filename: string, baseDir: string): string {
  const resolved = path.resolve(baseDir, filename);
  const normalizedBase = path.resolve(baseDir);
  if (!resolved.startsWith(normalizedBase + path.sep) && resolved !== normalizedBase) {
    throw new Error(`Path traversal blocked: ${filename}`);
  }
  return resolved;
}

async function ensureDir(dirPath: string): Promise<void> {
  await fs.promises.mkdir(dirPath, { recursive: true });
}

async function writeFileWithEncoding(
  filepath: string,
  content: string,
  encoding: 'text' | 'base64' = 'text'
): Promise<void> {
  await ensureDir(path.dirname(filepath));
  if (encoding === 'base64') {
    const buffer = Buffer.from(content, 'base64');
    await fs.promises.writeFile(filepath, buffer);
  } else {
    await fs.promises.writeFile(filepath, content, 'utf-8');
  }
}

async function readFileWithEncoding(
  filepath: string,
  encoding: 'text' | 'base64' = 'text'
): Promise<string> {
  const buffer = await fs.promises.readFile(filepath);
  return encoding === 'base64' ? buffer.toString('base64') : buffer.toString('utf-8');
}

interface ExecResult {
  exitCode: number;
  stdout: string;
  stderr: string;
  duration: number;
}

type ShellMode = 'cmd' | 'powershell';

function getShellConfig(shellMode: ShellMode): string | boolean {
  const isWindows = process.platform === 'win32';

  if (!isWindows) {
    return true;  // Use default /bin/sh on Unix
  }

  if (shellMode === 'cmd') {
    return 'cmd.exe';
  }

  // PowerShell (default on Windows)
  return 'powershell.exe';
}

function executeCommand(command: string, cwd: string, timeout: number, shellMode: ShellMode = 'powershell'): Promise<ExecResult> {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let stdout = '';
    let stderr = '';
    let killed = false;

    const proc = spawn(command, [], {
      shell: getShellConfig(shellMode),
      cwd,
      timeout,
    });

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('error', (err) => {
      stderr += err.message;
    });

    proc.on('close', (code, signal) => {
      const duration = Date.now() - startTime;
      if (signal === 'SIGTERM') {
        killed = true;
        stderr += `\nProcess killed after ${timeout}ms timeout`;
      }
      resolve({
        exitCode: code ?? (killed ? 124 : 1),
        stdout,
        stderr,
        duration,
      });
    });
  });
}

function getDefaultBrowserPaths() {
  const downloadsPath = path.join(LOCAL_ROOT, 'Downloads');
  const tracesDir = path.join(LOCAL_ROOT, 'Traces');
  return { downloadsPath, tracesDir };
}

const defaultBrowserPaths = getDefaultBrowserPaths();

const app = express();
const wsInstance = expressWs(app);
app.use(express.json({ limit: '100mb' }));

const ALLOWED_ORIGINS = (process.env.CORS_ALLOWED_ORIGINS || '').split(',').filter(Boolean);
app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  }
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  next();
});

// --- Authorization (Bearer) middleware ---
function verifyApiKeyWithUnify(apiKey: string): Promise<boolean> {
  return new Promise((resolve) => {
    const url = new URL(`${process.env.ORCHESTRA_URL}/user/basic-info`);
    const options = {
      method: 'GET',
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname + url.search,
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    };

    const requestLib = url.protocol === 'https:' ? https : http;
    const req = requestLib.request(options, (res) => {
      const code = res.statusCode || 0;
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        if (!(code >= 200 && code < 300)) return resolve(false);
        return resolve(true);
      });
    });
    req.on('error', () => {
      resolve(false);
    });
    req.end();
  });
}

async function auth(req: Request, res: Response, next: Function) {
  const authHeader = req.header('authorization') || '';
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return res.status(401).json({ error: 'unauthorized', message: 'Missing or invalid API key' });
  }
  const apiKey = match[1].trim();

  // Check 1: Bearer token must match UNIFY_KEY
  if (apiKey !== process.env.UNIFY_KEY) {
    return res.status(401).json({ error: 'unauthorized', message: 'Invalid API key' });
  }

  // Check 2: Verify with /user/basic-info endpoint
  try {
    const ok = await verifyApiKeyWithUnify(apiKey);
    if (!ok) {
      return res.status(401).json({ error: 'unauthorized', message: 'API key verification failed' });
    }
  } catch (e) {
    return res.status(401).json({ error: 'unauthorized', message: 'API key verification failed' });
  }

  next();
}

app.use(auth);

// --- CLI argument parsing ---
function parseIntArg(flag: string, defaultValue: number): number {
  const idx = process.argv.indexOf(flag);
  if (idx !== -1 && idx + 1 < process.argv.length) {
    const val = parseInt(process.argv[idx + 1], 10);
    return isNaN(val) ? defaultValue : val;
  }
  return defaultValue;
}

const ACT_HISTORY_DEPTH = parseIntArg('--history-depth', 5);
console.log(`[memory-carryover] Act history depth: ${ACT_HISTORY_DEPTH}`);

// --- Session registry ---
interface ActHistoryEntry {
  task: string;
  observations: Observation[];
}

interface SessionInfo {
  agent: BrowserAgent;
  mode: 'web' | 'desktop' | 'web-vm';
  createdAt: Date;
  lastAccessed: Date;
  actHistory: ActHistoryEntry[];
}

const activeSessions = new Map<string, SessionInfo>();
const SESSION_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

// Cleanup inactive sessions periodically
setInterval(() => {
  const now = Date.now();
  for (const [sessionId, session] of activeSessions.entries()) {
    if (now - session.lastAccessed.getTime() > SESSION_TIMEOUT_MS) {
      console.log(`Cleaning up inactive session: ${sessionId}`);
      session.agent.stop().catch((err: unknown) => console.error(`Error stopping session ${sessionId}:`, err));
      activeSessions.delete(sessionId);
      broadcastSessionEvent(sessionId, 'timeout');
    }
  }
}, 5 * 60 * 1000); // Check every 5 minutes

const port = process.env.PORT || 3000;

// --- WebSocket Log Broadcasting Logic ---
const logClients = new Set<WebSocket>();

function broadcastLog(message: string) {
  logClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

function broadcastSessionEvent(sessionId: string, reason: string) {
  broadcastLog(JSON.stringify({ __type: 'session:closed', sessionId, reason }));
}

// Monkey-patch console methods to capture and broadcast logs
const originalLog = console.log;
const originalError = console.error;
const originalWarn = console.warn;

console.log = (...args: any[]) => {
  const message = util.format(...args);
  broadcastLog(message);
  originalLog.apply(console, args);
};

console.error = (...args: any[]) => {
  const message = util.format(...args);
  broadcastLog(message);
  originalError.apply(console, args);
};

console.warn = (...args: any[]) => {
  const message = util.format(...args);
  broadcastLog(message);
  originalWarn.apply(console, args);
};

// --- WebSocket Endpoint Handler ---
wsInstance.app.ws('/logs/stream', async (ws: WebSocket, req: Request) => {
  // Authenticate WebSocket connection
  const authHeader = req.header('authorization') || '';
  const match = authHeader.match(/^Bearer\s+(.+)$/i);

  if (!match) {
    console.log('WebSocket connection rejected: No auth header');
    ws.close(1008, 'Missing or invalid API key');
    return;
  }

  const apiKeyRaw = match[1].trim();
  const apiKey = apiKeyRaw.split(/\s+/)[0];

  // Check 1: Bearer token must match UNIFY_KEY
  if (apiKey !== process.env.UNIFY_KEY) {
    console.log('WebSocket connection rejected: Invalid API key');
    ws.close(1008, 'Invalid API key');
    return;
  }

  // Check 2: Verify with /user/basic-info endpoint
  try {
    const ok = await verifyApiKeyWithUnify(apiKey);
    if (!ok) {
      console.log('WebSocket connection rejected: Auth failed');
      ws.close(1008, 'API key verification failed');
      return;
    }
  } catch (e) {
    console.log('WebSocket connection rejected: Auth error');
    ws.close(1008, 'API key verification failed');
    return;
  }

  console.log('Log stream client connected and authenticated.');
  logClients.add(ws);

  ws.on('close', () => {
    console.log('Log stream client disconnected.');
    logClients.delete(ws);
  });

  ws.on('error', (error: Error) => {
    console.error('Log stream client error:', error);
    logClients.delete(ws);
  });
});


// --- Demo Sites ---
// Port-to-directory mapping (same convention as the Python demo_sites.py)
const DEMO_SITE_DIRS: Record<number, string> = {
  4001: 'example',
  4002: 'vantage-portal',
};

const demoSiteProcesses: Map<number, ChildProcess> = new Map();

function isPortOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = net.createConnection({ port, host: '127.0.0.1' });
    sock.setTimeout(500);
    sock.on('connect', () => { sock.destroy(); resolve(true); });
    sock.on('error', () => { sock.destroy(); resolve(false); });
    sock.on('timeout', () => { sock.destroy(); resolve(false); });
  });
}

function waitForPort(port: number, timeoutMs = 5000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    const check = async () => {
      if (await isPortOpen(port)) return resolve(true);
      if (Date.now() >= deadline) return resolve(false);
      setTimeout(check, 200);
    };
    check();
  });
}

function findDemoSitesRoot(): string | null {
  // demo-sites/ lives inside agent-service/ so it's always co-located
  const candidates = [
    path.resolve(__dirname, '..', 'demo-sites'),           // dev: agent-service/src/../demo-sites
    path.resolve(__dirname, '..', '..', 'demo-sites'),     // compiled: agent-service/dist/../../demo-sites
    '/app/agent-service/demo-sites',                        // Docker
  ];
  for (const dir of candidates) {
    if (fs.existsSync(dir)) return dir;
  }
  return null;
}

async function ensureDemoSites(urlMappings: Record<string, string>): Promise<void> {
  const demoSitesRoot = findDemoSitesRoot();
  if (!demoSitesRoot) {
    console.warn('[demo-sites] No demo-sites directory found, skipping');
    return;
  }

  for (const [, replacement] of Object.entries(urlMappings)) {
    let parsed: URL;
    try { parsed = new URL(replacement); } catch { continue; }

    const host = parsed.hostname;
    if (host !== 'localhost' && host !== '127.0.0.1') continue;

    const port = parseInt(parsed.port, 10);
    if (!port) continue;

    if (await isPortOpen(port)) {
      console.log(`[demo-sites] Port ${port} already in use, skipping`);
      continue;
    }

    if (demoSiteProcesses.has(port)) {
      const existing = demoSiteProcesses.get(port)!;
      if (existing.exitCode === null) {
        console.log(`[demo-sites] Process for port ${port} still running (pid ${existing.pid})`);
        continue;
      }
      console.warn(`[demo-sites] Process for port ${port} exited, restarting`);
    }

    // Find the matching demo site directory
    const dirName = DEMO_SITE_DIRS[port];
    if (!dirName) {
      console.warn(`[demo-sites] No demo site mapped to port ${port}`);
      continue;
    }

    const siteDir = path.join(demoSitesRoot, dirName);
    const serverJs = path.join(siteDir, 'server.js');
    const indexHtml = path.join(siteDir, 'index.html');

    if (fs.existsSync(serverJs)) {
      console.log(`[demo-sites] Starting ${dirName} on port ${port} (node server.js)`);
      const proc = spawn('node', [serverJs, String(port)], {
        cwd: siteDir,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      proc.stdout?.on('data', (d: Buffer) => console.log(`[demo-sites:${dirName}] ${d.toString().trim()}`));
      proc.stderr?.on('data', (d: Buffer) => console.error(`[demo-sites:${dirName}] ${d.toString().trim()}`));
      proc.on('exit', (code) => console.log(`[demo-sites] ${dirName} exited with code ${code}`));
      demoSiteProcesses.set(port, proc);
    } else if (fs.existsSync(indexHtml)) {
      // Fallback: serve static directory with a minimal handler
      console.log(`[demo-sites] Starting static server for ${dirName} on port ${port}`);
      const staticServer = http.createServer((req, res) => {
        const filePath = path.join(siteDir, req.url === '/' ? 'index.html' : req.url || 'index.html');
        fs.readFile(filePath, (err, data) => {
          if (err) { res.writeHead(404); res.end('Not found'); return; }
          const ext = path.extname(filePath).toLowerCase();
          const mimeTypes: Record<string, string> = {'.html':'text/html','.css':'text/css','.js':'text/javascript','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml'};
          res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
          res.end(data);
        });
      });
      staticServer.listen(port, '0.0.0.0');
      // Wrap in a pseudo-ChildProcess shape for cleanup
      const fakeProc = { exitCode: null, kill: () => { staticServer.close(); } } as unknown as ChildProcess;
      demoSiteProcesses.set(port, fakeProc);
    } else {
      console.warn(`[demo-sites] ${dirName} has no server.js or index.html, skipping`);
      continue;
    }

    const ready = await waitForPort(port);
    if (ready) {
      console.log(`[demo-sites] ${dirName} ready on port ${port}`);
    } else {
      console.error(`[demo-sites] ${dirName} failed to start on port ${port} within timeout`);
    }
  }
}

// Cleanup demo site processes on exit
function cleanupDemoSites() {
  for (const [port, proc] of demoSiteProcesses) {
    try { proc.kill(); } catch {}
    console.log(`[demo-sites] Stopped process on port ${port}`);
  }
  demoSiteProcesses.clear();
}
process.on('SIGTERM', cleanupDemoSites);
process.on('SIGINT', cleanupDemoSites);
process.on('exit', cleanupDemoSites);


// --- Agent Initialization ---
console.log(`Starting Magnitude BrowserAgent...`);
app.listen(port, () => {
  console.log(`🚀 BrowserAgent service listening on http://localhost:${port}`);
});

const isAgentReady = (req: Request, res: Response, next: Function) => {
  let sessionId = req.body.sessionId;
  if (!sessionId) {
    // Desktop mode is singleton (one physical display, one session).
    // Callers that omit sessionId are targeting the desktop.
    const desktopEntry = [...activeSessions.entries()]
      .find(([, s]) => s.mode === "desktop");
    if (desktopEntry) {
      sessionId = desktopEntry[0];
      req.body.sessionId = sessionId;
    } else {
      return res.status(400).json({ error: 'no_desktop_session', message: 'No active desktop session. Call /start with mode=desktop first.' });
    }
  }
  const session = activeSessions.get(sessionId);
  if (!session) {
    return res.status(404).json({ error: 'session_not_found', message: `Session ${sessionId} not found.` });
  }
  session.lastAccessed = new Date();
  next();
};

const getLaunchOptions = (headless: boolean, downloadsPath: string | null = null, tracesDir: string | null = null) => {
  return { launchOptions: {
    headless: headless,
    args: [
      "--disable-blink-features=AutomationControlled",
      "--disable-features=IsolateOrigins,site-per-process",
      // "--enable-features=WebRtcV4L2VideoCapture",
      // "--auto-select-window-capture-source-by-title=Google",
      '--auto-select-desktop-capture-source="Entire screen"',
    ],
    downloadsPath: downloadsPath || undefined,
    tracesDir: tracesDir || undefined,
  }}
};

const startDesktop = async (): Promise<BrowserAgent> => {
  try {
    const encodedPassword = encodeURIComponent(process.env.UNIFY_KEY || '');
    const desktopUrl = `http://localhost:6080/custom.html?password=${encodedPassword}`;
    const desktopOrigin = new URL(desktopUrl).origin;
    const agent = await startBrowserAgent({
      url: desktopUrl,
      browser: getLaunchOptions(true),
      prompt: "You're controlling a noVNC virtual desktop page. Do not navigate to other page and use mouse and keyboard to control the browser and apps within the virtual desktop. There may be a terminal (xterm) app launched in the desktop for use.",
      narrate: true,
      // Route LLM calls through Orchestra/UniLLM proxy for billing and caching
      llm: {
        provider: 'openai-generic',
        options: {
          model: 'claude-4.6-sonnet@anthropic',
          baseUrl: `${process.env.UNITY_COMMS_URL}/unillm`,
          headers: {
            'Authorization': `Bearer ${process.env.UNIFY_KEY}`,
          },
          temperature: 0.2,
        }
      }
    });
    agent.context.setDefaultNavigationTimeout(90000);
    // Auto-grant clipboard permissions so the noVNC "Share clipboard?" popup is suppressed
    await agent.context.grantPermissions(
      ['clipboard-read', 'clipboard-write'],
      { origin: desktopOrigin },
    );
    console.log("✅ Desktop BrowserAgent started successfully.");
    return agent;
  } catch (err) {
    console.error("❌ Failed to start Desktop BrowserAgent:", err);
    throw err;
  }
}

const startBrowser = async (headless: boolean, urlMappings?: Record<string, string>): Promise<BrowserAgent> => {
  try {
    const agent = await startBrowserAgent({
      url: "https://www.google.com/",
      browser: getLaunchOptions(headless, defaultBrowserPaths.downloadsPath, defaultBrowserPaths.tracesDir),
      narrate: true,
      urlMappings,
      // Route LLM calls through Orchestra/UniLLM proxy for billing and caching
      llm: {
        provider: 'openai-generic',
        options: {
          model: 'claude-4.6-sonnet@anthropic',
          baseUrl: `${process.env.UNITY_COMMS_URL}/unillm`,
          headers: {
            'Authorization': `Bearer ${process.env.UNIFY_KEY}`,
          },
          temperature: 0.2,
        }
      }
    });
    agent.context.setDefaultNavigationTimeout(90000);
    console.log("✅ BrowserAgent started successfully.");
    return agent;
  } catch (err) {
    console.error("❌ Failed to start BrowserAgent:", err);
    throw err;
  }
}

const startBrowserOnVm = async (urlMappings?: Record<string, string>): Promise<BrowserAgent> => {
  try {
    const agent = await startBrowserAgent({
      url: "https://www.google.com/",
      browser: {
        launchOptions: {
          headless: false,
          args: [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            '--auto-select-desktop-capture-source="Entire screen"',
          ],
          downloadsPath: defaultBrowserPaths.downloadsPath || undefined,
          tracesDir: defaultBrowserPaths.tracesDir || undefined,
        },
        contextOptions: { viewport: null },
      },
      narrate: true,
      urlMappings,
      llm: {
        provider: 'openai-generic',
        options: {
          model: 'claude-4.6-sonnet@anthropic',
          baseUrl: `${process.env.UNITY_COMMS_URL}/unillm`,
          headers: {
            'Authorization': `Bearer ${process.env.UNIFY_KEY}`,
          },
          temperature: 0.2,
        }
      }
    });
    agent.context.setDefaultNavigationTimeout(90000);
    console.log("✅ Web-VM BrowserAgent started successfully.");
    return agent;
  } catch (err) {
    console.error("❌ Failed to start Web-VM BrowserAgent:", err);
    throw err;
  }
}

// --- API Endpoints ---
app.post('/start', async (req: Request, res: Response) => {
  const { headless, mode, label, urlMappings } = req.body;
  if (!mode || !['desktop', 'web', 'web-vm'].includes(mode)) {
    return res.status(400).json({
      error: 'bad_request',
      message:
        'Mode is required and must be "desktop", "web", or "web-vm".',
    });
  }

  // Desktop mode is singleton -- one physical display, one session.
  // Close any existing desktop session before creating a new one.
  if (mode === "desktop") {
    for (const [existingId, existing] of activeSessions.entries()) {
      if (existing.mode === "desktop") {
        console.log(`Replacing existing desktop session: ${existingId}`);
        existing.agent.stop().catch((err: unknown) =>
          console.error(`Error stopping old desktop session: ${err}`)
        );
        activeSessions.delete(existingId);
        broadcastSessionEvent(existingId, 'replaced');
      }
    }
  }

  const sessionId = randomUUID();
  const t0 = Date.now();
  console.log(`[start] BEGIN mode=${mode} sessionId=${sessionId}`);
  try {
    let agent: BrowserAgent;
    const mappings = urlMappings && typeof urlMappings === 'object' ? urlMappings as Record<string, string> : undefined;

    if (mappings) {
      await ensureDemoSites(mappings);
    }

    if (mode === "desktop") {
      agent = await startDesktop();
    } else if (mode === "web-vm") {
      agent = await startBrowserOnVm(mappings);
    } else {
      agent = await startBrowser(headless ?? false, mappings);
    }
    console.log(`[start] agent_created=${Date.now() - t0}ms mode=${mode}`);

    if (label && mode === 'web-vm') {
      try {
        await agent.context.addInitScript(`
          (function() {
            function _injectBadge() {
              if (document.getElementById('__mag_session_badge')) return;
              var b = document.createElement('div');
              b.id = '__mag_session_badge';
              b.textContent = ${JSON.stringify(String(label))};
              b.style.cssText = 'position:fixed;top:4px;right:4px;z-index:2147483647;'
                + 'background:rgba(30,30,30,0.85);color:#fff;padding:2px 8px;'
                + 'font:bold 12px/16px system-ui,sans-serif;border-radius:4px;'
                + 'pointer-events:none;user-select:none;';
              (document.body || document.documentElement).appendChild(b);
            }
            if (document.body) _injectBadge();
            else document.addEventListener('DOMContentLoaded', _injectBadge);
          })();
        `);
      } catch (badgeErr) {
        console.warn(`[start] Badge injection failed: ${badgeErr}`);
      }
    }

    activeSessions.set(sessionId, {
      agent,
      mode,
      createdAt: new Date(),
      lastAccessed: new Date(),
      actHistory: [],
    });

    console.log(`[start] DONE mode=${mode} sessionId=${sessionId} total=${Date.now() - t0}ms active_sessions=${activeSessions.size}`);
    res.json({ status: 'started', sessionId });
  } catch (err) {
    console.error(`[start] ERROR mode=${mode} after ${Date.now() - t0}ms:`, err);
    handleAgentError(err, res);
  }
});

app.post('/nav', isAgentReady, async (req: Request, res: Response) => {
  const { url, sessionId } = req.body;
  if (!url) return res.status(400).json({ error: 'bad_request', message: 'URL is required.' });
  try {
    const session = activeSessions.get(sessionId)!;
    await session.agent.nav(url);
    res.json({ status: 'navigated', url });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/act', isAgentReady, async (req: Request, res: Response) => {
  const { task, sessionId, lineage, verify } = req.body;
  if (!task) return res.status(400).json({ error: 'bad_request', message: 'Task description is required.' });
  try {
    const session = activeSessions.get(sessionId)!;
    const agent = session.agent;
    const actId = makeActId(task);

    const lineageLabel = Array.isArray(lineage) && lineage.length > 0
      ? `[${lineage.join('->')}->desktop.act] `
      : '[desktop.act] ';

    const memory = new AgentMemory({ promptCaching: true });

    // Fresh web/web-vm sessions already have a browser open and loaded.
    // Tell the LLM so it can no-op (return an empty action list) if the
    // task is simply asking to open a browser.
    if (session.actHistory.length === 0 && session.mode !== 'desktop') {
      memory.recordObservation(new Observation(
        'thought' as any,
        'user',
        'This is a freshly created browser session — the browser is already open and loaded. '
        + 'If the task is simply asking to open a browser, open a new browser window, or launch a browser, '
        + 'this has already been accomplished. Return an empty actions list.'
      ));
    }

    if (session.actHistory.length > 0) {
      let injectedCount = 0;
      for (const entry of session.actHistory) {
        memory.recordObservation(new Observation(
          'thought' as any,
          'user',
          `Previously completed task: "${entry.task}"`
        ));
        injectedCount++;
        for (const obs of entry.observations) {
          memory.recordObservation(obs);
          injectedCount++;
        }
      }
      console.log(`${lineageLabel}📋 Injecting history from ${session.actHistory.length} previous acts (${injectedCount} observations)`);
    } else {
      console.log(`${lineageLabel}📋 No prior act history in session`);
    }

    const boundary = memory.observationCount;

    const actT0 = Date.now();
    console.log(`${lineageLabel}🧠 Planning actions for: "${task}"${verify ? ' (verify=true)' : ''}`);

    const actActions = verify
      ? agent.actions
      : agent.actions.filter(a => !a.name.startsWith('task:'));
    const MAX_VERIFY_ITERATIONS = 5;
    const actionTraces: any[] = [];
    const iterationReasonings: string[] = [];
    const iterationPlannedActions: any[][] = [];
    let totalActionsExecuted = 0;

    for (let iteration = 0; iteration < (verify ? MAX_VERIFY_ITERATIONS : 1); iteration++) {
      if (iteration > 0) {
        console.log(`${lineageLabel}🔄 Verify pass ${iteration + 1}: re-observing and re-planning...`);
      }

      await agent.recordConnectorObservations(memory);

      if (MAGNITUDE_DEBUG) {
        try {
          const harness = agent.require(BrowserConnector).getHarness();
          const planImg = await harness.screenshot();
          debugSaveImage(actId, iteration === 0 ? 'planning_screenshot' : `verify_${iteration}_screenshot`, await planImg.toBase64());

          if (session.mode === 'desktop') {
            debugSaveImage(actId, iteration === 0 ? 'native_screenshot' : `verify_${iteration}_native`, nativeScreenshot());
          }
        } catch (debugErr) {
          console.warn(`[debug] Pre-plan screenshot capture failed: ${debugErr}`);
        }
      }

      const context = await agent.buildContext(memory);
      const { reasoning, actions } = await agent.models.partialAct(context, task, [], actActions);

      const planMs = Date.now() - actT0;
      console.log(`${lineageLabel}💭 Reasoning [${planMs}ms]: ${reasoning}`);
      console.log(`${lineageLabel}📋 Planned ${actions.length} action(s): ${actions.map(a => a.variant).join(', ')}`);

      iterationReasonings.push(reasoning);
      iterationPlannedActions.push(actions);
      memory.recordThought(reasoning);

      for (let i = 0; i < actions.length; i++) {
        const action = actions[i];
        const actionDef = agent.identifyAction(action);
        const rendered = actionDef.render(action);
        const detail = JSON.stringify(action);
        console.log(`${lineageLabel}🛠️ Action ${totalActionsExecuted + i + 1}: ${rendered} ${detail}`);

        const actionT0 = Date.now();
        let actionError: string | undefined;
        try {
          await agent.exec(action, memory);
        } catch (err) {
          actionError = err instanceof Error ? err.message : String(err);
          throw err;
        } finally {
          const actionMs = Date.now() - actionT0;
          console.log(`${lineageLabel}✅ Completed ${action.variant} [${actionMs}ms]`);

          const actionTrace: any = {
            index: totalActionsExecuted + i,
            iteration,
            variant: action.variant,
            params: action,
            rendered,
            executionMs: actionMs,
          };
          if (actionError) actionTrace.error = actionError;

          if (MAGNITUDE_DEBUG) {
            try {
              const harness = agent.require(BrowserConnector).getHarness();
              const postImg = await harness.screenshot();
              const coordLabel = ('x' in action && 'y' in action)
                ? `_${action.x}_${action.y}`
                : ('from' in action && typeof action.from === 'object')
                  ? `_${action.from.x}_${action.from.y}`
                  : '';
              const padIdx = String(totalActionsExecuted + i + 1).padStart(3, '0');
              debugSaveImage(
                actId,
                `post_action/${padIdx}_${action.variant.replace(/:/g, '_')}${coordLabel}`,
                await postImg.toBase64(),
              );
            } catch (debugErr) {
              console.warn(`[debug] Post-action screenshot failed: ${debugErr}`);
            }
          }

          actionTraces.push(actionTrace);
        }
      }

      totalActionsExecuted += actions.length;

      const taskDone = actions.some(a => a.variant === 'task:done');
      if (!verify || taskDone) break;
    }

    const totalMs = Date.now() - actT0;
    console.log(`${lineageLabel}🏁 ${totalActionsExecuted} action(s) executed across ${iterationReasonings.length} iteration(s) [${totalMs}ms]`);

    debugSaveTrace(actId, {
      actId,
      task,
      verify: !!verify,
      lineage: lineage ?? [],
      sessionMode: session.mode,
      sessionId,
      reasoning: iterationReasonings.join('\n---\n'),
      plannedActions: iterationPlannedActions,
      actionTraces,
      iterations: iterationReasonings.length,
      totalMs,
      historyDepth: session.actHistory.length,
      observationCountBefore: boundary,
    });

    const newObservations = memory.getObservationsSlice(boundary);
    const filtered = newObservations.filter(obs => {
      const src = obs.source;
      return src.startsWith('thought') || src.startsWith('action:taken:');
    });

    session.actHistory.push({ task, observations: filtered });
    if (session.actHistory.length > ACT_HISTORY_DEPTH) {
      session.actHistory = session.actHistory.slice(-ACT_HISTORY_DEPTH);
    }

    console.log(`[memory-carryover] Stored ${filtered.length} filtered observations for task "${task}" (history: ${session.actHistory.length}/${ACT_HISTORY_DEPTH})`);

    const thoughts = filtered
      .filter(obs => obs.source.startsWith('thought'))
      .map(obs => String(obs.content))
      .join('\n');

    let screenshot = '';
    try {
      const connector = session.agent.require(BrowserConnector);
      const harness = connector.getHarness();
      const rawImage = await harness.screenshot();
      const image = await connector.transformScreenshot(rawImage);
      screenshot = await image.toBase64();
    } catch (screenshotErr) {
      console.warn(`[act] Post-act screenshot failed: ${screenshotErr}`);
    }

    res.json({ status: 'success', summary: thoughts, screenshot });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/execute-actions', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId, actions } = req.body;
  if (!actions || !Array.isArray(actions) || actions.length === 0) {
    return res.status(400).json({
      error: 'bad_request',
      message: 'actions is required and must be a non-empty array of action objects.',
    });
  }

  try {
    const session = activeSessions.get(sessionId)!;
    const agent = session.agent;
    const t0 = Date.now();

    console.log(`[execute-actions] Executing ${actions.length} direct action(s) for session ${sessionId}`);

    await agent.executeTrajectory(actions, { memory: agent.memory, recordObservations: false });

    const execMs = Date.now() - t0;
    console.log(`[execute-actions] ${actions.length} action(s) executed [${execMs}ms]`);

    let screenshot = '';
    let cursorPosition: { x: number; y: number } | null = null;
    try {
      const connector = agent.require(BrowserConnector);
      const harness = connector.getHarness();
      const rawImage = await harness.screenshot();
      const image = await connector.transformScreenshot(rawImage);
      screenshot = await image.toBase64();
      cursorPosition = harness.getCursorPosition();
    } catch (screenshotErr) {
      console.warn(`[execute-actions] Post-execution screenshot failed: ${screenshotErr}`);
    }

    res.json({ status: 'success', screenshot, cursorPosition });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/extract', isAgentReady, async (req: Request, res: Response) => {
  const { instructions, schema, bypassDomProcessing, sessionId } = req.body;
  if (!instructions) {
    return res.status(400).json({ error: 'bad_request', message: 'Extraction instructions are required.' });
  }
  const maxRetries = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const zodSchema = schema ? jsonSchemaToZod(schema) : z.string();
      const session = activeSessions.get(sessionId)!;
      const shouldBypassDomProcessing =
        bypassDomProcessing === true || session.mode === 'desktop';

      // Desktop sessions are rendered through the live noVNC iframe, so DOM
      // expansion is both meaningless and destructive. Always use screenshot-
      // only extraction there, even if the caller forgets to request it.
      if (shouldBypassDomProcessing) {
        const screenshot = await session.agent.require(BrowserConnector).getHarness().screenshot();
        const data = await (session.agent.models as any).extract(instructions, zodSchema as ZodTypeAny, screenshot, '');
        return res.json({ data });
      } else {
        // Use the standard extraction method with DOM processing
        const data = await (session.agent as any).extract(instructions, zodSchema as ZodTypeAny);
        return res.json({ data });
      }
    } catch (err: unknown) {
      lastError = err;
      // Check if the error is related to the LLM returning invalid JSON.
      // Added a check for "Unexpected token" which can also indicate a JSON parsing issue.
      if (err instanceof Error && (err.message.includes('HTTP body is not JSON') || err.message.includes('Unexpected token'))) {
        console.warn(`Attempt ${attempt} failed with a transient JSON parsing error. Retrying in ${attempt}s...`);
        await sleep(attempt * 1000); // Wait a bit longer each time
      } else {
        // If it's a different error, fail immediately
        return handleAgentError(err, res);
      }
    }
  }

  // If all retries failed, handle the last recorded error
  console.error(`All ${maxRetries} retries failed for the extract request.`);
  handleAgentError(lastError, res);
});

app.post('/query', isAgentReady, async (req: Request, res: Response) => {
  const { query, schema, sessionId } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'bad_request', message: 'Query is required.' });
  }
  try {
    const zodSchema: ZodTypeAny = schema ? jsonSchemaToZod(schema) : z.any();
    const session = activeSessions.get(sessionId)!;
    const data: unknown = await (session.agent as any).query(query, zodSchema);
    res.json({ data });
  } catch (err) {
    handleAgentError(err, res);
  }
});

// --- Native desktop screenshot via OS commands ---

function nativeScreenshotCommand(dest: string): string {
  switch (process.platform) {
    case 'win32':
      // PowerShell: capture full primary screen using System.Drawing
      return [
        'powershell.exe -NoProfile -Command "',
        'Add-Type -AssemblyName System.Windows.Forms;',
        'Add-Type -AssemblyName System.Drawing;',
        '$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds;',
        '$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height);',
        '$g = [System.Drawing.Graphics]::FromImage($bmp);',
        '$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size);',
        '$g.Dispose();',
        `$bmp.Save('${dest.replace(/'/g, "''")}');`,
        '$bmp.Dispose();"',
      ].join(' ');
    case 'darwin':
      return `screencapture -x "${dest}"`;
    default:
      // Linux / other Unix — xfce4-screenshooter ships with xfce4-goodies
      // (installed in the desktop Docker image). Falls back to scrot, then
      // ImageMagick's import for non-XFCE environments.
      return `xfce4-screenshooter -f -s "${dest}" 2>/dev/null || scrot "${dest}" 2>/dev/null || import -window root "${dest}"`;
  }
}

function nativeScreenshot(): string {
  const dest = path.join(os.tmpdir(), `unity-screenshot-${randomUUID()}.png`);
  try {
    execSync(nativeScreenshotCommand(dest), { timeout: 10_000 });
    const buf = fs.readFileSync(dest);
    return buf.toString('base64');
  } finally {
    try { fs.unlinkSync(dest); } catch (_) { /* already cleaned or never created */ }
  }
}

let _screenshotInFlight = 0;

app.post('/screenshot', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  _screenshotInFlight++;
  const t0 = Date.now();
  const session = activeSessions.get(sessionId)!;
  console.log(`[screenshot] START session=${sessionId} mode=${session.mode} in_flight=${_screenshotInFlight}`);
  try {
    // Use harness screenshot + transformScreenshot for ALL modes. This ensures the
    // screenshot coordinate space matches the click coordinate space (both go through
    // the Playwright page). For desktop mode, this captures the noVNC page which
    // renders the VM desktop with noVNC's own scaling — the same coordinate space
    // that page.mouse.click() uses.
    const connector = session.agent.require(BrowserConnector);
    const harness = connector.getHarness();
    const tHarness = Date.now();
    console.log(`[screenshot] harness_acquired=${tHarness - t0}ms`);
    const rawImage = await harness.screenshot();
    const tCapture = Date.now();
    console.log(`[screenshot] playwright_capture=${tCapture - tHarness}ms`);
    const image = await connector.transformScreenshot(rawImage);
    const base64Image = await image.toBase64();
    const cursorPosition = harness.getCursorPosition();
    const tEncode = Date.now();
    console.log(`[screenshot] base64_encode=${tEncode - tCapture}ms b64_len=${base64Image.length} total=${tEncode - t0}ms`);

    res.json({ screenshot: base64Image, cursorPosition });
    _screenshotInFlight--;
    console.log(`[screenshot] DONE total=${Date.now() - t0}ms in_flight=${_screenshotInFlight}`);
  } catch (err) {
    _screenshotInFlight--;
    console.error(`[screenshot] ERROR after ${Date.now() - t0}ms in_flight=${_screenshotInFlight}:`, err);
    handleAgentError(err, res, 'screenshot_failed');
  }
});

app.post('/eval', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId, expression } = req.body;
  if (!expression) {
    return res.status(400).json({ error: 'bad_request', message: 'expression is required.' });
  }
  try {
    const session = activeSessions.get(sessionId)!;
    const harness = session.agent.require(BrowserConnector).getHarness();
    const result = await harness.page.evaluate(expression);
    res.json({ result });
  } catch (err) {
    handleAgentError(err, res, 'eval_failed');
  }
});

app.post('/viewport-info', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    const harness = session.agent.require(BrowserConnector).getHarness();
    const page = harness.page;

    const playwrightViewport = page.viewportSize();

    const jsInfo = await page.evaluate(() => ({
      innerWidth: window.innerWidth,
      innerHeight: window.innerHeight,
      outerWidth: window.outerWidth,
      outerHeight: window.outerHeight,
      devicePixelRatio: window.devicePixelRatio,
      scrollX: window.scrollX,
      scrollY: window.scrollY,
      clientWidth: document.documentElement.clientWidth,
      clientHeight: document.documentElement.clientHeight,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
      screenAvailWidth: window.screen.availWidth,
      screenAvailHeight: window.screen.availHeight,
    }));

    const screenshotBuffer = await page.screenshot({ type: 'png' });
    // PNG IHDR: width at bytes 16-19, height at bytes 20-23 (big-endian uint32)
    const rawScreenshotDims = {
      width: screenshotBuffer.readUInt32BE(16),
      height: screenshotBuffer.readUInt32BE(20),
    };

    console.log(`[viewport-info] mode=${session.mode} playwright=${JSON.stringify(playwrightViewport)} js=${JSON.stringify(jsInfo)} rawScreenshot=${JSON.stringify(rawScreenshotDims)}`);

    res.json({
      mode: session.mode,
      playwrightViewport,
      jsViewport: jsInfo,
      rawScreenshotDims,
      rescaledScreenshotDims: {
        width: Math.round(rawScreenshotDims.width / jsInfo.devicePixelRatio),
        height: Math.round(rawScreenshotDims.height / jsInfo.devicePixelRatio),
      },
    });
  } catch (err) {
    handleAgentError(err, res, 'viewport_info_failed');
  }
});

app.post('/state', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    const page = session.agent.page;
    const url = page.url();
    const title = await page.title();
    res.json({ url, title });
  } catch (err) {
    handleAgentError(err, res, 'state_failed');
  }
});

// --- Helper: Get full page content with iframe expansion ---
async function getFullPageContentForExtraction(page: any): Promise<string> {
  // Get all iframe element handles
  const iframeHandles = await page.locator('iframe').elementHandles();

  // Iterate through each iframe handle and expand inline
  for (const iframeHandle of iframeHandles) {
    const frame = await iframeHandle.contentFrame();
    if (frame) {
      const iframeContent = await frame.content();
      await iframeHandle.evaluate((iframeNode: HTMLIFrameElement, { content }: { content: string }) => {
        const div = document.createElement('div');
        const parser = new DOMParser();
        const doc = parser.parseFromString(content, 'text/html');
        while (doc.body.firstChild) {
          div.appendChild(doc.body.firstChild);
        }
        const headElements = doc.head.querySelectorAll('style, link[rel="stylesheet"]');
        headElements.forEach(el => div.appendChild(el.cloneNode(true)));
        div.dataset.expandedFromIframe = 'true';
        div.dataset.iframeSrc = iframeNode.getAttribute('src') || '';
        iframeNode.parentNode?.replaceChild(div, iframeNode);
      }, { content: iframeContent });
    }
  }

  return page.content();
}

// --- /links endpoint: Extract all links from current page ---
app.post('/links', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId, sameDomain, selector } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    const page = session.agent.page;
    const currentUrl = page.url();
    const currentHostname = new URL(currentUrl).hostname;

    // Extract all links via page.evaluate()
    const linkSelector = selector || 'a[href]';
    const links: Array<{ href: string; text: string }> = await page.evaluate((sel: string) => {
      return Array.from(document.querySelectorAll(sel))
        .map(a => ({
          href: (a as HTMLAnchorElement).href,
          text: (a as HTMLAnchorElement).innerText.trim().slice(0, 200)
        }))
        .filter(l => l.href && l.href.startsWith('http'));
    }, linkSelector);

    // Deduplicate by href
    const seen = new Set<string>();
    const uniqueLinks = links.filter(l => {
      if (seen.has(l.href)) return false;
      seen.add(l.href);
      return true;
    });

    // Optional: filter to same domain
    const filtered = sameDomain === true
      ? uniqueLinks.filter(l => {
          try {
            return new URL(l.href).hostname === currentHostname;
          } catch {
            return false;
          }
        })
      : uniqueLinks;

    res.json({
      base_url: new URL(currentUrl).origin,
      current_url: currentUrl,
      links: filtered,
      total: filtered.length
    });
  } catch (err) {
    handleAgentError(err, res, 'links_failed');
  }
});

// --- /content endpoint: Get raw page content (no LLM) ---
app.post('/content', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId, format } = req.body;
  // format: 'html' | 'text' | 'markdown' (default: 'markdown')
  const outputFormat = format || 'markdown';

  try {
    const session = activeSessions.get(sessionId)!;
    const page = session.agent.page;
    const url = page.url();
    const title = await page.title();

    let content: string;

    if (outputFormat === 'text') {
      // Plain text extraction
      content = await page.innerText('body');
    } else if (outputFormat === 'html') {
      // Raw HTML with iframe expansion
      content = await getFullPageContentForExtraction(page);
    } else {
      // Markdown (default) - use magnitude-extract
      const htmlContent = await getFullPageContentForExtraction(page);

      const partitionOptions: PartitionOptions = {
        extractImages: true,
        extractForms: true,
        extractLinks: true,
        skipNavigation: false,
        minTextLength: 3,
        includeOriginalHtml: false,
        includeMetadata: true
      };

      const result = partitionHtml(htmlContent, partitionOptions);

      const markdownOptions: MarkdownSerializerOptions = {
        includeMetadata: false,
        includePageNumbers: false,
        includeElementIds: false,
        includeCoordinates: false,
        preserveHierarchy: true,
        escapeSpecialChars: true,
        includeFormFields: true,
        includeImageMetadata: true
      };

      content = serializeToMarkdown(result, markdownOptions);
    }

    res.json({ url, title, content, format: outputFormat });
  } catch (err) {
    handleAgentError(err, res, 'content_failed');
  }
});

app.post('/stop', async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: 'bad_request', message: 'sessionId is required.' });
  }

  const session = activeSessions.get(sessionId);
  if (!session) {
    return res.status(404).json({ error: 'session_not_found', message: `Session ${sessionId} not found.` });
  }

  try {
    await session.agent.stop();
    activeSessions.delete(sessionId);
    broadcastSessionEvent(sessionId, 'stop');
    res.json({ status: 'stopped' });
    console.log(`BrowserAgent stopped for session ${sessionId}.`);
  } catch (err) {
    handleAgentError(err, res, 'stop_failed');
  }
});

app.post('/interrupt_action', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    session.agent.interrupt();
    res.json({ status: 'interrupted', message: 'The current agent action has been interrupted.' });
  } catch (err) {
    handleAgentError(err, res, 'interrupt_failed');
  }
});

app.post('/pause', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    session.agent.pause();
    res.json({ status: 'paused', message: 'The agent has been paused.' });
  } catch (err) {
    handleAgentError(err, res, 'pause_failed');
  }
});

app.post('/resume', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    session.agent.resume();
    res.json({ status: 'resumed', message: 'The agent has been resumed.' });
  } catch (err) {
    handleAgentError(err, res, 'resume_failed');
  }
});

// --- /exec endpoint: Execute shell commands (use /files first to upload files) ---
app.post('/exec', auth, async (req: Request, res: Response) => {
  const { command, cwd, timeout, shell_mode } = req.body;
  const execId = randomUUID().slice(0, 8);

  if (!command || typeof command !== 'string') {
    return res.status(400).json({ error: 'bad_request', message: 'command is required and must be a string.' });
  }

  const workDir = cwd || LOCAL_ROOT;
  const execTimeout = typeof timeout === 'number' && timeout > 0 ? timeout : DEFAULT_EXEC_TIMEOUT;
  const shellMode: ShellMode = shell_mode === 'cmd' ? 'cmd' : 'powershell';

  try {
    const resolvedWorkDir = path.resolve(workDir);
    await ensureDir(resolvedWorkDir);

    console.log(`[exec] Running command: ${command} (cwd: ${resolvedWorkDir}, timeout: ${execTimeout}ms, shell: ${shellMode}, execId: ${execId})`);
    const result = await executeCommand(command, resolvedWorkDir, execTimeout, shellMode);

    res.json({
      status: result.exitCode === 0 ? 'success' : 'error',
      exitCode: result.exitCode,
      stdout: result.stdout,
      stderr: result.stderr,
      duration: result.duration,
      cwd: resolvedWorkDir,
      execId,
    });
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error(`[exec] Error: ${errorMessage}`);
    res.status(500).json({
      error: 'exec_failed',
      message: errorMessage,
      execId,
    });
  }
});

// --- /files endpoint: Unified file management (JSON + Multipart) ---

// Handler for JSON requests
async function handleFilesJson(req: Request, res: Response) {
  const { action, files, filenames, path: subPath, filename, encoding } = req.body;

  if (!action || typeof action !== 'string') {
    return res.status(400).json({ error: 'bad_request', message: 'action is required.' });
  }

  const baseDir = LOCAL_ROOT;

  try {
    switch (action) {
      case 'save': {
        if (!Array.isArray(files) || files.length === 0) {
          return res.status(400).json({ error: 'bad_request', message: 'files array is required for save action.' });
        }

        const savedFiles: string[] = [];
        for (const file of files) {
          if (!file.filename || typeof file.filename !== 'string') {
            return res.status(400).json({ error: 'bad_request', message: 'Each file must have a filename.' });
          }
          if (typeof file.content !== 'string') {
            return res.status(400).json({ error: 'bad_request', message: 'Each file must have content.' });
          }

          const sanitizedPath = sanitizePath(file.filename, baseDir);
          const fileEncoding = file.encoding === 'base64' ? 'base64' : 'text';
          await writeFileWithEncoding(sanitizedPath, file.content, fileEncoding);
          savedFiles.push(file.filename);
          console.log(`[files] Saved: ${sanitizedPath}`);
        }

        return res.json({ status: 'saved', files: savedFiles });
      }

      case 'delete': {
        if (!Array.isArray(filenames) || filenames.length === 0) {
          return res.status(400).json({ error: 'bad_request', message: 'filenames array is required for delete action.' });
        }

        const deletedFiles: string[] = [];
        for (const fname of filenames) {
          if (typeof fname !== 'string') continue;
          const sanitizedPath = sanitizePath(fname, baseDir);
          try {
            await fs.promises.unlink(sanitizedPath);
            deletedFiles.push(fname);
            console.log(`[files] Deleted: ${sanitizedPath}`);
          } catch (err: any) {
            if (err.code !== 'ENOENT') throw err;
            // File doesn't exist, skip silently
          }
        }

        return res.json({ status: 'deleted', files: deletedFiles });
      }

      case 'list': {
        const listPath = subPath ? sanitizePath(subPath, baseDir) : baseDir;
        await ensureDir(listPath);

        const entries = await fs.promises.readdir(listPath, { withFileTypes: true });
        const fileList = await Promise.all(
          entries.map(async (entry) => {
            const fullPath = path.join(listPath, entry.name);
            const stats = await fs.promises.stat(fullPath);
            return {
              name: entry.name,
              type: entry.isDirectory() ? 'directory' : 'file',
              size: stats.size,
              modified: stats.mtime.toISOString(),
            };
          })
        );

        return res.json({
          path: subPath || '.',
          files: fileList,
        });
      }

      case 'read': {
        if (!filename || typeof filename !== 'string') {
          return res.status(400).json({ error: 'bad_request', message: 'filename is required for read action.' });
        }

        const sanitizedPath = sanitizePath(filename, baseDir);
        const fileEncoding = encoding === 'base64' ? 'base64' : 'text';
        const content = await readFileWithEncoding(sanitizedPath, fileEncoding);

        return res.json({
          filename,
          content,
          encoding: fileEncoding,
        });
      }

      default:
        return res.status(400).json({
          error: 'bad_request',
          message: `Unknown action: ${action}. Valid actions: save, delete, list, read.`,
        });
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error(`[files] Error: ${errorMessage}`);
    res.status(500).json({
      error: 'files_failed',
      message: errorMessage,
    });
  }
}

// Handler for multipart requests (large file uploads)
async function handleFilesMultipart(req: Request, res: Response) {
  const targetDir = (req.body.target_dir as string) || '';
  const uploadedFiles = req.files as Express.Multer.File[];

  if (!uploadedFiles || uploadedFiles.length === 0) {
    return res.status(400).json({ error: 'bad_request', message: 'No files uploaded.' });
  }

  const baseDir = LOCAL_ROOT;
  const savedFiles: string[] = [];
  const errors: string[] = [];

  for (const file of uploadedFiles) {
    try {
      const originalName = file.originalname;
      const destFilename = targetDir ? `${targetDir}/${originalName}` : originalName;

      const destPath = sanitizePath(destFilename, baseDir);
      await ensureDir(path.dirname(destPath));
      await fs.promises.rename(file.path, destPath);

      savedFiles.push(destFilename);
      console.log(`[files] Saved (multipart): ${destPath}`);
    } catch (err) {
      // Clean up temp file on error
      try {
        await fs.promises.unlink(file.path);
      } catch (_e) {
        // ignore cleanup errors
      }

      const errorMessage = err instanceof Error ? err.message : String(err);
      errors.push(`${file.originalname}: ${errorMessage}`);
      console.error(`[files] Error saving ${file.originalname}: ${errorMessage}`);
    }
  }

  if (errors.length > 0 && savedFiles.length === 0) {
    return res.status(500).json({
      error: 'upload_failed',
      message: 'All files failed to upload',
      errors,
    });
  }

  res.json({
    status: errors.length > 0 ? 'partial' : 'saved',
    files: savedFiles,
    errors: errors.length > 0 ? errors : undefined,
  });
}

// Route with content-type detection
app.post('/files', (req: Request, res: Response) => {
  const contentType = req.headers['content-type'] || '';

  if (contentType.includes('multipart/form-data')) {
    // Use multer middleware for multipart uploads
    uploadMiddleware.array('files', 100)(req, res, (err) => {
      if (err) {
        const message = err instanceof Error ? err.message : String(err);
        return res.status(400).json({ error: 'upload_error', message });
      }
      handleFilesMultipart(req, res);
    });
  } else {
    // JSON request
    handleFilesJson(req, res);
  }
});

app.get('/sessions', auth, async (_req: Request, res: Response) => {
  const sessions = Array.from(activeSessions.entries()).map(([sessionId, session]) => ({
    sessionId,
    mode: session.mode,
    createdAt: session.createdAt,
    lastAccessed: session.lastAccessed,
  }));
  res.json({ sessions });
});


function handleAgentError(err: unknown, res: Response, defaultErrorType = 'unknown') {
  if (err instanceof AgentError) {
    const agentErr = err as Error & { options: { variant: string; adaptable?: boolean } };
    console.error(`AgentError (${agentErr.options.variant}): ${agentErr.message}`);
    res.status(400).json({
      error: agentErr.options.variant,
      message: agentErr.message,
      adaptable: agentErr.options.adaptable
    });
  } else {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error(`Unknown Error: ${errorMessage}`);
    res.status(500).json({
      error: defaultErrorType,
      message: errorMessage
    });
  }
}

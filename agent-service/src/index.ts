import express, { Request, Response } from 'express';
import https from 'https';
import http from 'http';
import expressWs from 'express-ws';
import WebSocket from 'ws';
import util from 'util';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions } from 'magnitude-core';
import { z, ZodTypeAny, ZodAny, ZodType } from 'zod';
import { partitionHtml, serializeToMarkdown, PartitionOptions, MarkdownSerializerOptions } from 'magnitude-extract';
import dotenv from 'dotenv';
dotenv.config();
import os from 'os';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import { spawn } from 'child_process';
import multer from 'multer';
import { jsonSchemaToZod } from './jsonSchemaToZod';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

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

// Execute command in interactive user session (for COM automation like Excel)
async function executeCommandInUserSession(
  command: string,
  cwd: string,
  timeout: number,
  execId: string
): Promise<ExecResult> {
  const startTime = Date.now();
  const taskName = `unity_exec_${execId}`;
  const scriptFile = path.join(LOCAL_ROOT, `_script_${execId}.ps1`);
  const resultFile = path.join(LOCAL_ROOT, `_result_${execId}.json`);

  // Escape single quotes for PowerShell
  const escapedCwd = cwd.replace(/\\/g, '\\\\').replace(/'/g, "''");
  const escapedCommand = command.replace(/'/g, "''");
  const escapedResultFile = resultFile.replace(/\\/g, '\\\\');

  // PowerShell script that executes command and saves results to JSON
  const scriptContent = `
$ErrorActionPreference = 'Continue'
$startTime = Get-Date
$stdout = ''
$stderr = ''
$exitCode = 0

try {
    Set-Location -Path '${escapedCwd}'
    $output = Invoke-Expression '${escapedCommand}' 2>&1
    $stdout = ($output | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "\`n"
    $stderr = ($output | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }) -join "\`n"
} catch {
    $stderr = $_.Exception.Message
    $exitCode = 1
}

$duration = ((Get-Date) - $startTime).TotalMilliseconds

$resultJson = @{
    exitCode = $exitCode
    stdout = $stdout
    stderr = $stderr
    duration = [int]$duration
} | ConvertTo-Json

# Write without BOM (Out-File adds BOM which breaks JSON.parse in Node.js)
[System.IO.File]::WriteAllText('${escapedResultFile}', $resultJson, [System.Text.UTF8Encoding]::new($false))
`;

  await writeFileWithEncoding(scriptFile, scriptContent, 'text');

  const escapedScriptFile = scriptFile.replace(/\\/g, '\\\\');

  // PowerShell script to create and run scheduled task in user session
  const createTaskScript = `
$taskName = '${taskName}'
$scriptPath = '${escapedScriptFile}'

# Get the currently logged-in user
$loggedInUser = (Get-WmiObject -Class Win32_ComputerSystem).UserName

if (-not $loggedInUser) {
    Write-Error 'No user logged in'
    exit 1
}

Write-Host "Running task as user: $loggedInUser"

# Create scheduled task action (hidden window)
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \`"$scriptPath\`""

# Create principal for interactive user session
$principal = New-ScheduledTaskPrincipal -UserId $loggedInUser -LogonType Interactive -RunLevel Highest

# Register and run the task
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $taskName

# Wait for task to complete
$maxWait = ${timeout}
$waited = 0
while ($waited -lt $maxWait) {
    Start-Sleep -Milliseconds 500
    $waited += 500
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task.State -eq 'Ready') {
        break
    }
}

# Cleanup task
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
`;

  return new Promise((resolve) => {
    const proc = spawn(createTaskScript, [], {
      shell: 'powershell.exe',
      cwd: LOCAL_ROOT,
      timeout,
    });

    let createStdout = '';
    let createStderr = '';

    proc.stdout.on('data', (data) => {
      createStdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      createStderr += data.toString();
    });

    proc.on('close', async () => {
      let result: ExecResult = {
        exitCode: 1,
        stdout: createStdout,
        stderr: createStderr || 'Task execution failed',
        duration: Date.now() - startTime,
      };

      // Wait a moment for result file to be written
      await new Promise(r => setTimeout(r, 1000));

      try {
        const resultJson = await fs.promises.readFile(resultFile, 'utf-8');
        const parsed = JSON.parse(resultJson);
        result = {
          exitCode: parsed.exitCode ?? 0,
          stdout: parsed.stdout ?? '',
          stderr: parsed.stderr ?? '',
          duration: parsed.duration ?? (Date.now() - startTime),
        };
      } catch (e) {
        result.stderr += `\nFailed to read result file: ${e}`;
      }

      // Cleanup temp files
      try {
        await fs.promises.unlink(scriptFile);
      } catch (_e) { /* ignore */ }
      try {
        await fs.promises.unlink(resultFile);
      } catch (_e) { /* ignore */ }

      resolve(result);
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

// Session registry: maps sessionId to BrowserAgent
interface SessionInfo {
  agent: BrowserAgent;
  mode: 'web' | 'desktop';
  createdAt: Date;
  lastAccessed: Date;
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


// --- Agent Initialization ---
console.log(`Starting Magnitude BrowserAgent...`);
app.listen(port, () => {
  console.log(`🚀 BrowserAgent service listening on http://localhost:${port}`);
});

const isAgentReady = (req: Request, res: Response, next: Function) => {
  const sessionId = req.body.sessionId;
  if (!sessionId) {
    return res.status(400).json({ error: 'bad_request', message: 'sessionId is required.' });
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
    const agent = await startBrowserAgent({
      url: `http://localhost:6080/custom.html?password=${process.env.UNIFY_KEY}`,
      browser: getLaunchOptions(true),
      prompt: "You're controlling a noVNC virtual desktop page. Do not navigate to other page and use mouse and keyboard to control the browser and apps within the virtual desktop. There may be a terminal (xterm) app launched in the desktop for use.",
      narrate: true,
      // Route LLM calls through Orchestra/UniLLM proxy for billing and caching
      llm: {
        provider: 'openai-generic',
        options: {
          model: 'claude-4.5-opus@anthropic',
          baseUrl: `${process.env.UNITY_COMMS_URL}/unillm`,
          headers: {
            'Authorization': `Bearer ${process.env.UNIFY_KEY}`,
          },
          temperature: 0.2,
        }
      }
    });
    agent.context.setDefaultNavigationTimeout(90000);
    console.log("✅ Desktop BrowserAgent started successfully.");
    return agent;
  } catch (err) {
    console.error("❌ Failed to start Desktop BrowserAgent:", err);
    throw err;
  }
}

const startBrowser = async (headless: boolean): Promise<BrowserAgent> => {
  try {
    const agent = await startBrowserAgent({
      url: "https://www.duckduckgo.com/",
      browser: getLaunchOptions(headless, defaultBrowserPaths.downloadsPath, defaultBrowserPaths.tracesDir),
      narrate: true,
      // Route LLM calls through Orchestra/UniLLM proxy for billing and caching
      llm: {
        provider: 'openai-generic',
        options: {
          model: 'claude-4.5-opus@anthropic',
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

// --- API Endpoints ---
app.post('/start', async (req: Request, res: Response) => {
  const { headless, mode } = req.body;
  if (!mode || (mode !== "desktop" && mode !== "web")) {
    return res.status(400).json({
      error: 'bad_request',
      message:
        'Mode is required and must be either "desktop" or "web".',
    });
  }

  const sessionId = randomUUID();
  try {
    let agent: BrowserAgent;
    if (mode === "desktop") {
      agent = await startDesktop();
    } else {
      agent = await startBrowser(headless ?? false);
    }

    activeSessions.set(sessionId, {
      agent,
      mode,
      createdAt: new Date(),
      lastAccessed: new Date(),
    });

    res.json({ status: 'started', sessionId });
  } catch (err) {
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
  const { task, sessionId, override_cache } = req.body;
  if (!task) return res.status(400).json({ error: 'bad_request', message: 'Task description is required.' });
  try {
    const session = activeSessions.get(sessionId)!;
    await session.agent.act(task, { override_cache: override_cache === true } as any);
    res.json({ status: 'success', message: `Task "${task}" completed.` });
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

      // If bypassDomProcessing is true, use screenshot-only extraction
      if (bypassDomProcessing === true) {
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
    const queryFn = (session.agent as unknown as { query: (q: unknown, s: ZodTypeAny) => Promise<unknown> }).query;
    const dataUnknown: unknown = await queryFn(query, zodSchema);
    res.json({ data: dataUnknown });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/screenshot', isAgentReady, async (req: Request, res: Response) => {
  const { sessionId } = req.body;
  try {
    const session = activeSessions.get(sessionId)!;
    const harness = session.agent.require(BrowserConnector).getHarness();
    const image = await harness.screenshot();
    const base64Image = await image.toBase64();
    res.json({ screenshot: base64Image });
  } catch (err) {
    handleAgentError(err, res, 'screenshot_failed');
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

// --- /exec endpoint: Execute shell commands (use /files first to upload files) ---
// Pass user_session=true for commands that need interactive session (Excel, COM automation)
app.post('/exec', async (req: Request, res: Response) => {
  const { command, cwd, timeout, shell_mode, user_session } = req.body;
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

    let result: ExecResult;

    // Use user_session=true for commands that need interactive session (Excel, COM, etc.)
    if (user_session === true && process.platform === 'win32') {
      console.log(`[exec] Running in USER SESSION: ${command} (cwd: ${resolvedWorkDir}, execId: ${execId})`);
      result = await executeCommandInUserSession(command, resolvedWorkDir, execTimeout, execId);
    } else {
      console.log(`[exec] Running command: ${command} (cwd: ${resolvedWorkDir}, timeout: ${execTimeout}ms, shell: ${shellMode}, execId: ${execId})`);
      result = await executeCommand(command, resolvedWorkDir, execTimeout, shellMode);
    }

    res.json({
      status: result.exitCode === 0 ? 'success' : 'error',
      exitCode: result.exitCode,
      stdout: result.stdout,
      stderr: result.stderr,
      duration: result.duration,
      cwd: resolvedWorkDir,
      execId,
      userSession: user_session === true,
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

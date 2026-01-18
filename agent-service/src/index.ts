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

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// --- JSON Schema to Zod Conversion Utility ---
function jsonSchemaToZod(schema: any, definitions: any = {}, visitedRefs = new Set<string>()): ZodTypeAny {
  if (typeof schema !== 'object' || schema === null) {
    return z.any();
  }

  // Use root definitions if provided, otherwise extract from the current schema
  const defs = Object.keys(definitions).length > 0 ? definitions : (schema.$defs || schema.definitions || {});

  // Handle references and recursion
  if (schema.$ref) {
    const refName = schema.$ref;
    if (visitedRefs.has(refName)) {
      // If we've seen this ref in the current path, it's a recursive type.
      // We return a lazy schema that will resolve later.
      return z.lazy(() => jsonSchemaToZod({$ref: refName}, defs, new Set([...visitedRefs])));
    }

    visitedRefs.add(refName);

    const refPath = refName.split('/');
    const defName = refPath.pop();
    const resolvedSchema = defs[defName];

    if (!resolvedSchema) {
      throw new Error(`Could not resolve schema reference: ${refName}`);
    }
    // Pass the definitions down to the recursive call
    return jsonSchemaToZod(resolvedSchema, defs, visitedRefs);
  }

  // Handle unions and optionals
  if (schema.anyOf) {
    const nonNullTypes = schema.anyOf.filter((s: any) => s.type !== 'null');

    // Check if this is a simple optional type (e.g., string | null)
    if (schema.anyOf.length > nonNullTypes.length && nonNullTypes.length === 1) {
      const baseSchema = { ...schema, ...nonNullTypes[0] };
      delete baseSchema.anyOf; // Prevent infinite recursion

      // Recursively call jsonSchemaToZod on the now-complete schema and make it optional
      return jsonSchemaToZod(baseSchema, defs, visitedRefs).optional().nullable();
    }

    // Fallback for more complex unions (e.g., string | number)
    const unionTypes = schema.anyOf.map((s: any) => jsonSchemaToZod(s, defs, visitedRefs));
    return z.union(unionTypes as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle type arrays
  if (Array.isArray(schema.type)) {
      // This is another common pattern for Optional fields.
      const hasNull = schema.type.includes('null');
      const nonNullTypes = schema.type.filter((t: string) => t !== 'null');

      if (hasNull && nonNullTypes.length === 1) {
          // This handles cases like `type: ['number', 'null']`
          const baseType = jsonSchemaToZod({ ...schema, type: nonNullTypes[0] }, defs, visitedRefs);
          return baseType.optional().nullable();
      }

      const types = schema.type.map((type: string) => jsonSchemaToZod({ ...schema, type }, defs, visitedRefs));
      return z.union(types as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle enums and literals
  if (schema.enum) {
    if (schema.enum.length === 1) return z.literal(schema.enum[0]);
    const isStringEnum = schema.enum.every((item: any) => typeof item === 'string');
    if (isStringEnum) return z.enum(schema.enum as [string, ...string[]]);
    return z.union(schema.enum.map((item: any) => z.literal(item)));
  }
  if (schema.const) return z.literal(schema.const);

  switch (schema.type) {
    case 'string': {
      let zodString = z.string();
      if (schema.minLength !== undefined) zodString = zodString.min(schema.minLength);
      if (schema.maxLength !== undefined) zodString = zodString.max(schema.maxLength);
      if (schema.pattern) zodString = zodString.regex(new RegExp(schema.pattern));
      if (schema.format === 'email') zodString = zodString.email();
      if (schema.format === 'uuid') zodString = zodString.uuid();
      if (schema.format === 'uri' || schema.format === 'url') zodString = zodString.url();
      if (schema.format === 'date-time') zodString = zodString.datetime();
      return zodString;
    }
    case 'number':
    case 'integer': {
      let zodNum = schema.type === 'integer' ? z.number().int() : z.number();
      if (schema.minimum !== undefined) zodNum = zodNum.gte(schema.minimum);
      if (schema.exclusiveMinimum !== undefined) zodNum = zodNum.gt(schema.exclusiveMinimum);
      if (schema.maximum !== undefined) zodNum = zodNum.lte(schema.maximum);
      if (schema.exclusiveMaximum !== undefined) zodNum = zodNum.lt(schema.exclusiveMaximum);
      if (schema.multipleOf !== undefined) zodNum = zodNum.multipleOf(schema.multipleOf);
      return zodNum;
    }
    case 'boolean': return z.boolean();
    case 'null': return z.null();
    case 'array': {
      let itemSchema: ZodTypeAny = z.any();
      if (schema.items) {
        itemSchema = jsonSchemaToZod(schema.items, defs, visitedRefs);
      }
      let zodArray = z.array(itemSchema);
      if (schema.minItems !== undefined) zodArray = zodArray.min(schema.minItems);
      if (schema.maxItems !== undefined) zodArray = zodArray.max(schema.maxItems);
      return zodArray;
    }
    case 'object': {
      const shape: { [key: string]: ZodTypeAny } = {};
      if (schema.properties) {
        for (const key in schema.properties) {
          const propSchema = jsonSchemaToZod(schema.properties[key], defs, visitedRefs);
          shape[key] = schema.required?.includes(key) ? propSchema : propSchema.optional();
        }
      }
      let zodObject: ZodTypeAny = z.object(shape);
      if (schema.additionalProperties === false) {
        zodObject = z.object(shape).strict();
      } else if (typeof schema.additionalProperties === 'object') {
        zodObject = z.object(shape).catchall(jsonSchemaToZod(schema.additionalProperties, defs, visitedRefs));
      }
      return zodObject;
    }
  }

  if (schema.properties) return jsonSchemaToZod({ ...schema, type: 'object' }, defs, visitedRefs);

  return z.any();
}

function getDefaultBrowserPaths() {
  const base = path.join(os.tmpdir(), 'unify', 'assistant', 'browser');
  const downloadsPath = path.join(base, 'install');
  const tracesDir = path.join(base, 'traces');
  try {
    fs.mkdirSync(downloadsPath, { recursive: true });
    fs.mkdirSync(tracesDir, { recursive: true });
  } catch (_e) {
    // ignore directory creation errors; downstream may still handle
  }
  return { downloadsPath, tracesDir };
}

const defaultBrowserPaths = getDefaultBrowserPaths();

const app = express();
const wsInstance = expressWs(app);
app.use(express.json({ limit: '10mb' }));

// --- Authorization (Bearer) middleware ---
function verifyApiKeyWithUnify(apiKey: string): Promise<boolean> {
  return new Promise((resolve) => {
    const url = new URL(`${process.env.UNIFY_BASE_URL}/user/basic-info`);
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
  const apiKey = match[1];

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
  mode: 'browser' | 'desktop';
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

  const apiKey = match[1];

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
  if (!mode || (mode !== "desktop" && mode !== "browser")) {
    return res.status(400).json({ error: 'bad_request', message: 'Mode is required and must be either "desktop" or "browser".' });
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

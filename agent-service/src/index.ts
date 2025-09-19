import express, { Request, Response } from 'express';
import https from 'https';
import http from 'http';
import expressWs from 'express-ws';
import WebSocket from 'ws';
import util from 'util';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions } from 'magnitude-core';
import { z, ZodTypeAny, ZodAny, ZodType } from 'zod';
import dotenv from 'dotenv';
dotenv.config();
import os from 'os';
import path from 'path';
import fs from 'fs';

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
function verifyApiKeyWithUnify(apiKey: string, assistant_email: string): Promise<boolean> {
  return new Promise((resolve) => {
    const url = new URL(`${process.env.UNIFY_BASE_URL}/assistant?email=${assistant_email}`);
    const options = {
      method: 'GET',
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname + url.search,
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    };

    // Use the appropriate request method based on protocol
    const requestLib = url.protocol === 'https:' ? https : http;
    const req = requestLib.request(options, (res) => {
      const code = res.statusCode || 0;
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        if (!(code >= 200 && code < 300)) return resolve(false);
        if (!body || body.trim().length === 0) return resolve(false);
        try {
          // Using default assistant for testing, auth passes since apikey is valid
          if (assistant_email.includes('agent') || assistant_email.includes('assistant')) {
            return resolve(true);
          }

          const json = JSON.parse(body);
          // Treat empty payloads as invalid: {"info": []}, {}, []
          if (Array.isArray(json)) return resolve(json.length > 0);
          if (json && typeof json === 'object') {
            if (Array.isArray((json as any).info)) return resolve((json as any).info.length > 0);
            return resolve(Object.keys(json).length > 0);
          }
          if (typeof json === 'string') return resolve(json.trim().length > 0);
          return resolve(!!json);
        } catch (_e) {
          // Non-JSON: accept only if non-empty body
          return resolve(body.trim().length > 0);
        }
      });
    });
    req.on('error', (err) => {

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
  const keys = match[1].split(' ');
  const apikey = keys[0];
  const assistant_email = keys[1];

  try {
    const ok = await verifyApiKeyWithUnify(apikey, assistant_email);
    if (!ok) {
      return res.status(401).json({ error: 'unauthorized', message: 'API key verification failed' });
    }
  } catch (e) {

    return res.status(401).json({ error: 'unauthorized', message: 'API key verification failed' });
  }

  next();
}

app.use(auth);

let agentMode: string | null = null;
let browserAgent: BrowserAgent | null = null;
let desktopBrowserAgent: BrowserAgent | null = null;
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

  const keys = match[1].split(' ');
  const apikey = keys[0];
  const assistant_email = keys[1];

  try {
    const ok = await verifyApiKeyWithUnify(apikey, assistant_email);
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
  if (!browserAgent) {
    return res.status(503).json({ error: 'agent_not_ready', message: 'BrowserAgent is not yet initialized.' });
  }
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

const startDesktop = () => {
  startBrowserAgent({
    url: `http://localhost:6080/vnc.html?resize=scale&autoreconnect=1&autoconnect=1&password=${process.env.UNIFY_KEY}`,
    browser: getLaunchOptions(true),
    prompt: "You're controlling a noVNC virtual desktop page. Do not navigate to other page and use mouse and keyboard to control the browser and apps within the virtual desktop. There may be a terminal (xterm) app launched in the desktop for use.",
    narrate: true,
  }).then(agent => {
    browserAgent = agent;
    console.log("✅ BrowserAgent started successfully.");
  }).catch(err => {
    console.error("❌ Failed to start BrowserAgent:", err);
    process.exit(1);
  });

  startBrowserAgent({
    url: "https://www.duckduckgo.com/",
    browser: getLaunchOptions(false, defaultBrowserPaths.downloadsPath, defaultBrowserPaths.tracesDir),
  }).then(agent => {
    desktopBrowserAgent = agent;
    console.log("✅ Desktop BrowserAgent started successfully.");
  }).catch(err => {
    console.error("❌ Failed to start Desktop BrowserAgent:", err);
    process.exit(1);
  });
}

const startBrowser = (headless: boolean) => {
  startBrowserAgent({
    url: "https://www.duckduckgo.com/",
    browser: getLaunchOptions(headless, defaultBrowserPaths.downloadsPath, defaultBrowserPaths.tracesDir),
    narrate: true,
  }).then(agent => {
    browserAgent = agent;
    console.log("✅ BrowserAgent started successfully.");
  }).catch(err => {
    console.error("❌ Failed to start BrowserAgent:", err);
    process.exit(1);
  });
}

// --- API Endpoints ---
app.post('/start', async (req: Request, res: Response) => {
  const { headless, mode } = req.body;
  if (!mode || (mode !== "desktop" && mode !== "browser")) return res.status(400).json({ error: 'bad_request', message: 'Mode is required and must be either "desktop" or "browser".' });
  agentMode = mode;
  try {
    if (agentMode === "desktop") {
      startDesktop();
    } else {
      startBrowser(headless);
    }
    res.json({ status: 'started' });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/nav', isAgentReady, async (req: Request, res: Response) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: 'bad_request', message: 'URL is required.' });
  try {
    await browserAgent!.nav(url);
    res.json({ status: 'navigated', url });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/act', isAgentReady, async (req: Request, res: Response) => {
  const { task } = req.body;
  if (!task) return res.status(400).json({ error: 'bad_request', message: 'Task description is required.' });
  try {
    await browserAgent!.act(task);
    res.json({ status: 'success', message: `Task "${task}" completed.` });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.post('/extract', isAgentReady, async (req: Request, res: Response) => {
  const { instructions, schema } = req.body;
  if (!instructions) {
    return res.status(400).json({ error: 'bad_request', message: 'Extraction instructions are required.' });
  }
  const maxRetries = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const zodSchema = schema ? jsonSchemaToZod(schema) : z.string();
      const extractFn = (browserAgent as unknown as { extract: (i: unknown, s: ZodTypeAny) => Promise<unknown> }).extract;
      const dataUnknown: unknown = await extractFn(instructions, zodSchema as ZodTypeAny);
      const data = dataUnknown as z.infer<typeof zodSchema>;
      // If successful, send the response and exit the loop
      return res.json({ data });
    } catch (err: unknown) {
      lastError = err;
      // Check if the error is related to the LLM returning invalid JSON
      if (err instanceof Error && err.message.includes('HTTP body is not JSON')) {
        console.warn(`Attempt ${attempt} failed with a transient error. Retrying in ${attempt}s...`);
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
  const { query, schema } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'bad_request', message: 'Query is required.' });
  }
  try {
    const zodSchema = schema ? jsonSchemaToZod(schema) : z.any();
    const queryFn = (browserAgent as unknown as { query: (q: unknown, s: ZodTypeAny) => Promise<unknown> }).query;
    const dataUnknown: unknown = await queryFn(query, zodSchema as ZodTypeAny);
    const data = dataUnknown as z.infer<typeof zodSchema>;
    res.json({ data });
  } catch (err) {
    handleAgentError(err, res);
  }
});

app.get('/screenshot', isAgentReady, async (_req: Request, res: Response) => {
  try {
    const harness = browserAgent!.require(BrowserConnector).getHarness();
    const image = await harness.screenshot();
    const base64Image = await image.toBase64();
    res.json({ screenshot: base64Image });
  } catch (err) {
    handleAgentError(err, res, 'screenshot_failed');
  }
});

app.get('/state', isAgentReady, async (_req: Request, res: Response) => {
  try {
    const page = browserAgent!.page;
    const url = page.url();
    const title = await page.title();
    res.json({ url, title });
  } catch (err) {
    handleAgentError(err, res, 'state_failed');
  }
});

app.post('/stop', isAgentReady, async (_req: Request, res: Response) => {
  try {
    if (agentMode === "desktop") {
      await desktopBrowserAgent!.stop();
      desktopBrowserAgent = null;
      console.log("Desktop BrowserAgent stopped.");
    }
    await browserAgent!.stop();
    browserAgent = null;
    agentMode = null;
    res.json({ status: 'stopped' });
    console.log("BrowserAgent stopped.");
  } catch (err) {
    handleAgentError(err, res, 'stop_failed');
  }
});

app.post('/interrupt_action', isAgentReady, async (_req: Request, res: Response) => {
  try {
    if (browserAgent) {
      browserAgent.interrupt();
      res.json({ status: 'interrupted', message: 'The current agent action has been interrupted.' });
    } else {
      res.status(404).json({ error: 'agent_not_found', message: 'No active agent to interrupt.' });
    }
  } catch (err) {
    handleAgentError(err, res, 'interrupt_failed');
  }
});


function handleAgentError(err: unknown, res: Response, defaultErrorType = 'unknown') {
  if (err instanceof AgentError) {
    console.error(`AgentError (${err.options.variant}): ${err.message}`);
    res.status(400).json({
      error: err.options.variant,
      message: err.message,
      adaptable: err.options.adaptable
    });
  } else {
    console.error(`Unknown Error: ${String(err)}`);
    res.status(500).json({
      error: defaultErrorType,
      message: String(err)
    });
  }
}

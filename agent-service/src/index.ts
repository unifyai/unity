import express, { Request, Response } from 'express';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions } from 'magnitude-core';
import { z, ZodTypeAny, ZodAny } from 'zod';
import dotenv from 'dotenv';
dotenv.config();
import codesandboxRouter from './codesandbox';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
// --- Helper to parse command-line arguments ---
const args = process.argv.slice(2);
const isHeadless = args.includes('--headless');

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

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use('/codesandbox', codesandboxRouter);

let browserAgent: BrowserAgent | null = null;
const port = process.env.PORT || 3000;

// --- Agent Initialization ---
console.log(`Starting Magnitude BrowserAgent (Headless: ${isHeadless})...`);

const browserOptions: BrowserOptions = {
    launchOptions: {
        headless: isHeadless,
        args: [
          "--disable-blink-features=AutomationControlled",
          "--disable-features=IsolateOrigins,site-per-process",
          // "--enable-features=WebRtcV4L2VideoCapture",
          "--auto-select-window-capture-source-by-title=Google",
        ],
        downloadsPath: `/home/agent/Downloads`,
        tracesDir: `/home/agent/traces`,
    },
};

// Starts desktop agent
startBrowserAgent({
    url: "http://localhost:6080/vnc.html?resize=scale&autoconnect=1",
    browser: browserOptions,
    prompt: "You're controlling a noVNC virtual desktop page. Do not navigate to other page and use mouse and keyboard to control the terminal, apps and browser within the virtual desktop.",
  }).then(agent => {
// startBrowserAgent({ browser: browserOptions })
//   .then(agent => {
    browserAgent = agent;
    console.log("✅ BrowserAgent started successfully.");
    app.listen(port, () => {
      console.log(`🚀 BrowserAgent service listening on http://localhost:${port}`);
    });
  })
  .catch(err => {
    console.error("❌ Failed to start BrowserAgent:", err);
    process.exit(1);
  });

// Browser to be controlled through desktop
startBrowserAgent({
  url: "https://www.duckduckgo.com/",
  browser: {
    launchOptions: {
      headless: false,
      downloadsPath: `/home/browser/Downloads`,
      tracesDir: `/home/browser/traces`,
    }
  }})

const isAgentReady = (req: Request, res: Response, next: Function) => {
  if (!browserAgent) {
    return res.status(503).json({ error: 'agent_not_ready', message: 'BrowserAgent is not yet initialized.' });
  }
  next();
};

// --- API Endpoints ---
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
      const data = await browserAgent!.extract(instructions, zodSchema);
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
    await browserAgent!.stop();
    browserAgent = null;
    res.json({ status: 'stopped' });
    console.log("BrowserAgent stopped. Server will now exit.");
    setTimeout(() => process.exit(0), 100);
  } catch (err) {
    handleAgentError(err, res, 'stop_failed');
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

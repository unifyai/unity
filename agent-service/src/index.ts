import express, { Request, Response } from 'express';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions } from 'magnitude-core';
import { z, ZodTypeAny } from 'zod';
import dotenv from 'dotenv';
dotenv.config();

// --- Helper to parse command-line arguments ---
const args = process.argv.slice(2);
const isHeadless = args.includes('--headless');

// --- JSON Schema to Zod Conversion Utility ---
function jsonSchemaToZod(schema: any): ZodTypeAny {
  switch (schema.type) {
    case 'string':
      if (schema.enum) {
        return z.enum(schema.enum);
      }
      return z.string();
    case 'number':
      return z.number();
    case 'integer':
      return z.number().int();
    case 'boolean':
      return z.boolean();
    case 'null':
      return z.null();
    case 'array':
      if (!schema.items) {
        throw new Error('Array schema must have an "items" property.');
      }
      return z.array(jsonSchemaToZod(schema.items));
    case 'object':
      const shape: { [key: string]: ZodTypeAny } = {};
      if (schema.properties) {
        for (const key in schema.properties) {
          const propSchema = jsonSchemaToZod(schema.properties[key]);
          const isRequired = schema.required?.includes(key);
          shape[key] = isRequired ? propSchema : propSchema.optional();
        }
      }
      return z.object(shape);
    default:
      throw new Error(`Unsupported schema type: ${schema.type}`);
  }
}

const app = express();
app.use(express.json({ limit: '10mb' }));

let browserAgent: BrowserAgent | null = null;
const port = process.env.PORT || 3000;

// --- Agent Initialization ---
console.log(`Starting Magnitude BrowserAgent (Headless: ${isHeadless})...`);

const browserOptions: BrowserOptions = {
    launchOptions: {
        headless: isHeadless,
    },
};

startBrowserAgent({ browser: browserOptions })
  .then(agent => {
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


const isAgentReady = (req: Request, res: Response, next: Function) => {
  if (!browserAgent) {
    return res.status(503).json({ error: 'agent_not_ready', message: 'BrowserAgent is not yet initialized.' });
  }
  next();
};

// --- API Endpoints ---

app.get('/health', (_req, res) => {
  if (browserAgent) {
    res.status(200).json({ status: 'ready' });
  } else {
    res.status(503).json({ status: 'initializing' });
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
  try {
    const zodSchema = schema ? jsonSchemaToZod(schema) : z.string();
    const data = await browserAgent!.extract(instructions, zodSchema);
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

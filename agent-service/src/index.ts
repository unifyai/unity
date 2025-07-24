import express, { Request, Response } from 'express';
import { startBrowserAgent, BrowserAgent, BrowserConnector, AgentError, BrowserOptions } from 'magnitude-core';
import { z, ZodTypeAny } from 'zod';
import dotenv from 'dotenv';
dotenv.config();

// --- Helper to parse command-line arguments ---
const args = process.argv.slice(2);
const isHeadless = args.includes('--headless');

// --- JSON Schema to Zod Conversion Utility ---
function jsonSchemaToZod(schema: any, definitions: any = {}): ZodTypeAny {
  // Handle schema references ($ref)
  if (schema.$ref) {
    const refPath = schema.$ref.split('/');
    const defName = refPath.pop();
    const resolvedSchema = definitions[defName];
    if (!resolvedSchema) {
      throw new Error(`Could not resolve schema reference: ${schema.$ref}`);
    }
    return jsonSchemaToZod(resolvedSchema, definitions);
  }

  // Handle const (Literal types)
  if ('const' in schema) {
    return z.literal(schema.const);
  }

  // Handle enum as a special case
  if (schema.enum && Array.isArray(schema.enum)) {
    if (schema.enum.length === 1) {
      return z.literal(schema.enum[0]);
    }
    // For string enums
    if (schema.enum.every((val: any) => typeof val === 'string')) {
      return z.enum(schema.enum as [string, ...string[]]);
    }
    // For mixed enums, use union of literals
    const literals = schema.enum.map((val: any) => z.literal(val));
    return z.union(literals as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle anyOf (unions like Type | None)
  if (schema.anyOf && Array.isArray(schema.anyOf)) {
    const unionTypes = schema.anyOf.map((subSchema: any) =>
      jsonSchemaToZod(subSchema, definitions)
    );
    return z.union(unionTypes as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle oneOf (discriminated unions)
  if (schema.oneOf && Array.isArray(schema.oneOf)) {
    const unionTypes = schema.oneOf.map((subSchema: any) =>
      jsonSchemaToZod(subSchema, definitions)
    );
    return z.union(unionTypes as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle allOf (model inheritance/composition)
  if (schema.allOf && Array.isArray(schema.allOf)) {
    let result: ZodTypeAny | null = null;

    for (const subSchema of schema.allOf) {
      const zodSchema = jsonSchemaToZod(subSchema, definitions);

      if (!result) {
        result = zodSchema;
      } else if (result instanceof z.ZodObject && zodSchema instanceof z.ZodObject) {
        // Merge objects
        result = result.merge(zodSchema);
      } else {
        // For non-objects, intersection
        result = result.and(zodSchema);
      }
    }

    return result || z.object({});
  }

  // Handle type arrays (e.g., type: ["string", "null"])
  if (Array.isArray(schema.type)) {
    const unionTypes = schema.type.map((type: string) =>
      jsonSchemaToZod({ ...schema, type }, definitions)
    );
    return z.union(unionTypes as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle tuple types
  if (schema.type === 'array' && schema.prefixItems) {
    const tupleItems = schema.prefixItems.map((itemSchema: any) =>
      jsonSchemaToZod(itemSchema, definitions)
    );
    return z.tuple(tupleItems as [ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Enhanced type handlers with validation constraints
  const typeMap: { [key: string]: () => ZodTypeAny } = {
    string: () => {
      let zodString = z.string();

      // Apply string constraints
      if (schema.minLength !== undefined) zodString = zodString.min(schema.minLength);
      if (schema.maxLength !== undefined) zodString = zodString.max(schema.maxLength);
      if (schema.pattern) zodString = zodString.regex(new RegExp(schema.pattern));

      // Handle format constraints
      if (schema.format) {
        switch (schema.format) {
          case 'email': zodString = zodString.email(); break;
          case 'uuid': zodString = zodString.uuid(); break;
          case 'uri': zodString = zodString.url(); break;
          case 'url': zodString = zodString.url(); break;
          case 'date': zodString = zodString.regex(/^\d{4}-\d{2}-\d{2}$/); break;
          case 'date-time': zodString = zodString.datetime(); break;
          case 'time': zodString = zodString.regex(/^\d{2}:\d{2}:\d{2}$/); break;
          case 'ipv4': zodString = zodString.ip({ version: 'v4' }); break;
          case 'ipv6': zodString = zodString.ip({ version: 'v6' }); break;
          // Add more formats as needed
        }
      }

      return zodString;
    },

    number: () => {
      let zodNumber = z.number();

      // Apply numeric constraints
      if (schema.minimum !== undefined) {
        zodNumber = schema.exclusiveMinimum === true
          ? zodNumber.gt(schema.minimum)
          : zodNumber.gte(schema.minimum);
      }
      if (schema.maximum !== undefined) {
        zodNumber = schema.exclusiveMaximum === true
          ? zodNumber.lt(schema.maximum)
          : zodNumber.lte(schema.maximum);
      }
      if (schema.multipleOf !== undefined) {
        zodNumber = zodNumber.multipleOf(schema.multipleOf);
      }

      return zodNumber;
    },

    integer: () => {
      let zodInt = z.number().int();

      // Apply integer constraints (same as number)
      if (schema.minimum !== undefined) {
        zodInt = schema.exclusiveMinimum === true
          ? zodInt.gt(schema.minimum)
          : zodInt.gte(schema.minimum);
      }
      if (schema.maximum !== undefined) {
        zodInt = schema.exclusiveMaximum === true
          ? zodInt.lt(schema.maximum)
          : zodInt.lte(schema.maximum);
      }
      if (schema.multipleOf !== undefined) {
        zodInt = zodInt.multipleOf(schema.multipleOf);
      }

      return zodInt;
    },

    boolean: () => z.boolean(),
    null: () => z.null(),
    array: () => {
      // Handle tuple types (fixed-length arrays)
      if (schema.prefixItems) {
        const tupleItems = schema.prefixItems.map((itemSchema: any) =>
          jsonSchemaToZod(itemSchema, definitions)
        );
        return z.tuple(tupleItems as [ZodTypeAny, ...ZodTypeAny[]]);
      }

      if (!schema.items) {
        // Array of any
        return z.array(z.any());
      }

      let zodArray = z.array(jsonSchemaToZod(schema.items, definitions));

      // Apply array constraints
      if (schema.minItems !== undefined) zodArray = zodArray.min(schema.minItems);
      if (schema.maxItems !== undefined) zodArray = zodArray.max(schema.maxItems);
      if (schema.uniqueItems === true) {
        // Note: Zod doesn't have built-in unique validation, but we can add a custom refinement
        return zodArray.refine(
          (items) => new Set(items).size === items.length,
          { message: "Array must contain unique items" }
        );
      }

      return zodArray;
    },

    object: () => {
      const shape: { [key: string]: ZodTypeAny } = {};
      // Handle properties
      if (schema.properties) {
        for (const key in schema.properties) {
          const propSchema = jsonSchemaToZod(schema.properties[key], definitions);
          const isRequired = schema.required?.includes(key);
          shape[key] = isRequired ? propSchema : propSchema.optional();
        }
      }

      let zodObject: ZodTypeAny = z.object(shape);

      // Handle additionalProperties (for Dict types)
      if (schema.additionalProperties === false) {
        zodObject = (zodObject as z.ZodObject<any>).strict();
      } else if (schema.additionalProperties === true) {
        zodObject = (zodObject as z.ZodObject<any>).catchall(z.any());
      } else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
        zodObject = (zodObject as z.ZodObject<any>).catchall(
          jsonSchemaToZod(schema.additionalProperties, definitions)
        );
      }

      // Handle patternProperties
      if (schema.patternProperties) {
        // Note: Zod doesn't directly support pattern properties,
        // but we can use catchall with refinement
        for (const pattern in schema.patternProperties) {
          const regex = new RegExp(pattern);
          const valueSchema = jsonSchemaToZod(schema.patternProperties[pattern], definitions);

          zodObject = (zodObject as z.ZodObject<any>).catchall(z.any()).refine(
            (obj) => {
              for (const key in obj) {
                if (regex.test(key) && !(key in shape)) {
                  try {
                    valueSchema.parse(obj[key]);
                  } catch {
                    return false;
                  }
                }
              }
              return true;
            },
            { message: `Pattern properties must match schema for pattern: ${pattern}` }
          );
        }
      }

      // Handle minProperties/maxProperties
      if (schema.minProperties !== undefined) {
        zodObject = zodObject.refine(
          (obj: any) => Object.keys(obj).length >= schema.minProperties,
          { message: `Object must have at least ${schema.minProperties} properties` }
        );
      }
      if (schema.maxProperties !== undefined) {
        zodObject = zodObject.refine(
          (obj: any) => Object.keys(obj).length <= schema.maxProperties,
          { message: `Object must have at most ${schema.maxProperties} properties` }
        );
      }

      return zodObject;
    },
  };

  // Handle type-specific logic
  if (schema.type && typeMap[schema.type]) {
    return typeMap[schema.type]();
  }

  // Handle schema without explicit type but with properties (common in some generators)
  if (schema.properties && !schema.type) {
    return typeMap.object();
  }

  // If we get here, it's an unsupported schema
  console.warn('Unsupported schema:', JSON.stringify(schema, null, 2));
  throw new Error(`Unsupported schema type: ${schema.type || 'unknown'}`);
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
    const zodSchema = schema ? jsonSchemaToZod(schema, schema.$defs || {}) : z.string();

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

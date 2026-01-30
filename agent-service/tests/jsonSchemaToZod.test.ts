import { strict as assert } from "node:assert";
import { jsonSchemaToZod } from "../src/jsonSchemaToZod";

function run(name: string, fn: () => void) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (err) {
    console.error(`fail - ${name}`);
    console.error(err);
    process.exitCode = 1;
  }
}

/**
 * Convert-to-Zod should be stable for schemas that reuse `$ref` definitions.
 *
 * This is common for JSON Schemas emitted by Pydantic where the same submodel
 * appears in multiple fields and containers.
 */
run("reused $ref should not stack overflow", () => {
  const schema = {
    $defs: {
      JobListing: {
        type: "object",
        properties: {
          title: { type: "string" },
        },
        required: ["title"],
        additionalProperties: false,
      },
    },
    type: "object",
    properties: {
      first: { $ref: "#/$defs/JobListing" },
      second: { $ref: "#/$defs/JobListing" },
      many: { type: "array", items: { $ref: "#/$defs/JobListing" } },
    },
    required: ["first", "second", "many"],
    additionalProperties: false,
  };

  const zodSchema = jsonSchemaToZod(schema);

  const parsed = zodSchema.parse({
    first: { title: "a" },
    second: { title: "b" },
    many: [{ title: "c" }],
  });

  assert.deepStrictEqual(parsed, {
    first: { title: "a" },
    second: { title: "b" },
    many: [{ title: "c" }],
  });
});

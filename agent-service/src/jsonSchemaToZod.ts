import { z, ZodTypeAny } from "zod";

/**
 * Convert a JSON Schema (Pydantic-style) into a Zod schema.
 *
 * NOTE:
 * - This function is used by agent-service `/extract` to validate/shape model outputs.
 * - It supports `$defs`/`definitions` and `$ref`.
 */
export function jsonSchemaToZod(
  schema: any,
  definitions: any = {},
  visitedRefs = new Set<string>(),
): ZodTypeAny {
  if (typeof schema !== "object" || schema === null) {
    return z.any();
  }

  // Use root definitions if provided, otherwise extract from the current schema
  const defs =
    Object.keys(definitions).length > 0
      ? definitions
      : schema.$defs || schema.definitions || {};

  // Handle references and recursion
  if (schema.$ref) {
    const refName = schema.$ref;
    if (visitedRefs.has(refName)) {
      // If we've seen this ref in the current path, it's a recursive type.
      // We return a lazy schema that will resolve later.
      return z.lazy(() =>
        jsonSchemaToZod({ $ref: refName }, defs, new Set([...visitedRefs])),
      );
    }

    visitedRefs.add(refName);

    const refPath = refName.split("/");
    const defName = refPath.pop();
    const resolvedSchema = defs[defName as string];

    if (!resolvedSchema) {
      throw new Error(`Could not resolve schema reference: ${refName}`);
    }
    // Pass the definitions down to the recursive call
    return jsonSchemaToZod(resolvedSchema, defs, visitedRefs);
  }

  // Handle unions and optionals
  if (schema.anyOf) {
    const nonNullTypes = schema.anyOf.filter((s: any) => s.type !== "null");

    // Check if this is a simple optional type (e.g., string | null)
    if (schema.anyOf.length > nonNullTypes.length && nonNullTypes.length === 1) {
      const baseSchema = { ...schema, ...nonNullTypes[0] };
      delete baseSchema.anyOf; // Prevent infinite recursion

      // Recursively call jsonSchemaToZod on the now-complete schema and make it optional
      return jsonSchemaToZod(baseSchema, defs, visitedRefs).optional().nullable();
    }

    // Fallback for more complex unions (e.g., string | number)
    const unionTypes = schema.anyOf.map((s: any) =>
      jsonSchemaToZod(s, defs, visitedRefs),
    );
    return z.union(unionTypes as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle type arrays
  if (Array.isArray(schema.type)) {
    // This is another common pattern for Optional fields.
    const hasNull = schema.type.includes("null");
    const nonNullTypes = schema.type.filter((t: string) => t !== "null");

    if (hasNull && nonNullTypes.length === 1) {
      // This handles cases like `type: ['number', 'null']`
      const baseType = jsonSchemaToZod(
        { ...schema, type: nonNullTypes[0] },
        defs,
        visitedRefs,
      );
      return baseType.optional().nullable();
    }

    const types = schema.type.map((type: string) =>
      jsonSchemaToZod({ ...schema, type }, defs, visitedRefs),
    );
    return z.union(types as [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]);
  }

  // Handle enums and literals
  if (schema.enum) {
    if (schema.enum.length === 1) return z.literal(schema.enum[0]);
    const isStringEnum = schema.enum.every((item: any) => typeof item === "string");
    if (isStringEnum) return z.enum(schema.enum as [string, ...string[]]);
    return z.union(schema.enum.map((item: any) => z.literal(item)));
  }
  if (schema.const) return z.literal(schema.const);

  switch (schema.type) {
    case "string": {
      let zodString = z.string();
      if (schema.minLength !== undefined) zodString = zodString.min(schema.minLength);
      if (schema.maxLength !== undefined) zodString = zodString.max(schema.maxLength);
      if (schema.pattern) zodString = zodString.regex(new RegExp(schema.pattern));
      if (schema.format === "email") zodString = zodString.email();
      if (schema.format === "uuid") zodString = zodString.uuid();
      if (schema.format === "uri" || schema.format === "url") zodString = zodString.url();
      if (schema.format === "date-time") zodString = zodString.datetime();
      return zodString;
    }
    case "number":
    case "integer": {
      let zodNum = schema.type === "integer" ? z.number().int() : z.number();
      if (schema.minimum !== undefined) zodNum = zodNum.gte(schema.minimum);
      if (schema.exclusiveMinimum !== undefined) zodNum = zodNum.gt(schema.exclusiveMinimum);
      if (schema.maximum !== undefined) zodNum = zodNum.lte(schema.maximum);
      if (schema.exclusiveMaximum !== undefined) zodNum = zodNum.lt(schema.exclusiveMaximum);
      if (schema.multipleOf !== undefined) zodNum = zodNum.multipleOf(schema.multipleOf);
      return zodNum;
    }
    case "boolean":
      return z.boolean();
    case "null":
      return z.null();
    case "array": {
      let itemSchema: ZodTypeAny = z.any();
      if (schema.items) {
        itemSchema = jsonSchemaToZod(schema.items, defs, visitedRefs);
      }
      let zodArray = z.array(itemSchema);
      if (schema.minItems !== undefined) zodArray = zodArray.min(schema.minItems);
      if (schema.maxItems !== undefined) zodArray = zodArray.max(schema.maxItems);
      return zodArray;
    }
    case "object": {
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
      } else if (typeof schema.additionalProperties === "object") {
        zodObject = z.object(shape).catchall(
          jsonSchemaToZod(schema.additionalProperties, defs, visitedRefs),
        );
      }
      return zodObject;
    }
  }

  if (schema.properties) return jsonSchemaToZod({ ...schema, type: "object" }, defs, visitedRefs);

  return z.any();
}

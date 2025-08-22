import express from "express";
import type { Request, Response } from "express";
import { CodeSandbox } from "@codesandbox/sdk";

// ---------------------------------------------------------------------------
// Env & SDK initialisation
// ---------------------------------------------------------------------------
const templateId = process.env.CODESANDBOX_TEMPLATE_ID;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function ensureSandbox(userId: string) {
  // Look for an existing sandbox with title === userId, otherwise clone template
  const sdk = new CodeSandbox(process.env.CODESANDBOX_API_TOKEN);
  const list = await sdk.sandboxes.list();
  let sandboxId = list.sandboxes.find((s: any) => s.title === userId)?.id;
  if (!sandboxId) {
    const sandbox = await sdk.sandboxes.create({ title: userId, id: templateId });
    sandboxId = sandbox.id;
  }
  return sandboxId;
}

function buildFilePath(project: string, filename?: string) {
  // Always place files under their project directory inside the sandbox.
  return filename ? `${project}/${filename}` : project;
}

function isBinaryUpload(req: Request) {
  return !!(req.is("application/octet-stream") ||
    req.is("application/vnd.debian.binary-package") ||
    req.is("application/x-debian-package"));
}

function pickDebContentType(filename?: string) {
  if (filename && filename.endsWith(".deb")) {
    return "application/vnd.debian.binary-package";
  }
  return "application/octet-stream";
}

export const codesandboxRouter = express.Router();

// Body parsers local to this router
codesandboxRouter.use(express.json({ limit: "200mb" }));
codesandboxRouter.use(express.raw({
  type: ["application/octet-stream", "application/vnd.debian.binary-package", "application/x-debian-package"],
  limit: "500mb",
}));

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// Create or overwrite a file
codesandboxRouter.post("/file", async (req: Request, res: Response) => {
  try {
    const binary = isBinaryUpload(req);

    // Support both query params (for binary uploads) and JSON body (for text/base64)
    const userId = (binary ? req.query.user_id : (req.body?.user_id)) as string | undefined;
    const project = (binary ? req.query.project : (req.body?.project)) as string | undefined;
    const filename = (binary ? req.query.filename : (req.body?.filename)) as string | undefined;

    if (!userId || !project || !filename) {
      return res.status(400).json({ detail: "Missing user_id, project or filename" });
    }

    let bytesToWrite: Uint8Array;

    if (binary) {
      if (!Buffer.isBuffer(req.body)) {
        return res.status(400).json({ detail: "Expected binary body for this content-type" });
      }
      bytesToWrite = new Uint8Array(req.body);
    } else {
      const { content, content_base64, encoding } = req.body || {};

      let contentToWrite = content as any;

      // Auto-fill .env when empty string provided
      if (filename === ".env" && content === "") {
        contentToWrite = `UNIFY_KEY=${req.get("apiKey")}\nUNIFY_PROJECT=${project}`;
      }

      if (typeof content_base64 === "string" || encoding === "base64") {
        const b64 = typeof content_base64 === "string" ? content_base64 : content;
        if (typeof b64 !== "string") {
          return res.status(400).json({ detail: "When encoding is base64, provide content_base64 or content as string" });
        }
        contentToWrite = Buffer.from(b64, "base64");
      }

      if (contentToWrite === undefined) {
        return res.status(400).json({ detail: "Missing content or content_base64" });
      }

      bytesToWrite = Buffer.isBuffer(contentToWrite)
        ? new Uint8Array(contentToWrite)
        : new TextEncoder().encode(contentToWrite);
    }

    const sdk = new CodeSandbox(process.env.CODESANDBOX_API_TOKEN);
    const sandboxId = await ensureSandbox(userId);
    const sandbox = await (await sdk.sandboxes.resume(sandboxId)).connect();

    const filePath = buildFilePath(project, filename);
    await sandbox.fs.writeFile(filePath, bytesToWrite);

    return res.json({ detail: "File written", file_path: filePath });
  } catch (err: any) {
    console.error("[file] POST error", err);
    return res.status(500).json({ detail: "Failed to write file" });
  }
});

// Delete a file or directory
codesandboxRouter.delete("/file", async (req: Request, res: Response) => {
  try {
    const { user_id: userId, project, filename, isDirectory = false } = (req.body && Object.keys(req.body).length)
      ? req.body
      : { user_id: req.query.user_id, project: req.query.project, filename: req.query.filename, isDirectory: req.query.isDirectory === "true" } as any;

    if (!userId || !project || (!isDirectory && !filename)) {
      return res.status(400).json({ detail: "Missing user_id, project or filename" });
    }

    const sdk = new CodeSandbox(process.env.CODESANDBOX_API_TOKEN);
    const sandboxId = await ensureSandbox(userId as string);
    const sandbox = await (await sdk.sandboxes.resume(sandboxId)).connect();

    const filePath = buildFilePath(project as string, filename as string | undefined);

    if (isDirectory) {
      const cmd = sandbox.commands.run(`rm -rf "${filePath}"`);
      await cmd;
    } else {
      await sandbox.fs.remove(filePath);
    }

    return res.json({ detail: "File or directory deleted", file_path: filePath });
  } catch (err: any) {
    console.error("[file] DELETE error", err);
    return res.status(500).json({ detail: "Failed to delete file" });
  }
});

// Rename a file
codesandboxRouter.put("/file", async (req: Request, res: Response) => {
  try {
    const { user_id: userId, project, old_filename, new_filename } = req.body || {};

    if (!userId || !project || !old_filename || !new_filename) {
      return res.status(400).json({ detail: "Missing user_id, project, old_filename or new_filename" });
    }

    const sdk = new CodeSandbox(process.env.CODESANDBOX_API_TOKEN);
    const sandboxId = await ensureSandbox(userId);
    const sandbox = await (await sdk.sandboxes.resume(sandboxId)).connect();

    const oldPath = buildFilePath(project, old_filename);
    const newPath = buildFilePath(project, new_filename);

    const cmd = sandbox.commands.run(`mv "${oldPath}" "${newPath}"`);
    await cmd;

    return res.json({ detail: "File renamed", old_path: oldPath, new_path: newPath });
  } catch (err: any) {
    console.error("[file] PUT error", err);
    return res.status(500).json({ detail: "Failed to rename file" });
  }
});

// Retrieve file content or list project directory
codesandboxRouter.get("/file", async (req: Request, res: Response) => {
  try {
    const userId = req.query.user_id as string | undefined;
    const project = req.query.project as string | undefined;
    const filename = (req.query.filename as string | undefined) ?? undefined;
    const isDirectory = req.query.isDirectory === "true";
    const as = req.query.as as string | undefined; // "base64" | "binary"
    const download = req.query.download === "true";

    if (!userId || !project) {
      return res.status(400).json({ detail: "Missing user_id or project" });
    }

    const sdk = new CodeSandbox(process.env.CODESANDBOX_API_TOKEN);
    const sandboxId = await ensureSandbox(userId);
    const sandbox = await (await sdk.sandboxes.resume(sandboxId)).connect();

    if (isDirectory) {
      const readDirRecursive = async (currentPath: string, prefix: string, visited: Set<string>): Promise<any[]> => {
        if (visited.has(currentPath)) return [];
        visited.add(currentPath);

        const entries: any[] = await sandbox.fs.readdir(currentPath);
        const results: any[] = [];

        for (const entry of entries) {
          const fullName = prefix ? `${prefix}${entry.name}` : entry.name;
          results.push({ ...entry, name: fullName });
          if (entry.type === "directory" && !entry.isSymlink) {
            const childPath = `${currentPath}/${entry.name}`;
            const childPrefix = `${fullName}/`;
            const childEntries = await readDirRecursive(childPath, childPrefix, visited);
            results.push(...childEntries);
          }
        }
        return results;
      };

      const dirPath = buildFilePath(project);
      const files = await readDirRecursive(dirPath, "", new Set());
      return res.json({ files });
    }

    if (!filename) {
      return res.status(400).json({ detail: "Missing filename for file read" });
    }

    const filePath = buildFilePath(project, filename);
    const data = await sandbox.fs.readFile(filePath);
    const buffer = Buffer.from(data);

    const isDeb = filename.endsWith(".deb");

    if (as === "base64") {
      return res.json({ content_base64: buffer.toString("base64") });
    }

    if (download || as === "binary" || isDeb) {
      const contentType = pickDebContentType(filename);
      res.set("Content-Type", contentType);
      res.send(buffer);
      return;
    }

    const decoded = buffer.toString("utf-8");
    return res.json({ content: decoded });
  } catch (err: any) {
    console.error("[file] GET error", err);
    return res.status(500).json({ detail: "Failed to read from filesystem" });
  }
});

export default codesandboxRouter;

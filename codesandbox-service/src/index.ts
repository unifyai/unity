import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
dotenv.config();
import { CodeSandbox } from '@codesandbox/sdk';
import https from 'https';
import http from 'http';

// Env vars per docs + provided reference implementation (standardized names)
const CSB_TOKEN = process.env.CODESANDBOX_API_TOKEN || '';
const CSB_TEMPLATE_ID = process.env.CODESANDBOX_TEMPLATE_ID || '';
const PORT = process.env.CODESANDBOX_SERVICE_PORT || process.env.PORT || 3100;

// Minimal bearer check mirroring agent-service auth style
function verifyApiKeyWithUnify(apiKey: string, assistant_email: string): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const base = process.env.ORCHESTRA_URL || '';
      if (!base) return resolve(false);
      const url = new URL(`${base}/assistant?email=${assistant_email}`);
      const options = {
        method: 'GET',
        hostname: url.hostname,
        port: (url.port || (url.protocol === 'https:' ? '443' : '80')),
        path: url.pathname + url.search,
        headers: { Authorization: `Bearer ${apiKey}` },
      } as any;
      const lib = url.protocol === 'https:' ? https : http;
      const req = lib.request(options, (res) => {
        const code = res.statusCode || 0;
        let body = '';
        res.on('data', (c) => { body += c; });
        res.on('end', () => {
          if (!(code >= 200 && code < 300)) return resolve(false);
          if (!body || body.trim().length === 0) return resolve(false);
          try {
            if (assistant_email.includes('agent') || assistant_email.includes('assistant')) return resolve(true);
            const json = JSON.parse(body);
            if (Array.isArray(json)) return resolve(json.length > 0);
            if (json && typeof json === 'object') {
              if (Array.isArray((json as any).info)) return resolve((json as any).info.length > 0);
              return resolve(Object.keys(json).length > 0);
            }
            if (typeof json === 'string') return resolve(json.trim().length > 0);
            return resolve(!!json);
          } catch {
            return resolve(body.trim().length > 0);
          }
        });
      });
      req.on('error', () => resolve(false));
      req.end();
    } catch {
      resolve(false);
    }
  });
}

async function auth(req: Request, res: Response, next: Function) {
  const hdr = req.header('authorization') || '';
  const match = hdr.match(/^Bearer\s+(.+)$/i);
  if (!match) return res.status(401).json({ error: 'unauthorized', message: 'Missing or invalid API key' });
  const parts = match[1].split(' ');
  const apiKey = parts[0] || '';
  const assistantEmail = parts[1] || '';
  const ok = await verifyApiKeyWithUnify(apiKey, assistantEmail);
  if (!ok) return res.status(401).json({ error: 'unauthorized', message: 'API key verification failed' });
  next();
}

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(auth);

// In-memory open sandboxes keyed by sandboxId
type OpenEntry = { sandbox: any };
const openSandboxes = new Map<string, OpenEntry>();

async function ensureSandbox(userId: string): Promise<string> {
  const sdk = new CodeSandbox(CSB_TOKEN);
  const list = await sdk.sandbox.list();
  let sandboxId = list.find((s: any) => s.title === userId)?.id;
  if (!sandboxId) {
    const sandbox = await sdk.sandbox.create({ title: userId, template: CSB_TEMPLATE_ID || undefined });
    sandboxId = sandbox.id;
  }
  return sandboxId;
}

async function openSandboxById(sandboxId: string): Promise<any> {
  if (openSandboxes.has(sandboxId)) {
    return openSandboxes.get(sandboxId)!.sandbox;
  }
  const sdk = new CodeSandbox(CSB_TOKEN);
  const sandbox = await sdk.sandbox.open(sandboxId);
  openSandboxes.set(sandboxId, { sandbox });
  return sandbox;
}

// Lifecycle endpoints (optional)
app.post('/sandboxes/create', async (_req: Request, res: Response) => {
  try {
    const sdk = new CodeSandbox(CSB_TOKEN);
    const created = await sdk.sandbox.create({ template: CSB_TEMPLATE_ID || undefined });
    res.json({ sandboxId: created.id });
  } catch (e: any) {
    res.status(500).json({ error: 'create_failed', message: String(e?.message || e) });
  }
});

app.post('/sandboxes/:id/connect', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    await openSandboxById(id);
    res.json({ sandboxId: id });
  } catch (e: any) {
    res.status(500).json({ error: 'connect_failed', message: String(e?.message || e) });
  }
});

// FS endpoints per https://codesandbox.io/docs/sdk/filesystem
app.get('/fs/:id/readdir', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const dir = (req.query.dir as string) || '/';
    const sandbox = await openSandboxById(id);
    const entries = await sandbox.fs.readdir(dir);
    res.json({ items: entries });
  } catch (e: any) {
    res.status(500).json({ error: 'readdir_failed', message: String(e?.message || e) });
  }
});

app.get('/fs/:id/readFile', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const path = (req.query.path as string) || '/README.md';
    const sandbox = await openSandboxById(id);
    const data = await sandbox.fs.readFile(path);
    const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
    res.setHeader('Content-Type', 'application/octet-stream');
    res.send(buf);
  } catch (e: any) {
    res.status(500).json({ error: 'read_failed', message: String(e?.message || e) });
  }
});

app.post('/fs/:id/writeFile', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const { path, data, encoding } = req.body || {};
    if (!path) return res.status(400).json({ error: 'bad_request', message: 'path required' });
    const sandbox = await openSandboxById(id);
    const buf = typeof data === 'string' ? Buffer.from(data, encoding || 'utf8') : Buffer.from(data || '');
    await sandbox.fs.writeFile(path, buf);
    res.json({ status: 'ok' });
  } catch (e: any) {
    res.status(500).json({ error: 'write_failed', message: String(e?.message || e) });
  }
});

app.post('/fs/:id/rename', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const { oldPath, newPath } = req.body || {};
    if (!oldPath || !newPath) return res.status(400).json({ error: 'bad_request', message: 'oldPath and newPath required' });
    const sandbox = await openSandboxById(id);
    // Use shell mv to match reference implementation semantics
    const cmd = sandbox.shells.run(`mv "${oldPath}" "${newPath}"`);
    await cmd;
    res.json({ status: 'ok' });
  } catch (e: any) {
    res.status(500).json({ error: 'rename_failed', message: String(e?.message || e) });
  }
});

app.post('/fs/:id/move', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const { oldPath, newParentPath } = req.body || {};
    if (!oldPath || !newParentPath) return res.status(400).json({ error: 'bad_request', message: 'oldPath and newParentPath required' });
    const name = String(oldPath).split('/').pop() || '';
    const sep = newParentPath.endsWith('/') ? '' : '/';
    const newPath = `${newParentPath}${sep}${name}`;
    const sandbox = await openSandboxById(id);
    const cmd = sandbox.shells.run(`mv "${oldPath}" "${newPath}"`);
    await cmd;
    res.json({ status: 'ok', newPath });
  } catch (e: any) {
    res.status(500).json({ error: 'move_failed', message: String(e?.message || e) });
  }
});

app.get('/fs/:id/stat', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const path = (req.query.path as string) || '/';
    const sandbox = await openSandboxById(id);
    if (sandbox.fs.stat) {
      const st = await sandbox.fs.stat(path);
      return res.json({ stat: st });
    }
    // Fallback: infer by trying to read file
    try {
      const data = await sandbox.fs.readFile(path);
      const size = Buffer.isBuffer(data) ? data.length : Buffer.byteLength(String(data));
      return res.json({ stat: { path, size, isDir: false } });
    } catch (_e) {
      // If read fails, assume directory
      return res.json({ stat: { path, isDir: true } });
    }
  } catch (e: any) {
    res.status(500).json({ error: 'stat_failed', message: String(e?.message || e) });
  }
});

app.post('/fs/:id/mkdir', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const { path } = req.body || {};
    if (!path) return res.status(400).json({ error: 'bad_request', message: 'path required' });
    const sandbox = await openSandboxById(id);
    await sandbox.fs.mkdir(path);
    res.json({ status: 'ok' });
  } catch (e: any) {
    res.status(500).json({ error: 'mkdir_failed', message: String(e?.message || e) });
  }
});

app.post('/fs/:id/remove', async (req: Request, res: Response) => {
  try {
    const id = req.params.id;
    const { path, recursive } = req.body || {};
    if (!path) return res.status(400).json({ error: 'bad_request', message: 'path required' });
    const sandbox = await openSandboxById(id);
    if (recursive) {
      const cmd = sandbox.shells.run(`rm -rf "${path}"`);
      await cmd;
    } else {
      await sandbox.fs.remove(path);
    }
    res.json({ status: 'ok' });
  } catch (e: any) {
    res.status(500).json({ error: 'remove_failed', message: String(e?.message || e) });
  }
});

// High-level file routes mirroring reference implementation
function buildFilePath(project: string, filename?: string) {
  return filename ? `${project}/${filename}` : project;
}

app.post('/file', async (req: Request, res: Response) => {
  try {
    const { user_id: userId, project, filename, content } = req.body || {};
    if (!userId || !project || !filename || typeof content !== 'string') {
      return res.status(400).json({ detail: 'Missing user_id, project, filename or content' });
    }
    let contentToWrite: string = content;
    if (filename === '.env' && content === '') {
      const apiKeyHeader = req.header('apiKey') || '';
      const unifyKey = apiKeyHeader || process.env.UNIFY_KEY || '';
      contentToWrite = `UNIFY_KEY=${unifyKey}\nUNIFY_PROJECT=${project}`;
    }
    const sandboxId = await ensureSandbox(String(userId));
    const sandbox = await openSandboxById(sandboxId);
    const filePath = buildFilePath(project, filename);
    const encoded = Buffer.from(contentToWrite, 'utf8');
    await sandbox.fs.writeFile(filePath, encoded);
    return res.json({ detail: 'File written', file_path: filePath });
  } catch (e: any) {
    return res.status(500).json({ detail: 'Failed to write file' });
  }
});

app.delete('/file', async (req: Request, res: Response) => {
  try {
    const { user_id: userId, project, filename, isDirectory = false } = req.body || {};
    if (!userId || !project || (!isDirectory && !filename)) {
      return res.status(400).json({ detail: 'Missing user_id, project or filename' });
    }
    const sandboxId = await ensureSandbox(String(userId));
    const sandbox = await openSandboxById(sandboxId);
    const filePath = buildFilePath(project, filename);
    if (isDirectory) {
      const cmd = sandbox.shells.run(`rm -rf "${filePath}"`);
      await cmd;
    } else {
      await sandbox.fs.remove(filePath);
    }
    return res.json({ detail: 'File or directory deleted', file_path: filePath });
  } catch (e: any) {
    return res.status(500).json({ detail: 'Failed to delete file' });
  }
});

app.put('/file', async (req: Request, res: Response) => {
  try {
    const { user_id: userId, project, old_filename, new_filename } = req.body || {};
    if (!userId || !project || !old_filename || !new_filename) {
      return res.status(400).json({ detail: 'Missing user_id, project, old_filename or new_filename' });
    }
    const sandboxId = await ensureSandbox(String(userId));
    const sandbox = await openSandboxById(sandboxId);
    const oldPath = buildFilePath(project, old_filename);
    const newPath = buildFilePath(project, new_filename);
    const cmd = sandbox.shells.run(`mv "${oldPath}" "${newPath}"`);
    await cmd;
    return res.json({ detail: 'File renamed', old_path: oldPath, new_path: newPath });
  } catch (e: any) {
    return res.status(500).json({ detail: 'Failed to rename file' });
  }
});

app.get('/file', async (req: Request, res: Response) => {
  const params = req.query as Record<string, string>;
  const userId = params['user_id'];
  const project = params['project'];
  const filename = params['filename'];
  const isDirectory = params['isDirectory'] === 'true';
  if (!userId || !project) {
    return res.status(400).json({ detail: 'Missing user_id or project' });
  }
  try {
    const sandboxId = await ensureSandbox(String(userId));
    const sandbox = await openSandboxById(sandboxId);
    if (isDirectory) {
      const readDirRecursive = async (currentPath: string, prefix: string, visited: Set<string>): Promise<any[]> => {
        if (visited.has(currentPath)) return [];
        visited.add(currentPath);
        const entries: any[] = await sandbox.fs.readdir(currentPath);
        const results: any[] = [];
        for (const entry of entries) {
          const fullName = prefix ? `${prefix}${entry.name}` : entry.name;
          results.push({ ...entry, name: fullName });
          if (entry.type === 'directory' && !entry.isSymlink) {
            const childPath = `${currentPath}/${entry.name}`;
            const childPrefix = `${fullName}/`;
            const childEntries = await readDirRecursive(childPath, childPrefix, visited);
            results.push(...childEntries);
          }
        }
        return results;
      };
      const dirPath = buildFilePath(project);
      const files = await readDirRecursive(dirPath, '', new Set());
      return res.json({ files });
    } else {
      if (!filename) return res.status(400).json({ detail: 'Missing filename for file read' });
      const filePath = buildFilePath(project, filename);
      const data = await sandbox.fs.readFile(filePath);
      const decoded = Buffer.isBuffer(data) ? data.toString('utf8') : new TextDecoder().decode(data);
      return res.json({ content: decoded });
    }
  } catch (e: any) {
    return res.status(500).json({ detail: 'Failed to read from filesystem' });
  }
});

app.get('/health', (_req: Request, res: Response) => res.json({ status: 'ok' }));

app.listen(PORT, () => {
  console.log(`🚀 CodeSandbox service listening on http://localhost:${PORT}`);
});

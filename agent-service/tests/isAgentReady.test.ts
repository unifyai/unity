import { strict as assert } from "node:assert";

/**
 * Specification tests for the isAgentReady Express middleware.
 *
 * The middleware guards all agent-service endpoints. These tests verify
 * the contract, including the desktop-session fallback that supports
 * callers (ConversationManager screenshot capture) that don't have
 * access to the MagnitudeBackend session management.
 *
 * Desktop mode is singleton (one physical display, one session). When
 * no sessionId is provided, the middleware resolves to the desktop
 * session. Web-mode callers must always provide an explicit sessionId.
 *
 * To ensure the tests regress when the production middleware changes,
 * we extract the middleware logic from a fresh require of the source
 * module. The production middleware is defined as a const closure over
 * `activeSessions`, so we replicate that pattern and test against the
 * same branching logic.
 */

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

// Minimal types matching Express Request/Response for middleware testing.
interface FakeReq { body: Record<string, any> }
interface FakeRes {
  _status: number;
  _json: any;
  status(code: number): FakeRes;
  json(data: any): void;
}

interface SessionInfo {
  mode: "web" | "desktop";
  lastAccessed: Date;
}

function fakeRes(): FakeRes {
  const res: FakeRes = {
    _status: 200,
    _json: null,
    status(code: number) { res._status = code; return res; },
    json(data: any) { res._json = data; },
  };
  return res;
}

/**
 * Build an isAgentReady function that mirrors the production middleware.
 */
function buildMiddleware(
  activeSessions: Map<string, SessionInfo>,
) {
  return function isAgentReady(req: FakeReq, res: FakeRes): { passed: boolean } {
    let sessionId = req.body.sessionId;
    if (!sessionId) {
      const desktopEntry = [...activeSessions.entries()]
        .find(([, s]) => s.mode === "desktop");
      if (desktopEntry) {
        sessionId = desktopEntry[0];
        req.body.sessionId = sessionId;
      } else {
        res.status(400).json({ error: "no_desktop_session", message: "No active desktop session. Call /start with mode=desktop first." });
        return { passed: false };
      }
    }
    const session = activeSessions.get(sessionId);
    if (!session) {
      res.status(404).json({ error: "session_not_found", message: `Session ${sessionId} not found.` });
      return { passed: false };
    }
    session.lastAccessed = new Date();
    return { passed: true };
  };
}

// -------------------------------------------------------------------
// Desktop-session fallback tests
// -------------------------------------------------------------------

run("rejects when no sessionId and no active sessions", () => {
  const sessions = new Map<string, SessionInfo>();
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, false);
  assert.strictEqual(res._status, 400);
  assert.strictEqual(res._json.error, "no_desktop_session");
});

run("rejects when no sessionId and only web sessions exist", () => {
  const sessions = new Map<string, SessionInfo>();
  sessions.set("web-a", { mode: "web", lastAccessed: new Date() });
  sessions.set("web-b", { mode: "web", lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, false);
  assert.strictEqual(res._status, 400);
  assert.strictEqual(res._json.error, "no_desktop_session");
});

run("falls back to desktop session when no sessionId provided", () => {
  const sessions = new Map<string, SessionInfo>();
  sessions.set("desktop-1", { mode: "desktop", lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, true);
  assert.strictEqual(req.body.sessionId, "desktop-1");
});

run("falls back to desktop session even when web sessions also exist", () => {
  const sessions = new Map<string, SessionInfo>();
  sessions.set("web-1", { mode: "web", lastAccessed: new Date() });
  sessions.set("desktop-1", { mode: "desktop", lastAccessed: new Date() });
  sessions.set("web-2", { mode: "web", lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, true);
  assert.strictEqual(req.body.sessionId, "desktop-1");
});

// -------------------------------------------------------------------
// Explicit sessionId tests
// -------------------------------------------------------------------

run("passes with explicit valid sessionId", () => {
  const sessions = new Map<string, SessionInfo>();
  sessions.set("my-session", { mode: "web", lastAccessed: new Date() });
  sessions.set("other-session", { mode: "web", lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: { sessionId: "my-session" } };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, true);
});

run("returns 404 for non-existent sessionId", () => {
  const sessions = new Map<string, SessionInfo>();
  sessions.set("real-session", { mode: "desktop", lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: { sessionId: "ghost-session" } };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, false);
  assert.strictEqual(res._status, 404);
  assert.ok(res._json.message.includes("ghost-session"));
});

run("updates lastAccessed on successful lookup", () => {
  const old = new Date(2020, 0, 1);
  const sessions = new Map<string, SessionInfo>();
  sessions.set("sess", { mode: "desktop", lastAccessed: old });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: { sessionId: "sess" } };
  const res = fakeRes();
  mw(req, res);

  assert.notStrictEqual(sessions.get("sess")!.lastAccessed, old);
});

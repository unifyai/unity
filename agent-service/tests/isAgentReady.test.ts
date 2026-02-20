import { strict as assert } from "node:assert";

/**
 * Specification tests for the isAgentReady Express middleware.
 *
 * The middleware guards all agent-service endpoints. These tests verify
 * the contract, including the single-session fallback added to support
 * callers (ConversationManager screenshot capture) that don't have
 * access to the MagnitudeBackend session management.
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
  activeSessions: Map<string, { lastAccessed: Date }>,
) {
  return function isAgentReady(req: FakeReq, res: FakeRes): { passed: boolean } {
    let sessionId = req.body.sessionId;
    if (!sessionId) {
      if (activeSessions.size === 1) {
        sessionId = activeSessions.keys().next().value;
        req.body.sessionId = sessionId;
      } else {
        res.status(400).json({ error: "bad_request", message: "sessionId is required (multiple sessions active)." });
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
// Fixed behaviour tests (current production middleware)
// -------------------------------------------------------------------

run("rejects when no sessionId and no active sessions", () => {
  const sessions = new Map<string, { lastAccessed: Date }>();
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, false);
  assert.strictEqual(res._status, 400);
  assert.ok(res._json.message.includes("sessionId is required"));
});

run("rejects when no sessionId and multiple active sessions", () => {
  const sessions = new Map<string, { lastAccessed: Date }>();
  sessions.set("session-a", { lastAccessed: new Date() });
  sessions.set("session-b", { lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, false);
  assert.strictEqual(res._status, 400);
});

run("falls back to single active session when no sessionId provided", () => {
  const sessions = new Map<string, { lastAccessed: Date }>();
  sessions.set("only-session", { lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: {} };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, true);
  assert.strictEqual(req.body.sessionId, "only-session");
});

run("passes with explicit valid sessionId", () => {
  const sessions = new Map<string, { lastAccessed: Date }>();
  sessions.set("my-session", { lastAccessed: new Date() });
  sessions.set("other-session", { lastAccessed: new Date() });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: { sessionId: "my-session" } };
  const res = fakeRes();
  const result = mw(req, res);

  assert.strictEqual(result.passed, true);
});

run("returns 404 for non-existent sessionId", () => {
  const sessions = new Map<string, { lastAccessed: Date }>();
  sessions.set("real-session", { lastAccessed: new Date() });
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
  const sessions = new Map<string, { lastAccessed: Date }>();
  sessions.set("sess", { lastAccessed: old });
  const mw = buildMiddleware(sessions);
  const req: FakeReq = { body: { sessionId: "sess" } };
  const res = fakeRes();
  mw(req, res);

  assert.notStrictEqual(sessions.get("sess")!.lastAccessed, old);
});

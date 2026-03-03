import { strict as assert } from "node:assert";

/**
 * Tests for fresh-session context injection in the /act handler.
 *
 * When a web/web-vm session has no prior act history, the handler injects
 * a system observation telling the LLM that the browser is already open.
 * This lets the LLM no-op (empty action list) when the task is purely
 * "open the browser", rather than inventing a side-effect like opening a
 * new tab.
 *
 * We replicate the injection conditional from the production /act handler
 * and verify it fires for the right session states.
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

interface SessionInfo {
  mode: "web" | "web-vm" | "desktop";
  actHistory: { task: string }[];
}

const FRESH_SESSION_NOTE =
  "This is a freshly created browser session — the browser is already open and loaded. "
  + "If the task is simply asking to open a browser, open a new browser window, or launch a browser, "
  + "this has already been accomplished. Return an empty actions list.";

/**
 * Mirror the production conditional: inject the fresh-session note when
 * actHistory is empty AND mode is not desktop.
 */
function shouldInjectFreshSessionContext(session: SessionInfo): boolean {
  return session.actHistory.length === 0 && session.mode !== "desktop";
}

// -------------------------------------------------------------------
// Cases where the context note SHOULD be injected
// -------------------------------------------------------------------

run("injects for fresh web session (empty actHistory)", () => {
  const session: SessionInfo = { mode: "web", actHistory: [] };
  assert.strictEqual(shouldInjectFreshSessionContext(session), true);
});

run("injects for fresh web-vm session (empty actHistory)", () => {
  const session: SessionInfo = { mode: "web-vm", actHistory: [] };
  assert.strictEqual(shouldInjectFreshSessionContext(session), true);
});

// -------------------------------------------------------------------
// Cases where the context note should NOT be injected
// -------------------------------------------------------------------

run("does NOT inject for desktop session (even if fresh)", () => {
  const session: SessionInfo = { mode: "desktop", actHistory: [] };
  assert.strictEqual(shouldInjectFreshSessionContext(session), false);
});

run("does NOT inject for web session with prior history", () => {
  const session: SessionInfo = {
    mode: "web",
    actHistory: [{ task: "Navigate to google.com" }],
  };
  assert.strictEqual(shouldInjectFreshSessionContext(session), false);
});

run("does NOT inject for web-vm session with prior history", () => {
  const session: SessionInfo = {
    mode: "web-vm",
    actHistory: [{ task: "Open the browser" }],
  };
  assert.strictEqual(shouldInjectFreshSessionContext(session), false);
});

run("does NOT inject for desktop session with prior history", () => {
  const session: SessionInfo = {
    mode: "desktop",
    actHistory: [{ task: "Click the terminal" }],
  };
  assert.strictEqual(shouldInjectFreshSessionContext(session), false);
});

// -------------------------------------------------------------------
// Verify the injected note content is stable
// -------------------------------------------------------------------

run("fresh-session note mentions browser is already open", () => {
  assert.ok(FRESH_SESSION_NOTE.includes("browser is already open"));
});

run("fresh-session note instructs empty actions list", () => {
  assert.ok(FRESH_SESSION_NOTE.includes("empty actions list"));
});

import { strict as assert } from "node:assert";

/**
 * Tests for viewport handling in BrowserProvider._createAndTrackContext.
 *
 * When contextOptions.viewport is null, the browser page follows the window
 * size dynamically (no CDP device-metrics override). When it's a fixed
 * {width, height} object, the CDP override pins the viewport.
 *
 * These tests replicate the resolution logic from browserProvider.ts and
 * verify the branching contract.
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

// ── Replicate the resolution logic from browserProvider._createAndTrackContext ──

interface ContextOptions {
  viewport?: { width: number; height: number } | null;
  deviceScaleFactor?: number;
}

/**
 * Determines whether the CDP device-metrics override should be applied.
 * Mirrors the logic in browserProvider.ts _createAndTrackContext.
 */
function shouldApplyCdpOverride(contextOptions?: ContextOptions): boolean {
  const resolvedViewport = contextOptions?.viewport;
  return !!resolvedViewport;
}

/**
 * Returns the viewport dimensions for the CDP override, or null if dynamic.
 */
function resolveViewport(contextOptions?: ContextOptions): { width: number; height: number } | null {
  const resolvedViewport = contextOptions?.viewport;
  return resolvedViewport ?? null;
}

// ── Replicate the DEFAULT_BROWSER_CONTEXT_OPTIONS merge from newContext ──

const DEFAULT_BROWSER_CONTEXT_OPTIONS: ContextOptions = {
  viewport: { width: 1024, height: 768 },
};

function mergeContextOptions(callerOptions?: ContextOptions): ContextOptions {
  return {
    ...DEFAULT_BROWSER_CONTEXT_OPTIONS,
    ...(callerOptions ?? {}),
  };
}

// -------------------------------------------------------------------
// Fixed viewport (default behavior)
// -------------------------------------------------------------------

run("applies CDP override with default viewport when no contextOptions provided", () => {
  const merged = mergeContextOptions(undefined);
  assert.strictEqual(shouldApplyCdpOverride(merged), true);
  const vp = resolveViewport(merged);
  assert.deepStrictEqual(vp, { width: 1024, height: 768 });
});

run("applies CDP override with default viewport when empty contextOptions provided", () => {
  const merged = mergeContextOptions({});
  assert.strictEqual(shouldApplyCdpOverride(merged), true);
  const vp = resolveViewport(merged);
  assert.deepStrictEqual(vp, { width: 1024, height: 768 });
});

run("applies CDP override with custom fixed viewport", () => {
  const merged = mergeContextOptions({ viewport: { width: 1920, height: 1080 } });
  assert.strictEqual(shouldApplyCdpOverride(merged), true);
  const vp = resolveViewport(merged);
  assert.deepStrictEqual(vp, { width: 1920, height: 1080 });
});

// -------------------------------------------------------------------
// Dynamic viewport (viewport: null)
// -------------------------------------------------------------------

run("skips CDP override when viewport is explicitly null", () => {
  const merged = mergeContextOptions({ viewport: null });
  assert.strictEqual(shouldApplyCdpOverride(merged), false);
  assert.strictEqual(resolveViewport(merged), null);
});

run("null viewport overrides the default fixed viewport in merge", () => {
  const merged = mergeContextOptions({ viewport: null });
  assert.strictEqual(merged.viewport, null);
});

// -------------------------------------------------------------------
// Web-VM mode configuration (agent-service startBrowserOnVm)
// -------------------------------------------------------------------

run("web-vm browser options produce dynamic viewport after merge", () => {
  // This mirrors what startBrowserOnVm passes:
  //   browser: { ..., contextOptions: { viewport: null } }
  const webVmContextOptions: ContextOptions = { viewport: null };
  const merged = mergeContextOptions(webVmContextOptions);
  assert.strictEqual(shouldApplyCdpOverride(merged), false);
  assert.strictEqual(merged.viewport, null);
});

run("headless web mode retains fixed viewport (no contextOptions override)", () => {
  // startBrowser does not pass contextOptions, so defaults apply
  const merged = mergeContextOptions(undefined);
  assert.strictEqual(shouldApplyCdpOverride(merged), true);
  assert.deepStrictEqual(merged.viewport, { width: 1024, height: 768 });
});

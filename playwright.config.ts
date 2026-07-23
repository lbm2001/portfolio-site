import { defineConfig, devices } from "@playwright/test";

// E2E runs against the REAL Cloudflare worker (workerd via `wrangler dev`), not
// `next start` — passing here means the OpenNext bundle builds and serves, which
// is the "deploys cleanly" guarantee this suite exists for.
//
//   npm run e2e:build   — copy VLA assets + `next build` + OpenNext bundle
//   npm run e2e         — this config (starts the preview server itself)
//   npm run e2e:full    — adds the slow train-to-convergence hero specs
//   npm run smoke:live  — same specs against https://lukasmueller.dev
//
// PLAYWRIGHT_BASE_URL switches the target to an already-running/deployed site
// and disables the local webServer.

const liveBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
const PORT = 8787; // wrangler dev default

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  // Hero training is CPU-bound (tfjs on SwiftShader); parallel workers starve
  // each other and turn training waits into flakes — always on CI, and locally
  // whenever the slow convergence specs are enabled (two viewports training
  // at once never finish inside their timeout).
  workers: process.env.CI || process.env.VLA_FULL ? 1 : undefined,
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never" }]]
    : [["list"]],
  timeout: 120_000,
  use: {
    baseURL: liveBaseUrl ?? `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    // Playwright's default is 0 — an action on an element that never appears
    // blocks until the whole test times out, which reads as a mysterious hang
    // rather than a failed assertion. Bound it.
    actionTimeout: 15_000,
    launchOptions: {
      // tfjs-webgl needs a GL context in headless Chromium; SwiftShader is a
      // software rasterizer that works everywhere (slower than a real GPU).
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    },
  },
  projects: [
    // ≥1100px: the desktop pipeline ring. Runs the full suite.
    {
      name: "desktop",
      use: { viewport: { width: 1440, height: 900 } },
    },
    // <1100px: pipeline is hidden behind the "Mini VLA Demo" CTA and trains a
    // smaller task. Only the hero specs care about this tier.
    {
      name: "mobile",
      testMatch: /hero.*\.spec\.ts/,
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
    // Real WebKit, not just Chromium/SwiftShader: the CPU-backend replay
    // fallback (Hero.tsx's replayFallback: true) exists specifically for
    // iOS/iPadOS's WebGL context cap, but until now nothing in this suite
    // ever ran on the engine that condition actually occurs on — only a
    // simulated abort on Chromium (tests/e2e/hero-error.spec.ts). The device
    // preset below drives real WebKit with an iOS user agent/viewport/touch
    // profile. The chromium-only `--use-angle=swiftshader` launch args above
    // don't apply to WebKit, hence the explicit empty override.
    {
      name: "webkit-mobile",
      testMatch: /hero.*\.spec\.ts/,
      use: { ...devices["iPhone 13"], launchOptions: {} },
    },
  ],
  webServer: liveBaseUrl
    ? undefined
    : {
        // Serves the worker built by `npm run e2e:build`; fails fast if
        // .open-next/ is missing.
        command: "npx opennextjs-cloudflare preview",
        url: `http://localhost:${PORT}`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});

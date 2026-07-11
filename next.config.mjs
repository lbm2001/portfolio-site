import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

const BUILD_ID = randomUUID();
writeFileSync(
  path.join(import.meta.dirname, "public/build-id.json"),
  JSON.stringify({ id: BUILD_ID }),
);

/** @type {(phase: string) => import('next').NextConfig} */
const config = (phase) => {
  /** @type {import('next').NextConfig} */
  const nextConfig = {
    env: { NEXT_PUBLIC_BUILD_ID: BUILD_ID },
    reactStrictMode: true,
    // mini-vla ships TS source; Next must transpile it (and bundle its module
    // Web Worker) as if it were first-party code.
    transpilePackages: ["mini-vla"],

    // Next serves prerendered pages with `s-maxage=31536000`, so a returning
    // visitor can be handed year-old HTML that still names the PREVIOUS
    // deploy's hashed chunks. Those files no longer exist (each Workers
    // deploy ships a fresh asset manifest), and the hero only reaches for its
    // trainer/tfjs chunks lazily, on "Start Training" — so the page looks
    // fine and the button silently does nothing. Revalidate the HTML on every
    // request instead; the ETag keeps that a cheap 304.
    //
    // Scoped to page routes ON PURPOSE. A catch-all would also match
    // /_next/static/*, whose content-hashed URLs must keep the immutable,
    // one-year caching that makes repeat visits fast (see public/_headers).
    async headers() {
      const pageRoutes = [
        "/",
        "/about",
        "/resume",
        "/projects",
        "/projects/:slug",
        "/blog",
        "/blog/:slug",
      ];
      return pageRoutes.map((source) => ({
        source,
        headers: [
          { key: "Cache-Control", value: "public, max-age=0, must-revalidate" },
        ],
      }));
    },
  };

  // mini-vla is pinned to a `github:` git-ref (a normal node_modules copy inside
  // the project root), which resolves with no help. This dev-only root widening
  // is a safety net for LOCAL mini-vla development: if you temporarily switch the
  // dep to `file:../mini-vla`, that symlink's realpath is a SIBLING dir outside
  // the lockfile-inferred root and `next dev` can't resolve it without this.
  // Harmless with the git-ref. Must stay DEV ONLY: the OpenNext build forces its
  // own trace root via NEXT_PRIVATE_OUTPUT_TRACE_ROOT, and a turbopack.root that
  // disagrees with it silently breaks module resolution.
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    nextConfig.turbopack = { root: path.resolve(import.meta.dirname, "..") };
    // Testing the hero on a real phone means loading `next dev` over the LAN IP
    // rather than localhost. Next 16 refuses to serve /_next dev resources to a
    // cross-origin host by default, so the page arrives as HTML and never
    // hydrates: every button is dead and no worker is ever constructed.
    // Override the machine-specific default per-shell with NEXT_DEV_ORIGIN.
    nextConfig.allowedDevOrigins = [process.env.NEXT_DEV_ORIGIN ?? "192.168.178.103"];
  }

  return nextConfig;
};

export default config;

import('@opennextjs/cloudflare').then(m => m.initOpenNextCloudflareForDev());

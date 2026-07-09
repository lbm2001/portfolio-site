import path from "node:path";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/** @type {(phase: string) => import('next').NextConfig} */
export default (phase) => {
  /** @type {import('next').NextConfig} */
  const nextConfig = {
    reactStrictMode: true,
    // mini-vla ships TS source; Next must transpile it (and bundle its module
    // Web Worker) as if it were first-party code.
    transpilePackages: ["mini-vla"],
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
    nextConfig.allowedDevOrigins = ["192.168.178.103"];
  }

  return nextConfig;
};

import('@opennextjs/cloudflare').then(m => m.initOpenNextCloudflareForDev());

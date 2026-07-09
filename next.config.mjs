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

  // Local dev installs mini-vla as `file:../mini-vla` — a symlink whose realpath
  // is a SIBLING directory, outside the root Turbopack infers from this repo's
  // lockfile, so `next dev` can't resolve it. Widen the root to the parent that
  // holds both repos. DEV ONLY: production installs mini-vla as a normal
  // node_modules copy (a `github:` git-ref dep) already inside the root, and the
  // OpenNext build forces its own trace root via NEXT_PRIVATE_OUTPUT_TRACE_ROOT
  // — declaring turbopack.root there just conflicts with it and breaks
  // resolution.
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    nextConfig.turbopack = { root: path.resolve(import.meta.dirname, "..") };
  }

  return nextConfig;
};

import('@opennextjs/cloudflare').then(m => m.initOpenNextCloudflareForDev());

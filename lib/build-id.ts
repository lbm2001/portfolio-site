// Single source of truth for the stale-tab build-id file's path and shape.
// components/Hero.tsx (reader) imports this directly. next.config.mjs
// (writer) and public/_headers (cache rule) each keep their own literal copy
// instead — next.config.mjs can't import this module (Next's config loader
// can't type-strip .ts; see its own comment), and public/_headers is a static
// file the build can't import into at all — so those two are kept in sync
// with this module only by convention, guarded by tests/unit/build-id.test.ts.
export const BUILD_ID_PATH = "/build-id.json";

export interface BuildIdPayload {
  id: string;
}

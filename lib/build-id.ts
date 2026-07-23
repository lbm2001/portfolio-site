// Single source of truth for the stale-tab build-id file's path and shape,
// shared by next.config.mjs (writer), components/Hero.tsx (reader), and
// public/_headers (cache rule) — previously duplicated across all three with
// nothing tying them together, so a rename in one place would have silently
// broken the other two. public/_headers is a static file the build can't
// import into, so its path is kept in sync by convention; the two TS/JS
// consumers below share this module instead.
export const BUILD_ID_PATH = "/build-id.json";

export interface BuildIdPayload {
  id: string;
}

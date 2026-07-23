// Single source of truth for a project source's slug — gen-projects-data.mjs's
// main() previously re-derived this independently (for log/catch messages)
// from buildOne()'s own copy (for actual use); an edit to one without the
// other could silently make logged output and actual output disagree
// (review round 1, finding #22). Exported to its own module, rather than
// just a local function, so tests/unit/content-data.test.ts can import the
// real formula instead of reimplementing it — importing gen-projects-data.mjs
// directly isn't an option, since it runs main() (real fetches) as a
// side effect of being loaded.
export function deriveSlug(source) {
  return source.slug ?? source.repo?.split("/")[1];
}

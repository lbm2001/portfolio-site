import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BUILD_ID_PATH } from "../../lib/build-id";

// components/Hero.tsx imports BUILD_ID_PATH directly, so it can't drift.
// next.config.mjs and public/_headers can't (see next.config.mjs's comment
// for why) — their copies are kept in sync only by convention (review round
// 1, finding #8). Pin both here: a rename of BUILD_ID_PATH without updating
// either file fails loud instead of silently breaking the stale-tab reload
// guard or dropping its cache-control rule.
describe("BUILD_ID_PATH", () => {
  const root = join(import.meta.dirname, "../..");

  it("matches the cache rule in public/_headers", () => {
    const headers = readFileSync(join(root, "public/_headers"), "utf8");
    expect(headers).toContain(`\n${BUILD_ID_PATH}\n`);
  });

  it("matches the literal next.config.mjs writes to", () => {
    const config = readFileSync(join(root, "next.config.mjs"), "utf8");
    expect(config).toContain(`BUILD_ID_PATH = "${BUILD_ID_PATH}"`);
  });

  // The path pinning above doesn't confirm the JSON shape next.config.mjs
  // writes still matches BuildIdPayload ({ id: string }) — components/Hero.tsx
  // destructures `{ id }` from the fetched body (Hero.tsx:1699) and reloads
  // whenever it disagrees with buildId, so a silently-renamed key would read
  // `undefined` there and force a reload on every visibilitychange.
  it("writes the payload under the key components/Hero.tsx reads (id)", () => {
    const config = readFileSync(join(root, "next.config.mjs"), "utf8");
    expect(config).toContain("JSON.stringify({ id: BUILD_ID })");
  });
});

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { nav } from "../../lib/content";

// The site's top-level page routes are hand-listed independently in three
// places, with nothing tying them together: next.config.mjs's `pageRoutes`
// (drives the must-revalidate Cache-Control rule — see its own comment for
// why that matters for the stale-chunk reload path), lib/content.ts's `nav`,
// and tests/e2e/helpers.ts's sitePageRoutes() (drives routes.spec.ts /
// caching.spec.ts's coverage — those two used to each build this list
// independently and had already drifted apart once before being unified into
// sitePageRoutes(), per that function's own comment). A route added to `nav`
// or sitePageRoutes() but forgotten in next.config.mjs silently loses the
// revalidate rule — and caching.spec.ts wouldn't catch it either, since it
// sweeps the same sitePageRoutes() list, not next.config.mjs's.
//
// Can't import next.config.mjs directly (Next's config loader is the only
// thing that can load it — see its own comment) or tests/e2e/helpers.ts
// (imports @playwright/test, a different test runtime) — read both as text,
// same approach as tests/unit/build-id.test.ts uses for next.config.mjs.
describe("top-level page routes stay in sync across next.config.mjs / nav / sitePageRoutes", () => {
  const root = join(import.meta.dirname, "../..");
  const configSrc = readFileSync(join(root, "next.config.mjs"), "utf8");
  const helpersSrc = readFileSync(join(root, "tests/e2e/helpers.ts"), "utf8");

  it("every nav link's href is one of next.config.mjs's page routes", () => {
    for (const { href } of nav) {
      expect(configSrc, `next.config.mjs's pageRoutes is missing "${href}"`).toContain(
        `"${href}"`,
      );
    }
  });

  it("next.config.mjs covers the home route nav doesn't list", () => {
    expect(configSrc).toContain('"/"');
  });

  it("sitePageRoutes()'s static routes are all in next.config.mjs's page routes", () => {
    // The five static entries before sitePageRoutes()'s dynamic slug spreads.
    const staticRoutes = ["/", "/about", "/projects", "/blog", "/resume"];
    for (const route of staticRoutes) {
      expect(
        helpersSrc,
        `tests/e2e/helpers.ts's sitePageRoutes() no longer lists "${route}" — update this test's staticRoutes too`,
      ).toContain(`"${route}"`);
      expect(
        configSrc,
        `next.config.mjs's pageRoutes is missing "${route}", which sitePageRoutes() covers`,
      ).toContain(`"${route}"`);
    }
  });
});

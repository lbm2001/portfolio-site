import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

// Regression guard for the "Start Training silently does nothing" outage.
//
// Next serves prerendered pages with `s-maxage=31536000`, so a returning
// visitor could be handed year-old HTML naming a PREVIOUS deploy's hashed
// chunks. Those files are gone after a deploy, and the hero only fetches its
// trainer/tfjs chunks lazily on "Start Training" — so the page looked fine and
// the button died. The fix is the `headers()` block in next.config.mjs, which
// must hold BOTH ways:
//
//   page routes      -> must-revalidate (always re-check the HTML; ETag → 304)
//   /_next/static/*  -> immutable       (content-hashed; long cache = fast repeats)
//
// Widening that rule to a catch-all would kill repeat-visit performance;
// dropping it brings the outage back. Neither shows up in any other test.

const root = join(import.meta.dirname, "../..");
const projects: { slug: string }[] = JSON.parse(
  readFileSync(join(root, "lib/projects-data.json"), "utf8"),
);
const posts: { slug: string }[] = JSON.parse(
  readFileSync(join(root, "lib/posts-data.json"), "utf8"),
);

const pageRoutes = [
  "/",
  "/about",
  "/projects",
  "/blog",
  "/resume",
  ...projects.map((p) => `/projects/${p.slug}`),
  ...posts.map((p) => `/blog/${p.slug}`),
];

test.describe("page routes revalidate", () => {
  for (const route of pageRoutes) {
    test(`${route} is must-revalidate, not long-lived`, async ({ request }) => {
      const res = await request.get(route);
      expect(res.status()).toBe(200);

      const cc = res.headers()["cache-control"] ?? "";
      expect(cc).toContain("must-revalidate");
      expect(cc).toMatch(/max-age=0\b/);
      // The immutable/long-lived caching that caused the stale-chunk outage.
      expect(cc).not.toContain("immutable");
      expect(cc).not.toMatch(/s-maxage=\d{3,}/);
    });
  }
});

test("page HTML revalidates cheaply via ETag (304)", async ({ request }) => {
  // must-revalidate is only affordable because the re-check is a 304.
  const res = await request.get("/");
  const etag = res.headers()["etag"];
  expect(etag, "page responses must carry an ETag").toBeTruthy();

  const revalidated = await request.get("/", { headers: { "If-None-Match": etag } });
  expect(revalidated.status()).toBe(304);
});

test("hashed static chunks stay immutable", async ({ page, request }) => {
  // Scoping guard: a catch-all `headers()` source would sweep these up too and
  // make every repeat visit re-download the JS.
  await page.goto("/");
  const chunk = await page
    .locator('script[src^="/_next/static/"]')
    .first()
    .getAttribute("src");
  expect(chunk, "homepage must load a hashed static chunk").toBeTruthy();

  const res = await request.get(chunk!);
  expect(res.status()).toBe(200);

  const cc = res.headers()["cache-control"] ?? "";
  expect(cc).toContain("immutable");
  expect(cc).toMatch(/max-age=31536000/);
  expect(cc).not.toContain("must-revalidate");
});

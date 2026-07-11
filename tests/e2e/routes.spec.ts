import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { expect, test } from "@playwright/test";
import { VLA_ASSET_BASE } from "../../lib/vla-assets";
import { collectPageErrors } from "./helpers";

// Every page the site serves, enumerated from the same committed data the
// build renders from — a slug in the JSON without a working page is exactly
// the kind of regression this sweep exists to catch.

const root = join(import.meta.dirname, "../..");
const projects: { slug: string }[] = JSON.parse(
  readFileSync(join(root, "lib/projects-data.json"), "utf8"),
);
const posts: { slug: string }[] = JSON.parse(
  readFileSync(join(root, "lib/posts-data.json"), "utf8"),
);

const routes = [
  "/",
  "/about",
  "/projects",
  "/blog",
  "/resume",
  ...projects.map((p) => `/projects/${p.slug}`),
  ...posts.map((p) => `/blog/${p.slug}`),
];

for (const route of routes) {
  test(`renders ${route} without errors`, async ({ page }) => {
    const errors = collectPageErrors(page);
    const response = await page.goto(route);
    expect(response?.status()).toBe(200);
    await expect(page.locator("h1").first()).toBeVisible();
    // goto already waited for `load`; give hydration a beat beyond it so
    // client-side crashes get counted
    await page.waitForTimeout(300);
    expect(errors).toEqual([]);
  });
}

test("unknown routes 404", async ({ page }) => {
  const response = await page.goto("/definitely-not-a-page");
  expect(response?.status()).toBe(404);
});

test("nav links reach their pages", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Projects", exact: true }).first().click();
  await expect(page).toHaveURL(/\/projects$/);
  await expect(page.locator("h1.page-title")).toHaveText("Projects");
});

test("resume PDF is served", async ({ request }) => {
  // The PDF is fetched at build time with a token (gitignored); a tokenless CI
  // build legitimately lacks it. Live and local builds must serve it.
  test.skip(
    !process.env.PLAYWRIGHT_BASE_URL && !existsSync(join(root, "public/resume.pdf")),
    "resume.pdf not staged in this build (no GITHUB_TOKEN)",
  );
  const res = await request.get("/resume.pdf");
  expect(res.status()).toBe(200);
  expect(res.headers()["content-type"]).toContain("pdf");
});

test("VLA embedding assets are deployed and reachable", async ({ request }) => {
  // scripts/copy-vla-assets.mjs must have staged mini-vla's assets into the
  // build, under the VERSIONED path Hero.tsx actually requests; a miss here is
  // the "Start Training silently dies" failure mode. The copy is RECURSIVE and
  // since v0.5.0 the tree nests assets/replay/ (the fallback's manifest +
  // checkpoint bins), so walk it and check every FILE — requesting a bare
  // directory would 404. Both sides derive from mini-vla/package.json (see
  // tests/unit/vla-assets.test.ts).
  const assetsRoot = join(root, "node_modules/mini-vla/assets");
  const walk = (dir: string): string[] =>
    readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const abs = join(dir, entry.name);
      return entry.isDirectory() ? walk(abs) : [relative(assetsRoot, abs)];
    });
  const files = walk(assetsRoot);
  expect(files.length).toBeGreaterThan(0);
  for (const rel of files) {
    const path = `${VLA_ASSET_BASE}/${rel.split(sep).join("/")}`;
    const res = await request.get(path);
    expect(res.status(), path).toBe(200);
    expect((await res.body()).length, path).toBeGreaterThan(0);
  }
});

test("VLA assets are cached immutably under their versioned path", async ({
  request,
}) => {
  // Safe only because the path carries the version — see public/_headers.
  const res = await request.get(`${VLA_ASSET_BASE}/vocab.txt`);
  expect(res.status()).toBe(200);
  const cc = res.headers()["cache-control"] ?? "";
  expect(cc).toContain("immutable");
  expect(cc).toMatch(/max-age=31536000/);
});

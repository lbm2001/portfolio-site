import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Post } from "../../lib/content";

// tests/e2e/routes.spec.ts and caching.spec.ts sweep /blog/[slug] by mapping
// over the real, committed lib/posts-data.json — which is currently `[]` (no
// posts published yet). That sweep silently generates ZERO test cases while
// the site has no posts, so a broken page component would ship completely
// undetected until the day a post is actually published (review round 1,
// finding #12). This test exercises the page's own logic against a synthetic
// fixture instead, independent of whatever lib/posts-data.json currently
// holds.

const FIXTURE_POSTS: Post[] = [
  {
    slug: "written-post",
    date: "2026-01-01",
    cat: "Notes",
    title: "A Written Post",
    excerpt: "An excerpt.",
    body: "Full **body** text.",
    aiAssisted: true,
  },
  {
    slug: "stub-post",
    date: "2026-02-01",
    cat: "Notes",
    title: "A Stub Post",
    excerpt: "Coming soon.",
    // no body — the "hasn't been written yet" placeholder path
  },
];

vi.mock("@/lib/content", () => ({
  posts: FIXTURE_POSTS,
  getPost: (slug: string) => FIXTURE_POSTS.find((p) => p.slug === slug),
  profile: { name: "Test Name" },
}));

// Flatten a React element tree into its rendered text, without a DOM —
// matches this repo's existing preference (see richtext.test.tsx) for
// asserting on structure directly rather than adding a rendering dependency.
// Deliberately does NOT execute nested function components (Nav uses hooks
// and would throw outside React's own render loop) — dangerouslySetInnerHTML
// (renderBody's output) is included as raw HTML, which is enough for a
// substring check.
function text(node: unknown): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(text).join("");
  const el = node as ReactElement<{
    children?: unknown;
    dangerouslySetInnerHTML?: { __html: string };
  }>;
  if (el?.props?.dangerouslySetInnerHTML) {
    return el.props.dangerouslySetInnerHTML.__html;
  }
  if (el?.props?.children !== undefined) return text(el.props.children);
  return "";
}

// True if some element in the tree was constructed with the given component
// function as its type — a structural check that doesn't require executing
// the component (AiDisclaimer has no hooks so text() could technically call
// it, but checking for its presence directly is more robust to its own
// internals changing).
function containsComponent(node: unknown, component: unknown): boolean {
  if (node == null || typeof node !== "object") return false;
  if (Array.isArray(node)) return node.some((n) => containsComponent(n, component));
  const el = node as ReactElement<{ children?: unknown }>;
  if (el.type === component) return true;
  return containsComponent(el.props?.children, component);
}

describe("app/blog/[slug]/page.tsx", () => {
  it("generateStaticParams maps every post to its slug", async () => {
    const { generateStaticParams } = await import("@/app/blog/[slug]/page");
    expect(generateStaticParams()).toEqual([
      { slug: "written-post" },
      { slug: "stub-post" },
    ]);
  });

  it("generateMetadata titles a known post and falls back for an unknown slug", async () => {
    const { generateMetadata } = await import("@/app/blog/[slug]/page");
    await expect(
      generateMetadata({ params: Promise.resolve({ slug: "written-post" }) }),
    ).resolves.toEqual({ title: "A Written Post · Test Name" });
    await expect(
      generateMetadata({ params: Promise.resolve({ slug: "missing" }) }),
    ).resolves.toEqual({ title: "Post" });
  });

  it("renders a written post's title, excerpt, body, and AI disclaimer", async () => {
    const { default: BlogPostPage } = await import("@/app/blog/[slug]/page");
    const { default: AiDisclaimer } = await import("@/components/AiDisclaimer");
    const el = await BlogPostPage({
      params: Promise.resolve({ slug: "written-post" }),
    });
    const rendered = text(el);
    expect(rendered).toContain("A Written Post");
    expect(rendered).toContain("An excerpt.");
    expect(rendered).toContain("<strong>body</strong>"); // renderBody's markdown output
    expect(containsComponent(el, AiDisclaimer)).toBe(true);
  });

  it("renders the not-yet-written placeholder for a post with no body", async () => {
    const { default: BlogPostPage } = await import("@/app/blog/[slug]/page");
    const { default: AiDisclaimer } = await import("@/components/AiDisclaimer");
    const el = await BlogPostPage({
      params: Promise.resolve({ slug: "stub-post" }),
    });
    const rendered = text(el);
    expect(rendered).toContain("hasn’t been written yet");
    expect(containsComponent(el, AiDisclaimer)).toBe(false);
  });

  it("calls notFound() for an unknown slug", async () => {
    const { default: BlogPostPage } = await import("@/app/blog/[slug]/page");
    await expect(
      BlogPostPage({ params: Promise.resolve({ slug: "does-not-exist" }) }),
    ).rejects.toThrow(/NEXT_HTTP_ERROR_FALLBACK|NOT_FOUND/);
  });
});

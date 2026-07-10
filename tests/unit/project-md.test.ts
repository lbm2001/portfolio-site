import { describe, expect, it } from "vitest";
import { parseProjectMd } from "../../lib/project-md";

describe("parseProjectMd", () => {
  it("parses the full documented frontmatter shape", () => {
    const md = [
      "<!--",
      "title: Project Title",
      "venue: Some Lab",
      "period: Apr–Sep 2025",
      "blurb: One-sentence teaser.",
      "aiAssisted: true",
      "tags:",
      "  - Graph Neural Networks",
      "  - MuJoCo",
      "links:",
      "  - label: PyPI",
      "    href: https://pypi.org/project/mujopy/",
      "-->",
      "Markdown body...",
    ].join("\n");
    expect(parseProjectMd(md)).toEqual({
      title: "Project Title",
      venue: "Some Lab",
      period: "Apr–Sep 2025",
      blurb: "One-sentence teaser.",
      aiAssisted: true,
      tags: ["Graph Neural Networks", "MuJoCo"],
      links: [{ label: "PyPI", href: "https://pypi.org/project/mujopy/" }],
      body: "Markdown body...",
    });
  });

  it("parses inline tags arrays", () => {
    const out = parseProjectMd('---\ntags: [a, "b c", d]\n---\nbody');
    expect(out.tags).toEqual(["a", "b c", "d"]);
  });

  it("accepts links with href before label", () => {
    const md = [
      "---",
      "links:",
      "  - href: https://example.com",
      "    label: Site",
      "---",
      "body",
    ].join("\n");
    expect(parseProjectMd(md).links).toEqual([
      { label: "Site", href: "https://example.com" },
    ]);
  });

  it("drops links missing a label or href", () => {
    const md = "---\nlinks:\n  - label: only-label\n---\nbody";
    expect(parseProjectMd(md).links).toBeUndefined();
  });

  it("treats a file without frontmatter as body-only", () => {
    expect(parseProjectMd("# Readme\n\ntext")).toEqual({ body: "# Readme\n\ntext" });
  });
});

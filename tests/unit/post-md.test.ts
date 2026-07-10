import { describe, expect, it } from "vitest";
import { parsePostMd } from "../../lib/post-md";

describe("parsePostMd", () => {
  it("parses HTML-comment frontmatter and strips the markers", () => {
    const md = [
      "<!---",
      "title: Attention From Scratch",
      "date: Jul 2026",
      "cat: ML",
      "excerpt: A teaser.",
      "aiAssisted: true",
      "--->",
      "Body **here**.",
    ].join("\n");
    expect(parsePostMd(md)).toEqual({
      title: "Attention From Scratch",
      date: "Jul 2026",
      cat: "ML",
      excerpt: "A teaser.",
      aiAssisted: true,
      body: "Body **here**.",
    });
  });

  it("parses --- fence frontmatter", () => {
    const md = "---\ntitle: Post\ncat: Robotics\n---\nBody.";
    const out = parsePostMd(md);
    expect(out.title).toBe("Post");
    expect(out.cat).toBe("Robotics");
    expect(out.body).toBe("Body.");
  });

  it("treats a file without frontmatter as body-only", () => {
    expect(parsePostMd("Just a body.\n\nSecond paragraph.")).toEqual({
      body: "Just a body.\n\nSecond paragraph.",
    });
  });

  it("strips quotes, ignores comments/unknown keys, normalizes CRLF and BOM", () => {
    const md = '﻿<!--\r\n# a comment line\r\ntitle: "Quoted"\r\nbogus: ignored\r\n-->\r\nBody.';
    const out = parsePostMd(md);
    expect(out.title).toBe("Quoted");
    expect(out.body).toBe("Body.");
    expect(out).not.toHaveProperty("bogus");
  });

  it("only accepts true/yes/1 for aiAssisted", () => {
    expect(parsePostMd("---\naiAssisted: yes\n---\nb").aiAssisted).toBe(true);
    expect(parsePostMd("---\naiAssisted: false\n---\nb").aiAssisted).toBe(false);
    expect(parsePostMd("b").aiAssisted).toBeUndefined();
  });

  it("keeps a $-math body intact (no accidental frontmatter match mid-file)", () => {
    const md = "Intro\n---\nnot: frontmatter\n---\nmore";
    // the fence is not at the start of the file, so nothing is parsed as keys
    expect(parsePostMd(md).body).toBe(md);
  });
});

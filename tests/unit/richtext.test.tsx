import { describe, expect, it } from "vitest";
import { renderBody } from "../../lib/richtext";

// renderBody returns <div dangerouslySetInnerHTML={{ __html }} />; grab the
// HTML string so we can assert on the build-time render without a DOM.
function html(md: string): string {
  const el = renderBody(md) as React.ReactElement<{
    dangerouslySetInnerHTML: { __html: string };
  }>;
  return el.props.dangerouslySetInnerHTML.__html;
}

describe("renderBody", () => {
  it("renders GitHub-flavored markdown", () => {
    const out = html("# Title\n\nSome **bold** and a [link](https://x.dev).\n\n| a | b |\n| - | - |\n| 1 | 2 |");
    expect(out).toContain("<h1>Title</h1>");
    expect(out).toContain("<strong>bold</strong>");
    // Outward-facing links open in a new tab.
    expect(out).toContain('<a href="https://x.dev" target="_blank" rel="noopener noreferrer">link</a>');
    expect(out).toContain("<table>");
  });

  it("renders $..$ as inline KaTeX and $$..$$ as display math", () => {
    const out = html("Inline $e = mc^2$ here.\n\n$$\\int_0^1 x\\,dx$$");
    expect(out).toContain('class="math-inline"');
    expect(out).toContain('class="math-display"');
    expect(out).toContain("katex");
  });

  it("turns a standalone captioned image line into a <figure>", () => {
    const out = html('![alt text](/projects/x.png "The caption")');
    expect(out).toContain('<figure class="body-figure">');
    expect(out).toContain('<img src="/projects/x.png" alt="alt text" />');
    expect(out).toContain("<figcaption>The caption</figcaption>");
  });

  it("keeps a mid-sentence image inline (no figure)", () => {
    const out = html("An inline ![icon](/i.png) image.");
    expect(out).not.toContain("<figure");
    expect(out).toContain("<img");
  });

  it("escapes HTML in figure captions and alt text", () => {
    const out = html('![<b>](/x.png "<script>")');
    expect(out).not.toContain("<script>");
    expect(out).toContain("&lt;script&gt;");
  });
});

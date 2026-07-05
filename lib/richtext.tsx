import { marked } from "marked";
import type { Tokens } from "marked";
import katex from "katex";

// Renders a Markdown article body (project write-ups + blog posts) to HTML at
// build time. Standard GitHub-flavored Markdown — headings, bold/italic, lists,
// links, code, blockquotes, tables — so a file renders the SAME on GitHub and on
// the site (these bodies live in each project's repo as `portfolio.md`). On top
// of plain Markdown we add two extensions:
//   $..$ / $$..$$        -> inline / display math (KaTeX, rendered at build)
//   ![alt](src "caption") -> a <figure> with the title as its <figcaption>,
//                            when the image is alone on its own line (a normal
//                            inline image mid-sentence still renders as <img>).
// Output classes (math-inline, math-display, body-figure) match app/globals.css.

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const blockMath = {
  name: "blockMath",
  level: "block" as const,
  start(src: string) {
    const i = src.indexOf("$$");
    return i < 0 ? undefined : i;
  },
  tokenizer(src: string) {
    const m = /^\$\$([\s\S]+?)\$\$/.exec(src);
    if (m) return { type: "blockMath", raw: m[0], text: m[1].trim() };
  },
  renderer(token: Tokens.Generic) {
    const html = katex.renderToString(token.text as string, {
      displayMode: true,
      throwOnError: false,
    });
    return `<div class="math-display">${html}</div>\n`;
  },
};

const inlineMath = {
  name: "inlineMath",
  level: "inline" as const,
  start(src: string) {
    const i = src.indexOf("$");
    return i < 0 ? undefined : i;
  },
  tokenizer(src: string) {
    const m = /^\$([^\n$]+?)\$/.exec(src);
    if (m) return { type: "inlineMath", raw: m[0], text: m[1].trim() };
  },
  renderer(token: Tokens.Generic) {
    const html = katex.renderToString(token.text as string, {
      throwOnError: false,
    });
    return `<span class="math-inline">${html}</span>`;
  },
};

// A standalone image line becomes a captioned <figure>; the optional Markdown
// title ("...") is the caption. Kept block-level so we never emit an (invalid)
// <figure> nested inside a <p>.
const figure = {
  name: "figure",
  level: "block" as const,
  start(src: string) {
    const m = /(^|\n)!\[/.exec(src);
    return m ? m.index + (m[1] ? 1 : 0) : undefined;
  },
  tokenizer(src: string) {
    const m = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)[ \t]*(?:\n|$)/.exec(src);
    if (m)
      return {
        type: "figure",
        raw: m[0],
        alt: m[1],
        href: m[2],
        caption: m[3] || "",
      };
  },
  renderer(token: Tokens.Generic) {
    const cap = token.caption
      ? `<figcaption>${escapeHtml(token.caption as string)}</figcaption>`
      : "";
    return `<figure class="body-figure"><img src="${token.href}" alt="${escapeHtml(
      token.alt as string,
    )}" />${cap}</figure>\n`;
  },
};

marked.use({ gfm: true, extensions: [blockMath, inlineMath, figure] });

/** Markdown body -> HTML string, rendered at build time. */
export function renderMarkdown(md: string): string {
  return marked.parse(md, { async: false }) as string;
}

/** Convenience wrapper returning the rendered body as a React element. */
export function renderBody(body: string): React.ReactNode {
  return <div dangerouslySetInnerHTML={{ __html: renderMarkdown(body) }} />;
}

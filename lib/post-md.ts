// A small, TARGETED parser for a blog post's `index.md`: a YAML-ish frontmatter
// block followed by a Markdown body. Sibling of lib/project-md.ts — same idea,
// different keys. NOT a general YAML parser; it understands exactly the scalar
// keys a post uses. Keeping it hand-rolled avoids a YAML dependency in the build.
//
// The frontmatter block can be delimited two ways (same as project-md.ts):
//   1) an HTML comment  <!-- ... -->  — hidden when GitHub renders the file, so
//      the repo page shows only the body (extra dashes like <!--- ... ---> are OK), or
//   2) a `---` YAML fence — which GitHub renders as a table at the top of the file.
//
// title/date/cat/excerpt fall back to sensible defaults in gen-blog-data.mjs when
// omitted, so a post with only a body still renders.
//
// Frontmatter shape:
//   <!---
//   title: Attention From Scratch
//   date: Jul 2026
//   cat: ML
//   excerpt: One-line teaser shown in the list and atop the article.
//   aiAssisted: true
//   --->
//   Markdown body...

import { splitFrontmatter, stripQuotes } from "./frontmatter";

export interface PostMd {
  title?: string;
  date?: string;
  cat?: string;
  excerpt?: string;
  aiAssisted?: boolean;
  body: string;
}

export function parsePostMd(raw: string): PostMd {
  const { block, body } = splitFrontmatter(raw);
  if (block === null) return { body };

  const out: PostMd = { body };

  for (const line of block.split("\n")) {
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const kv = /^([A-Za-z][\w]*):\s*(.*)$/.exec(line);
    if (!kv) continue;
    const key = kv[1];
    const value = kv[2].trim();

    if (key === "aiAssisted") {
      out.aiAssisted = /^(true|yes|1)$/i.test(value);
      continue;
    }

    const clean = stripQuotes(value);
    if (key === "title") out.title = clean;
    else if (key === "date") out.date = clean;
    else if (key === "cat") out.cat = clean;
    else if (key === "excerpt") out.excerpt = clean;
  }

  return out;
}

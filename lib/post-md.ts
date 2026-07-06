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

export interface PostMd {
  title?: string;
  date?: string;
  cat?: string;
  excerpt?: string;
  aiAssisted?: boolean;
  body: string;
}

function stripQuotes(s: string): string {
  const t = s.trim();
  if (
    (t.startsWith('"') && t.endsWith('"')) ||
    (t.startsWith("'") && t.endsWith("'"))
  ) {
    return t.slice(1, -1);
  }
  return t;
}

export function parsePostMd(raw: string): PostMd {
  // Normalize newlines; split off the frontmatter block if present. It may be an
  // HTML comment (<!-- ... -->, hidden on GitHub) or a --- fence; try the comment
  // first since a leading comment is unambiguous.
  const src = raw.replace(/\r\n/g, "\n").replace(/^﻿/, "");
  const comment = /^<!--[\s\S]*?-->[ \t]*\n?/.exec(src);
  const fence = /^---\n([\s\S]*?)\n---\n?/.exec(src);

  let block: string;
  let body: string;
  if (comment) {
    // Drop the <!-- and --> markers (plus any extra dashes) to get the raw
    // key/value lines. Non-`key: value` lines (e.g. stray dashes) are ignored below.
    block = comment[0].replace(/^<!--+/, "").replace(/--+>[ \t]*\n?$/, "");
    body = src.slice(comment[0].length).trim();
  } else if (fence) {
    block = fence[1];
    body = src.slice(fence[0].length).trim();
  } else {
    return { body: src.trim() };
  }

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

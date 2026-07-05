// A small, TARGETED parser for the `portfolio.md` file each project repo holds:
// a YAML-ish frontmatter block followed by a Markdown body. Like lib/resume.ts,
// this is NOT a general YAML parser — it understands exactly the keys the portfolio
// uses (scalars, an inline `tags` list, and a `links` list of {label, href}).
// Keeping it hand-rolled avoids a YAML dependency in the build.
//
// The frontmatter block can be delimited two ways:
//   1) an HTML comment  <!-- ... -->  — hidden when GitHub renders the README, so
//      the repo page shows only the body (extra dashes like <!--- ... ---> are OK), or
//   2) a `---` YAML fence — which GitHub renders as a table at the top of the file.
//
// Every field is optional. title/blurb/tags fall back to the repo's GitHub name /
// description / topics (see gen-projects-data.mjs), so they can be omitted here and
// defined on GitHub instead; venue/period/aiAssisted/links have no GitHub fallback.
//
// Frontmatter shape:
//   <!--
//   title: Project Title
//   venue: Some Lab
//   period: Apr–Sep 2025
//   blurb: One-sentence teaser (falls back to the repo's GitHub description).
//   aiAssisted: true
//   tags:
//     - Graph Neural Networks
//     - MuJoCo
//   links:
//     - label: PyPI
//       href: https://pypi.org/project/mujopy/
//   -->
//   Markdown body...

export interface ProjectMdLink {
  label: string;
  href: string;
}
export interface ProjectMd {
  title?: string;
  venue?: string;
  period?: string;
  blurb?: string;
  aiAssisted?: boolean;
  tags?: string[];
  links?: ProjectMdLink[];
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

export function parseProjectMd(raw: string): ProjectMd {
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

  const out: ProjectMd = { body };

  const lines = block.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim() || line.trim().startsWith("#")) continue;

    const kv = /^([A-Za-z][\w]*):\s*(.*)$/.exec(line);
    if (!kv) continue;
    const key = kv[1];
    const value = kv[2].trim();

    // `tags` / `links` may be an inline value or a following indented list.
    if (key === "tags") {
      const tags: string[] = [];
      // inline JSON-ish array: tags: [a, b]
      const inline = value.match(/^\[(.*)\]$/);
      if (inline) {
        for (const t of inline[1].split(","))
          if (t.trim()) tags.push(stripQuotes(t));
      }
      // indented "- item" lines
      while (i + 1 < lines.length && /^\s*-\s+/.test(lines[i + 1])) {
        tags.push(stripQuotes(lines[++i].replace(/^\s*-\s+/, "")));
      }
      if (tags.length) out.tags = tags;
      continue;
    }

    if (key === "links") {
      const links: ProjectMdLink[] = [];
      // indented list of "- label: X" then "  href: Y" (order-insensitive)
      while (i + 1 < lines.length && /^\s*-\s+/.test(lines[i + 1])) {
        const first = lines[++i].replace(/^\s*-\s+/, "");
        const cur: Partial<ProjectMdLink> = {};
        const apply = (s: string) => {
          const m = /^(label|href):\s*(.*)$/.exec(s.trim());
          if (m) cur[m[1] as "label" | "href"] = stripQuotes(m[2]);
        };
        apply(first);
        while (
          i + 1 < lines.length &&
          /^\s+\w+:/.test(lines[i + 1]) &&
          !/^\s*-\s+/.test(lines[i + 1])
        ) {
          apply(lines[++i]);
        }
        if (cur.label && cur.href) links.push(cur as ProjectMdLink);
      }
      if (links.length) out.links = links;
      continue;
    }

    if (key === "aiAssisted") {
      out.aiAssisted = /^(true|yes|1)$/i.test(value);
      continue;
    }

    const clean = stripQuotes(value);
    if (key === "title") out.title = clean;
    else if (key === "venue") out.venue = clean;
    else if (key === "period") out.period = clean;
    else if (key === "blurb") out.blurb = clean;
  }

  return out;
}

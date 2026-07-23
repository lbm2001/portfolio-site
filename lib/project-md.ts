// A small, TARGETED parser for the `README.md` file each project repo holds:
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

import { parseAiAssisted, parseKvLine, splitFrontmatter, stripQuotes } from "./frontmatter.ts";
// Same shape lib/content.ts's Project.links renders — share it instead of
// defining an identical type twice, so a field added to one can't silently
// diverge from the other (review round 1, finding #20).
import type { ProjectLink } from "./content.ts";

export interface ProjectMd {
  title?: string;
  venue?: string;
  period?: string;
  blurb?: string;
  aiAssisted?: boolean;
  tags?: string[];
  links?: ProjectLink[];
  body: string;
}

export function parseProjectMd(raw: string): ProjectMd {
  const { block, body } = splitFrontmatter(raw);
  if (block === null) return { body };

  const out: ProjectMd = { body };

  const lines = block.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const kv = parseKvLine(line);
    if (!kv) continue;
    const { key, value } = kv;

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
      const links: ProjectLink[] = [];
      // indented list of "- label: X" then "  href: Y" (order-insensitive)
      while (i + 1 < lines.length && /^\s*-\s+/.test(lines[i + 1])) {
        const first = lines[++i].replace(/^\s*-\s+/, "");
        const cur: Partial<ProjectLink> = {};
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
        if (cur.label && cur.href) links.push(cur as ProjectLink);
      }
      if (links.length) out.links = links;
      continue;
    }

    if (key === "aiAssisted") {
      out.aiAssisted = parseAiAssisted(value);
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

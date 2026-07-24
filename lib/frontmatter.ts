// Shared frontmatter splitting for the hand-rolled `.md` parsers (lib/post-md.ts
// and lib/project-md.ts). Both files describe a YAML-ish frontmatter block
// followed by a Markdown body, delimited two ways:
//   1) an HTML comment  <!-- ... -->  — hidden when GitHub renders the file, so
//      the repo page shows only the body (extra dashes like <!--- ... ---> are OK), or
//   2) a `---` YAML fence — which GitHub renders as a table at the top of the file.
// This module owns the delimiter detection and quote stripping; each parser keeps
// only its own key handling. NOT a general YAML parser — see the callers.

export function stripQuotes(s: string): string {
  const t = s.trim();
  if (
    (t.startsWith('"') && t.endsWith('"')) ||
    (t.startsWith("'") && t.endsWith("'"))
  ) {
    return t.slice(1, -1);
  }
  return t;
}

/** Parse one frontmatter line as `key: value`. Returns null for a blank line,
    a `#` comment, or anything not matching that shape — callers skip those. */
export function parseKvLine(line: string): { key: string; value: string } | null {
  if (!line.trim() || line.trim().startsWith("#")) return null;
  const kv = /^([A-Za-z][\w]*):\s*(.*)$/.exec(line);
  if (!kv) return null;
  return { key: kv[1], value: kv[2].trim() };
}

/** The `aiAssisted: true|yes|1` convention both parsers use. */
export function parseAiAssisted(value: string): boolean {
  return /^(true|yes|1)$/i.test(value);
}

/** Split raw file text into its frontmatter `block` (raw key/value lines, markers
    stripped) and the Markdown `body`. Returns `block: null` when there is no
    frontmatter — the whole input is the body. Tries the comment form first, since
    a leading comment is unambiguous. */
export function splitFrontmatter(raw: string): {
  block: string | null;
  body: string;
} {
  // Normalize newlines and drop a leading BOM before matching.
  const src = raw.replace(/\r\n/g, "\n").replace(/^﻿/, "");
  const comment = /^<!--[\s\S]*?-->[ \t]*\n?/.exec(src);
  const fence = /^---\n([\s\S]*?)\n---\n?/.exec(src);

  if (comment) {
    // Drop the <!-- and --> markers (plus any extra dashes) to get the raw
    // key/value lines. Non-`key: value` lines (e.g. stray dashes) are ignored by callers.
    const block = comment[0].replace(/^<!--+/, "").replace(/--+>[ \t]*\n?$/, "");
    return { block, body: src.slice(comment[0].length).trim() };
  }
  if (fence) {
    return { block: fence[1], body: src.slice(fence[0].length).trim() };
  }
  return { block: null, body: src.trim() };
}

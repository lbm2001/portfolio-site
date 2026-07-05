// A small, TARGETED parser for the "Jake Gutierrez" resume template that
// public/resume.tex uses — NOT a general LaTeX parser. It understands exactly the
// custom commands that template defines:
//   \begin{center} … name + \href contact links … \end{center}
//   \section{Title}
//   \resumeSubheading{title}{dates}{org}{location}
//   \resumeItem{bullet text}
//   Skills:  \textbf{Category}{: comma list} \\
// If the .tex's command set changes, this parser needs a matching tweak.

export interface ResumeContact {
  label: string;
  href: string;
}
export interface ResumeEntry {
  title: string;
  dates: string;
  org: string;
  location: string;
  bullets: string[];
}
export interface ResumeSkill {
  category: string;
  items: string;
}
export interface ResumeSection {
  title: string;
  entries: ResumeEntry[];
  skills: ResumeSkill[]; // used only by the Skills section
}
export interface Resume {
  name: string;
  location: string;
  contacts: ResumeContact[];
  sections: ResumeSection[];
}

// ---- LaTeX → plain text cleanup ---------------------------------------------

// Replace \href{url}{...text...} with a sentinel we can turn into a link later,
// but for inline body text we just keep the visible text.
function stripInline(s: string): string {
  let t = s;
  // \href{url}{text} → text  (drop \underline inside)
  t = t.replace(/\\href\{[^}]*\}\{([^}]*)\}/g, "$1");
  // remove common formatting wrappers, keeping their content
  t = t.replace(/\\(?:textbf|textit|underline|emph|small|scshape|href)\b/g, "");
  // \& → &, \% → %, \$ → $, \# → #, \_ → _
  t = t.replace(/\\([&%$#_])/g, "$1");
  // ranges and dashes
  t = t.replace(/---/g, "—").replace(/--/g, "–");
  // ties and spacing macros
  t = t.replace(/~/g, " ").replace(/\\,/g, " ").replace(/\\ /g, " ");
  // drop leftover \vspace{...}, \\ line breaks, stray braces from macros
  t = t.replace(/\\vspace\{[^}]*\}/g, "");
  t = t.replace(/\\\\/g, " ");
  t = t.replace(/\$\|\$/g, "|");
  t = t.replace(/[{}]/g, "");
  return t.replace(/\s+/g, " ").trim();
}

// Split a LaTeX command's brace-delimited arguments starting at `from`
// (which must point at the first "{"). Returns the args and the index after
// the last closing brace. Handles nested braces.
function readArgs(src: string, from: number, count: number): { args: string[]; end: number } {
  const args: string[] = [];
  let i = from;
  for (let a = 0; a < count; a++) {
    while (i < src.length && src[i] !== "{") i++;
    if (src[i] !== "{") break;
    let depth = 0;
    let start = i + 1;
    for (; i < src.length; i++) {
      if (src[i] === "{") depth++;
      else if (src[i] === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    args.push(src.slice(start, i));
    i++; // step past the closing brace
  }
  return { args, end: i };
}

// ---- Parser -----------------------------------------------------------------

export function parseResume(tex: string): Resume {
  // strip full-line comments so % inside them never confuses us
  const src = tex.replace(/(^|[^\\])%.*$/gm, "$1");

  // --- header (name + contacts) from the \begin{center} … \end{center} block
  let name = "";
  let location = "";
  const contacts: ResumeContact[] = [];
  const center = src.match(/\\begin\{center\}([\s\S]*?)\\end\{center\}/);
  if (center) {
    const block = center[1];
    const nameM = block.match(/\\textbf\{[^}]*?\\scshape\s*([^}\\]+)\}/) || block.match(/\\scshape\s*([^}\\]+)/);
    if (nameM) name = stripInline(nameM[1]);
    // first \small line usually holds "City, Country | href | href | href"
    // location = leading text before the first \href
    const small = block.split("\\small")[1] || block;
    const beforeFirstHref = small.split("\\href")[0];
    location = stripInline(beforeFirstHref).replace(/\|+\s*$/, "").trim();
    const hrefRe = /\\href\{([^}]*)\}\{([^}]*)\}/g;
    let m: RegExpExecArray | null;
    while ((m = hrefRe.exec(block))) {
      const href = m[1].trim();
      const label = stripInline(m[2]);
      contacts.push({ label, href });
    }
  }

  // --- sections
  const sections: ResumeSection[] = [];
  const sectionRe = /\\section\{([^}]*)\}/g;
  const heads: { title: string; index: number }[] = [];
  let sm: RegExpExecArray | null;
  while ((sm = sectionRe.exec(src))) {
    heads.push({ title: stripInline(sm[1]), index: sm.index });
  }

  for (let h = 0; h < heads.length; h++) {
    const start = heads[h].index;
    const end = h + 1 < heads.length ? heads[h + 1].index : src.length;
    const body = src.slice(start, end);
    const section: ResumeSection = { title: heads[h].title, entries: [], skills: [] };

    if (/^skills$/i.test(heads[h].title)) {
      // Skills: lines like  \textbf{Category}{: item, item} \\
      const skillRe = /\\textbf\{([^}]*)\}\{:?\s*([^}]*)\}/g;
      let km: RegExpExecArray | null;
      while ((km = skillRe.exec(body))) {
        const category = stripInline(km[1]);
        const items = stripInline(km[2]).replace(/^:\s*/, "");
        if (category) section.skills.push({ category, items });
      }
      sections.push(section);
      continue;
    }

    // Regular sections: walk \resumeSubheading / \resumeItem in order.
    let cur: ResumeEntry | null = null;
    const tokenRe = /\\resume(Subheading|Item)\b/g;
    let tm: RegExpExecArray | null;
    while ((tm = tokenRe.exec(body))) {
      if (tm[1] === "Subheading") {
        const { args } = readArgs(body, tm.index + tm[0].length, 4);
        cur = {
          title: stripInline(args[0] || ""),
          dates: stripInline(args[1] || ""),
          org: stripInline(args[2] || ""),
          location: stripInline(args[3] || ""),
          bullets: [],
        };
        section.entries.push(cur);
      } else {
        const { args } = readArgs(body, tm.index + tm[0].length, 1);
        const text = stripInline(args[0] || "");
        if (text && cur) cur.bullets.push(text);
      }
    }
    sections.push(section);
  }

  return { name, location, contacts, sections };
}

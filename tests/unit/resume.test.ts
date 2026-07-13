import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parseResume } from "../../lib/resume";

const FIXTURE = String.raw`
% a comment line with \section{Nope} inside
\begin{center}
  \textbf{\Huge \scshape Lukas M\"uller} \\ \vspace{1pt}
  \small Frankfurt, Germany $|$
  \href{mailto:contact@lukasmueller.dev}{\underline{contact@lukasmueller.dev}} $|$
  \href{https://github.com/lukasmueller-dev}{\underline{github.com/lukasmueller-dev}}
\end{center}

\section{Experience}
\resumeSubheading{Research Assistant}{Apr 2025 -- Sep 2025}{Some Lab \& Institute}{Frankfurt}
\resumeItem{Built a \textbf{VLA} pipeline with 95\% success.}
\resumeItem{Nested braces: \href{https://x.dev}{link text} end.}

\section{Projects}
\resumeSubheading
  {Optimal vs. Heuristic Policies for EV Charging}{Jan. 2026 -- Jun. 2026}
  {Aarhus University}{\href{https://github.com/lukasmueller-dev/au-mdt}{\underline{Code}} $\vert$ \href{https://lukasmueller.dev/projects/ev-charging-optimal-vs-heuristics}{\underline{Project Page}}}
\resumeItem{Modeled EV charging as an MDP.}

\section{Skills}
\textbf{Languages}{: Python, TypeScript} \\
\textbf{ML}{: PyTorch, TF.js} \\
`;

describe("parseResume (fixture)", () => {
  const r = parseResume(FIXTURE);

  it("extracts header name, location and contact links", () => {
    expect(r.name).not.toBe("");
    expect(r.location).toBe("Frankfurt, Germany");
    expect(r.contacts).toEqual([
      { label: "contact@lukasmueller.dev", href: "mailto:contact@lukasmueller.dev" },
      { label: "github.com/lukasmueller-dev", href: "https://github.com/lukasmueller-dev" },
    ]);
  });

  it("ignores commented-out sections", () => {
    expect(r.sections.map((s) => s.title)).toEqual(["Experience", "Projects", "Skills"]);
  });

  it("parses subheadings with escapes and attaches bullets in order", () => {
    const exp = r.sections[0].entries[0];
    expect(exp.title).toBe("Research Assistant");
    expect(exp.dates).toBe("Apr 2025 – Sep 2025");
    expect(exp.org).toBe("Some Lab & Institute");
    expect(exp.bullets).toEqual([
      "Built a VLA pipeline with 95% success.",
      "Nested braces: link text end.",
    ]);
  });

  it("keeps the 4th arg as a location for non-project entries", () => {
    const exp = r.sections[0].entries[0];
    expect(exp.location).toBe("Frankfurt");
    expect(exp.slug).toBeUndefined();
  });

  it("pulls the slug from a project entry's Project Page link and drops the links text", () => {
    const proj = r.sections[1].entries[0];
    expect(proj.title).toBe("Optimal vs. Heuristic Policies for EV Charging");
    expect(proj.slug).toBe("ev-charging-optimal-vs-heuristics");
    // the "Code | Project Page" links must NOT leak into the rendered location
    expect(proj.location).toBe("");
  });

  it("parses the skills section into category/items pairs", () => {
    expect(r.sections[2].skills).toEqual([
      { category: "Languages", items: "Python, TypeScript" },
      { category: "ML", items: "PyTorch, TF.js" },
    ]);
  });
});

// public/resume.tex is fetched at build time from the private source repo and
// is gitignored, so this only runs where a build has staged it (local dev, or
// CI with a token) — the fixture suite above covers tokenless CI.
const texPath = join(__dirname, "../../public/resume.tex");
describe.runIf(existsSync(texPath))("parseResume (fetched public/resume.tex)", () => {
  // Guards the real content path: if the template drifts away from what the
  // parser understands, the /resume page silently loses content.
  //
  // Read inside the test, not in the describe body: Vitest executes a suite's
  // factory during collection even when runIf skips the suite, so a read out
  // here throws ENOENT in tokenless CI, where the file is never staged.
  it("still understands the live template", () => {
    const r = parseResume(readFileSync(texPath, "utf8"));
    expect(r.name.length).toBeGreaterThan(0);
    expect(r.contacts.length).toBeGreaterThan(0);
    expect(r.sections.length).toBeGreaterThan(1);
    const entryCount = r.sections.reduce((n, s) => n + s.entries.length, 0);
    const skillCount = r.sections.reduce((n, s) => n + s.skills.length, 0);
    expect(entryCount + skillCount).toBeGreaterThan(0);
    for (const s of r.sections) {
      for (const e of s.entries) {
        expect(e.title.length).toBeGreaterThan(0);
        // raw LaTeX leaking through the cleanup would render on the page
        expect(e.title).not.toMatch(/[\\{}]/);
        for (const b of e.bullets) expect(b).not.toMatch(/\\(textbf|href|vspace)/);
      }
    }
  });
});

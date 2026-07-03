import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { profile } from "@/lib/content";
import { parseCv } from "@/lib/cv";

export const metadata: Metadata = {
  title: `CV · ${profile.name}`,
};

// Read + parse cv.tex at BUILD time only. This guarantees the page is fully
// prerendered — important on Cloudflare Workers (OpenNext), which has no
// filesystem at runtime, so the fs read must never happen on-demand.
export const dynamic = "force-static";

// Parsed at build time from the LaTeX source in public/.
function loadCv() {
  const tex = fs.readFileSync(path.join(process.cwd(), "public", "cv.tex"), "utf8");
  return parseCv(tex);
}

export default function CvPage() {
  const cv = loadCv();

  return (
    <main>
      <Nav />
      <article className="page page-narrow cv-page">
        <div className="cv-header">
          <div>
            <div className="label-mono">Curriculum Vitae</div>
            <h1 className="page-title">{cv.name || profile.name}</h1>
            <div className="cv-contacts">
              {cv.location && <span>{cv.location}</span>}
              {cv.contacts.map((c) => (
                <a key={c.href} href={c.href}>
                  {c.label}
                </a>
              ))}
            </div>
          </div>
          <a className="cv-btn" href="/cv.pdf">
            Download PDF ↓
          </a>
        </div>

        {cv.sections.map((s) => (
          <section key={s.title} className="cv-section">
            <h2 className="cv-section-title">{s.title}</h2>

            {s.skills.length > 0 && (
              <div className="cv-skills">
                {s.skills.map((k) => (
                  <div key={k.category} className="cv-skill">
                    <span className="cv-skill-cat">{k.category}</span>
                    <span className="cv-skill-items">{k.items}</span>
                  </div>
                ))}
              </div>
            )}

            {s.entries.map((e, i) => (
              <div key={i} className="cv-entry">
                <div className="cv-entry-row">
                  <span className="cv-entry-title">{e.title}</span>
                  {e.dates && <span className="cv-entry-dates">{e.dates}</span>}
                </div>
                {(e.org || e.location) && (
                  <div className="cv-entry-row cv-entry-sub">
                    <span>{e.org}</span>
                    {e.location && <span>{e.location}</span>}
                  </div>
                )}
                {e.bullets.length > 0 && (
                  <ul className="cv-bullets">
                    {e.bullets.map((b, j) => (
                      <li key={j}>{b}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </section>
        ))}
      </article>
      <Footer />
    </main>
  );
}

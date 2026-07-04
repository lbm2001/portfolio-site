import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { profile, resumeDownloadName } from "@/lib/content";
import type { Cv } from "@/lib/cv";
import cvData from "@/lib/cv-data.json";

export const metadata: Metadata = {
  title: `Resume · ${profile.name}`,
};

// The CV is parsed from public/resume.tex into lib/cv-data.json at BUILD time
// (scripts/gen-cv-data.mjs, run via the prebuild step). The page imports that
// JSON statically — it never touches the filesystem at request time. This is
// required on Cloudflare Workers (OpenNext), which has NO runtime filesystem:
// reading the .tex during render threw "Internal Server Error" on re-render.
export const dynamic = "force-static";

const cv = cvData as Cv;

export default function CvPage() {
  return (
    <main>
      <Nav />
      <article className="page page-narrow cv-page">
        <div className="cv-header">
          <div>
            <div className="label-mono">Resume</div>
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

        <div className="cv-download">
          <a className="cv-btn" href="/resume.pdf" download={resumeDownloadName()}>
            Download PDF ↓
          </a>
        </div>
      </article>
      <Footer />
    </main>
  );
}

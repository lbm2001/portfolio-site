import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { profile, resumeDownloadName, getProjectByTitle, externalLinkProps } from "@/lib/content";
import type { Resume } from "@/lib/resume";
import resumeData from "@/lib/resume-data.json";

export const metadata: Metadata = {
  title: `Resume · ${profile.name}`,
};

// The résumé is parsed from public/resume.tex into lib/resume-data.json at BUILD
// time (scripts/gen-resume-data.mjs, run via the prebuild step). The page imports
// that JSON statically — it never touches the filesystem at request time. This is
// required on Cloudflare Workers (OpenNext), which has NO runtime filesystem:
// reading the .tex during render threw "Internal Server Error" on re-render.
export const dynamic = "force-static";

const resume = resumeData as Resume;

// The .tex (and downloadable PDF) list a lukasmueller.dev website link among the
// contacts, which is redundant when the résumé is already being viewed on this
// site — drop it here rather than in the parser/source so the download is unaffected.
const siteContacts = resume.contacts.filter((c) => !/lukasmueller\.dev/i.test(c.href) || c.href.startsWith("mailto:"));

export default function ResumePage() {
  return (
    <main>
      <Nav />
      <article className="page page-narrow resume-page">
        <div className="resume-header">
          <div>
            <div className="label-mono">Resume</div>
            <h1 className="page-title">{resume.name || profile.name}</h1>
            <div className="resume-contacts">
              {resume.location && <span>{resume.location}</span>}
              {siteContacts.map((c) => (
                <a key={c.href} href={c.href} {...externalLinkProps(c.href)}>
                  {c.label}
                </a>
              ))}
            </div>
          </div>
        </div>

        {resume.sections.map((s) => {
          // Only the Projects section links its entries to project pages.
          const isProjects = /projects/i.test(s.title);
          return (
          <section key={s.title} className="resume-section">
            <h2 className="resume-section-title">{s.title}</h2>

            {s.skills.length > 0 && (
              <div className="resume-skills">
                {s.skills.map((k) => (
                  <div key={k.category} className="resume-skill">
                    <span className="resume-skill-cat">{k.category}</span>
                    <span className="resume-skill-items">{k.items}</span>
                  </div>
                ))}
              </div>
            )}

            {s.entries.map((e, i) => {
              const project = isProjects ? getProjectByTitle(e.title) : undefined;
              return (
              <div key={i} className="resume-entry">
                <div className="resume-entry-row">
                  <span className="resume-entry-title">
                    {project ? (
                      <a className="resume-entry-link" href={`/projects/${project.slug}`}>
                        {e.title}
                      </a>
                    ) : (
                      e.title
                    )}
                  </span>
                  {e.dates && <span className="resume-entry-dates">{e.dates}</span>}
                </div>
                {(e.org || e.location) && (
                  <div className="resume-entry-row resume-entry-sub">
                    <span>{e.org}</span>
                    {e.location && <span>{e.location}</span>}
                  </div>
                )}
                {e.bullets.length > 0 && (
                  <ul className="resume-bullets">
                    {e.bullets.map((b, j) => (
                      <li key={j}>{b}</li>
                    ))}
                  </ul>
                )}
              </div>
              );
            })}
          </section>
          );
        })}

        <div className="resume-download">
          <a className="resume-btn" href="/resume.pdf" download={resumeDownloadName()}>
            Download PDF ↓
          </a>
        </div>
      </article>
      <Footer />
    </main>
  );
}

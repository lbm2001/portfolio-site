import Link from "next/link";
import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { projects, profile } from "@/lib/content";

export const metadata: Metadata = {
  title: `Projects · ${profile.name}`,
};

export default function ProjectsPage() {
  return (
    <main>
      <Nav />
      <article className="page">
        <div className="page-head">
          <div className="label-mono">Projects</div>
          <h1 className="page-title">Selected Work</h1>
        </div>

        <div className="work-grid">
          {projects.map((p) => (
            <Link key={p.slug} className="work-card" href={`/projects/${p.slug}`}>
              <span className="work-meta">
                <span className="work-idx">{p.idx}</span>
                <span className="work-venue">{p.venue}</span>
              </span>
              <span className="work-title">{p.title}</span>
              <span className="work-blurb">{p.blurb}</span>
              <span className="work-tags">
                {p.tags.map((t) => (
                  <span key={t} className="tag">
                    {t}
                  </span>
                ))}
              </span>
              <span className="work-links">
                {p.links.map((lk) => (
                  <span key={lk.label} className="work-link">
                    {lk.label}
                  </span>
                ))}
              </span>
            </Link>
          ))}
        </div>
      </article>
      <Footer />
    </main>
  );
}

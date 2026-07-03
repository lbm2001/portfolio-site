import Link from "next/link";
import { projects } from "@/lib/content";

// Home preview: the selected-work grid, each card linking to its detail page.
export default function Work() {
  return (
    <section id="work" className="section">
      <div className="section-head">
        <div className="label-mono">02 — Selected Work</div>
        <Link href="/projects">All projects →</Link>
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
    </section>
  );
}

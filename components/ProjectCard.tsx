import Link from "next/link";
import { externalLinkProps, type Project } from "@/lib/content";

// A single project entry in the selected-work grid. The card itself links to
// its detail page (via a stretched overlay link so it doesn't wrap the whole
// card in an <a>), while the Code/Paper/etc. links jump straight to their
// target and sit above the overlay so they stay independently clickable.
// Shared by the home preview (Work) and the projects index page.
export default function ProjectCard({ project: p }: { project: Project }) {
  return (
    <div className="work-card">
      <Link
        className="work-card-overlay"
        href={`/projects/${p.slug}`}
        aria-label={p.title}
      />
      <span className="work-meta">
        {p.period && <span className="work-period">{p.period}</span>}
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
          <a key={lk.label} className="work-link" href={lk.href} {...externalLinkProps(lk.href)}>
            {lk.label}
          </a>
        ))}
      </span>
    </div>
  );
}

import Link from "next/link";
import type { Project } from "@/lib/content";

// A single project entry in the selected-work grid, linking to its detail page.
// Shared by the home preview (Work) and the projects index page.
export default function ProjectCard({ project: p }: { project: Project }) {
  return (
    <Link className="work-card" href={`/projects/${p.slug}`}>
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
          <span key={lk.label} className="work-link">
            {lk.label}
          </span>
        ))}
      </span>
    </Link>
  );
}

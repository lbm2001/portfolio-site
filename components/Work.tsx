import Link from "next/link";
import { projects } from "@/lib/content";
import ProjectCard from "./ProjectCard";

// Home preview: the selected-work grid, each card linking to its detail page.
export default function Work() {
  return (
    <section id="work" className="section">
      <div className="section-head">
        <div className="label-mono">03 Projects</div>
        <Link href="/projects">See all</Link>
      </div>
      <div className="work-grid">
        {projects.slice(0, 2).map((p) => (
          <ProjectCard key={p.slug} project={p} />
        ))}
      </div>
    </section>
  );
}

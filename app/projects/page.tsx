import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import ProjectCard from "@/components/ProjectCard";
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
          <div className="label-mono">Selected Work</div>
          <h1 className="page-title">Projects</h1>
        </div>

        <div className="work-grid work-list">
          {projects.map((p) => (
            <ProjectCard key={p.slug} project={p} />
          ))}
        </div>
      </article>
      <Footer />
    </main>
  );
}

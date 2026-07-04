import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { projects, getProject, profile } from "@/lib/content";
import { renderBody } from "@/lib/richtext";
import AiDisclaimer from "@/components/AiDisclaimer";

export function generateStaticParams() {
  return projects.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const project = getProject(slug);
  return { title: project ? `${project.title} · ${profile.name}` : "Project" };
}

export default async function ProjectPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const project = getProject(slug);
  if (!project) notFound();

  return (
    <main>
      <Nav />
      <article className="page page-narrow">
        <Link className="back-link" href="/projects">
          ← See all
        </Link>
        <div className="page-head">
          <div className="label-mono">{project.venue}</div>
          <h1 className="page-title">{project.title}</h1>
        </div>

        <p className="section-lead">{project.blurb}</p>

        {project.aiAssisted && <AiDisclaimer />}

        {project.body && (
          <div className="prose">{renderBody(project.body, project.figures)}</div>
        )}

        <div className="work-tags" style={{ marginBottom: 28 }}>
          {project.tags.map((t) => (
            <span key={t} className="tag">
              {t}
            </span>
          ))}
        </div>

        {project.links.length > 0 && (
          <div className="detail-links">
            {project.links.map((lk) => (
              <a key={lk.label} className="detail-link" href={lk.href}>
                {lk.label} →
              </a>
            ))}
          </div>
        )}
      </article>
      <Footer />
    </main>
  );
}

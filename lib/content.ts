import projectsData from "./projects-data.json";

export interface NavLink {
  label: string;
  href: string;
}

export interface ProjectLink {
  label: string;
  href: string;
}

export interface Project {
  slug: string;
  title: string;
  venue: string;
  /** Working period, shown in red in the meta line (e.g. "Apr–Sep 2025").
   *  Projects are listed newest→oldest by start date. */
  period?: string;
  blurb: string;
  tags: string[];
  links: ProjectLink[];
  /**
   * Full write-up shown on the detail page: GitHub-flavored Markdown, plus
   * inline `$..$` / block `$$..$$` math and captioned figures via a standalone
   * `![alt](src "caption")` image line. Rendered by lib/richtext.tsx.
   */
  body?: string;
  /** Shows the AI-assisted-writing disclaimer on the detail page when true. */
  aiAssisted?: boolean;
}

export interface Post {
  slug: string;
  date: string;
  cat: string;
  title: string;
  excerpt: string;
  /** Full article body. Omit until the post is actually written. */
  body?: string;
  /** Shows the AI-assisted-writing disclaimer on the detail page when true. */
  aiAssisted?: boolean;
}

export const profile = {
  name: "Lukas Müller",
  lead: "Building the next generation of intelligent machines. My current focus is on Vision-Language-Action models that enable robots to learn from demonstrations.",
  focus: "Imitation Learning, VLAs",
  field: "Machine Learning & Robot Learning",
  location: "Frankfurt, Germany",
  email: "contact@lukasmueller.dev",
  links: {
    github: "https://github.com/lbm2001",
    linkedin: "https://www.linkedin.com/in/lukas-m-695b06195/",
    email: "mailto:contact@lukasmueller.dev",
  },
};

export const nav: NavLink[] = [
  { label: "About", href: "/about" },
  { label: "Resume", href: "/resume" },
  { label: "Projects", href: "/projects" },
  { label: "Blog", href: "/blog" },
];

// Projects are pulled from each project's GitHub repo at BUILD time by
// scripts/gen-projects-data.mjs (run in the prebuild step) into
// lib/projects-data.json, imported at the top of this file as a static module —
// same pattern as the résumé (lib/resume-data.json). Nothing is fetched at request time.
// Edit a repo's portfolio.md / About / topics to change what shows here.
export const projects = projectsData as Project[];

export const posts: Post[] = [
  // No posts written yet. Add entries here as you write them, e.g.:
  // {
  //   slug: "my-first-post",
  //   date: "Jul 2026",
  //   cat: "ML",
  //   title: "My First Post",
  //   excerpt: "One-line teaser shown in the list and atop the article.",
  //   body: `First paragraph.\n\nSecond paragraph — blank lines separate paragraphs.`,
  // },
];

export const getProject = (slug: string) => projects.find((p) => p.slug === slug);
export const getPost = (slug: string) => posts.find((p) => p.slug === slug);

// Save-as filename for the resume PDF download, stamped with the current
// month/year (e.g. resume_lukas_mueller_07_2026.pdf). The served asset stays
// /resume.pdf — the browser's `download` attribute renames it on save.
export function resumeDownloadName() {
  const d = new Date();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  return `resume_lukas_mueller_${month}_${d.getFullYear()}.pdf`;
}

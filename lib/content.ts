import projectsData from "./projects-data.json";
import postsData from "./posts-data.json";

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
   *  Projects are listed newest→oldest by the period's start; "Ongoing" pins to the top. */
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
  lead: [
    "Building intelligent machines that learn.",
    "Passionate about probability theory, machine learning, and robot learning.",
    "Currently focused on VLA models for imitation learning."
  ],
  focus: "Imitation Learning, VLAs",
  field: "Machine Learning, Robot Learning",
  location: "Frankfurt, Germany",
  email: "contact@lukasmueller.dev",
  links: {
    github: "https://github.com/lukasmueller-dev",
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

// Blog posts are pulled from the single blog repo (config/blog.sources.json) at BUILD
// time by scripts/gen-blog-data.mjs (run in the prebuild step) into
// lib/posts-data.json, imported above as a static module — same pattern as
// projects. Nothing is fetched at request time; posts are sorted newest-first by
// the generator. Publish a post by adding posts/<slug>/index.md to the blog repo.
// An empty blog repo yields [], and the blog page shows "Writing Coming Soon".
export const posts = postsData as Post[];

export const getProject = (slug: string) => projects.find((p) => p.slug === slug);
export const getPost = (slug: string) => posts.find((p) => p.slug === slug);

// Props that open a link in a new tab — but only for outward-facing http(s)
// URLs (GitHub, LinkedIn, papers, demos). Internal paths ("/projects/…") and
// mailto: links get nothing, so they navigate in-place / hand off to the mail
// client. Spread onto any <a> whose href may be external: {...externalLinkProps(href)}.
export const externalLinkProps = (href: string) =>
  /^https?:/i.test(href) ? { target: "_blank", rel: "noopener noreferrer" } : {};

// Normalize a free-text title into a slug-shaped key: lowercase, drop
// parentheticals like "(Team Project)", and collapse anything non-alphanumeric
// into single hyphens. Used to bridge the résumé (parsed from LaTeX, titles
// drift from slugs — "Large-Scale Robot Generation" vs "large-scale-robot-generation").
const slugifyTitle = (s: string) =>
  s
    .toLowerCase()
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

// Match a résumé Projects entry to its project page by comparing the slugified
// title against each project's slug (and slugified title). Returns undefined
// when there's no page — callers fall back to plain text.
export const getProjectByTitle = (title: string) => {
  const key = slugifyTitle(title);
  return projects.find((p) => p.slug === key || slugifyTitle(p.title) === key);
};

// Save-as filename for the resume PDF download, stamped with the current
// month/year (e.g. resume-lukas-mueller-072026.pdf). The served asset stays
// /resume.pdf — the browser's `download` attribute renames it on save.
export function resumeDownloadName() {
  const d = new Date();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  return `resume-lukas-mueller-${month}${d.getFullYear()}.pdf`;
}

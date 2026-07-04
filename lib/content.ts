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
  idx: string;
  title: string;
  venue: string;
  blurb: string;
  tags: string[];
  links: ProjectLink[];
}

export interface Post {
  slug: string;
  date: string;
  cat: string;
  title: string;
  excerpt: string;
}

export const profile = {
  name: "Lukas Müller",
  tagline: "Robot Learning",
  lead: "A two-sentence lead about who you are and the thread connecting your work. This sits large to set the tone — swap for real copy.",
  focus: "RL & Imitation Learning",
  field: "Robotics",
  location: "Frankfurt, Germany",
  email: "contact@lukasmueller.dev",
  links: {
    github: "https://github.com/lbm2001",
    linkedin: "https://www.linkedin.com/in/lukas-m-695b06195/",
    scholar: "#",
    email: "mailto:contact@lukasmueller.dev",
  },
};

export const nav: NavLink[] = [
  { label: "About", href: "/about" },
  { label: "Projects", href: "/projects" },
  { label: "Blog", href: "/blog" },
  { label: "CV", href: "/cv" },
];

export const projects: Project[] = [
  {
    slug: "project-title-placeholder",
    idx: "01",
    title: "Project / Paper Title Placeholder",
    venue: "Venue · 2025",
    blurb:
      "One or two sentences describing the method and the key result. Replace with real content later.",
    tags: ["Reinforcement Learning", "Robotics"],
    links: [
      { label: "Paper", href: "#" },
      { label: "Code", href: "#" },
    ],
  },
  {
    slug: "second-project-placeholder",
    idx: "02",
    title: "Second Project Placeholder",
    venue: "Venue · 2024",
    blurb:
      "A short description of what this project explores and why it matters.",
    tags: ["Imitation Learning"],
    links: [{ label: "Paper", href: "#" }],
  },
  {
    slug: "third-project-placeholder",
    idx: "03",
    title: "Third Project Placeholder",
    venue: "Venue · 2023",
    blurb:
      "Description placeholder for the third highlighted project or publication.",
    tags: ["Locomotion", "Benchmark"],
    links: [
      { label: "Code", href: "#" },
      { label: "Website", href: "#" },
    ],
  },
  {
    slug: "fourth-project-placeholder",
    idx: "04",
    title: "Fourth Project Placeholder",
    venue: "Venue · 2022",
    blurb:
      "Description placeholder text for a fourth entry in your selected works.",
    tags: ["Control"],
    links: [{ label: "Paper", href: "#" }],
  },
];

export const posts: Post[] = [
  {
    slug: "blog-post-title-placeholder",
    date: "Jul 2025",
    cat: "Projects",
    title: "Blog Post Title Placeholder Goes Here",
    excerpt:
      "A short teaser line summarizing the post. Replace with the real excerpt.",
  },
  {
    slug: "another-post-about-a-method",
    date: "May 2025",
    cat: "ML",
    title: "Another Post About a Method or Idea",
    excerpt:
      "One sentence describing what the reader will learn from this article.",
  },
  {
    slug: "short-note-or-tutorial",
    date: "Feb 2025",
    cat: "Notes",
    title: "Short Note or Tutorial Placeholder",
    excerpt: "Teaser text placeholder for the third blog entry in the list.",
  },
];

export const getProject = (slug: string) => projects.find((p) => p.slug === slug);
export const getPost = (slug: string) => posts.find((p) => p.slug === slug);

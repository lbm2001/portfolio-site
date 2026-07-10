import { describe, expect, it } from "vitest";
import {
  getPost,
  getProject,
  posts,
  projects,
  resumeDownloadName,
} from "../../lib/content";
import resumeData from "../../lib/resume-data.json";

// The committed lib/*-data.json files are what the deployed pages actually
// render (nothing is fetched at request time). A bad regeneration — empty
// output, duplicate slugs, missing fields — would ship broken pages while
// `next build` still succeeds, so pin the invariants here.

const SLUG = /^[a-z0-9]+(?:[-_.][a-z0-9]+)*$/i;

describe("projects-data.json", () => {
  it("has at least one project with valid, unique slugs", () => {
    expect(projects.length).toBeGreaterThan(0);
    const slugs = projects.map((p) => p.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for (const s of slugs) expect(s).toMatch(SLUG);
  });

  it("every project has the fields the cards render", () => {
    for (const p of projects) {
      expect(p.title.length).toBeGreaterThan(0);
      expect(Array.isArray(p.tags)).toBe(true);
      for (const l of p.links) {
        expect(l.href).toMatch(/^https?:\/\//);
        expect(l.label.length).toBeGreaterThan(0);
      }
    }
  });

  it("still contains the mini-vla project the hero links to", () => {
    // Hero.tsx hard-links to /projects/mini-vla; losing that project 404s the link.
    expect(getProject("mini-vla")).toBeDefined();
  });
});

describe("posts-data.json", () => {
  it("has valid, unique slugs and list fields", () => {
    const slugs = posts.map((p) => p.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for (const p of posts) {
      expect(p.slug).toMatch(SLUG);
      expect(p.title.length).toBeGreaterThan(0);
      expect(p.date.length).toBeGreaterThan(0);
    }
  });

  it("getPost round-trips every slug", () => {
    for (const p of posts) expect(getPost(p.slug)).toBe(p);
  });
});

describe("resume-data.json", () => {
  it("matches the shape /resume renders", () => {
    expect(resumeData.name.length).toBeGreaterThan(0);
    expect(resumeData.contacts.length).toBeGreaterThan(0);
    expect(resumeData.sections.length).toBeGreaterThan(0);
    for (const c of resumeData.contacts) {
      expect(c.href).toMatch(/^(https?:|mailto:)/);
    }
  });
});

describe("resumeDownloadName", () => {
  it("stamps the current month/year", () => {
    const d = new Date();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    expect(resumeDownloadName()).toBe(
      `resume-lukas-mueller-${mm}${d.getFullYear()}.pdf`,
    );
  });
});

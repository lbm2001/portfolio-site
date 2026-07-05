#!/usr/bin/env node
// Parses public/resume.tex -> lib/resume-data.json at BUILD time so the /resume
// page can import the parsed data as a static module. This keeps ALL LaTeX parsing
// at build time and out of the request path: Cloudflare Workers has no filesystem,
// so reading the .tex during render (as the page used to) throws at runtime and
// yields "Internal Server Error" whenever OpenNext re-invokes the render.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseResume } from "../lib/resume.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tex = fs.readFileSync(path.join(root, "public", "resume.tex"), "utf8");
const resume = parseResume(tex);
fs.writeFileSync(path.join(root, "lib", "resume-data.json"), JSON.stringify(resume, null, 2) + "\n");
console.log("gen-resume-data: wrote lib/resume-data.json");

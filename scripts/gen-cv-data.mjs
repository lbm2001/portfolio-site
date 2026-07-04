#!/usr/bin/env node
// Parses public/resume.tex -> lib/cv-data.json at BUILD time so the /resume page
// can import the parsed data as a static module. This keeps ALL LaTeX parsing at
// build time and out of the request path: Cloudflare Workers has no filesystem,
// so reading the .tex during render (as the page used to) throws at runtime and
// yields "Internal Server Error" whenever OpenNext re-invokes the render.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseCv } from "../lib/cv.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tex = fs.readFileSync(path.join(root, "public", "resume.tex"), "utf8");
const cv = parseCv(tex);
fs.writeFileSync(path.join(root, "lib", "cv-data.json"), JSON.stringify(cv, null, 2) + "\n");
console.log("gen-cv-data: wrote lib/cv-data.json");

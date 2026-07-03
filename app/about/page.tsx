import Image from "next/image";
import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { profile } from "@/lib/content";

export const metadata: Metadata = {
  title: `About · ${profile.name}`,
};

export default function AboutPage() {
  return (
    <main>
      <Nav />
      <article className="page">
        <div className="page-head">
          <div className="label-mono">About</div>
          <h1 className="page-title">{profile.name}</h1>
        </div>

        <div className="about-head">
          <div className="about-photo about-photo-lg">
            <Image src="/photo.jpg" alt={profile.name} fill sizes="160px" />
          </div>
          <p className="section-lead">{profile.lead}</p>
        </div>

        <div className="prose">
          <p>
            Placeholder biography. Write a few paragraphs here about your
            background, what you work on, and the ideas that connect your
            projects. This copy is dummy text and can be replaced later.
          </p>
          <p>
            A second paragraph can go deeper — the problems you find most
            interesting, the methods you reach for, and where you want to take
            the work next.
          </p>
        </div>

        <div className="info-grid">
          <div className="info-cell">
            <div className="k">Focus</div>
            <div className="v">{profile.focus}</div>
          </div>
          <div className="info-cell">
            <div className="k">Field</div>
            <div className="v">{profile.field}</div>
          </div>
          <div className="info-cell">
            <div className="k">Location</div>
            <div className="v">{profile.location}</div>
          </div>
          <div className="info-cell">
            <div className="k">Email</div>
            <div className="v">
              <a href={profile.links.email}>{profile.email}</a>
            </div>
          </div>
        </div>
      </article>
      <Footer />
    </main>
  );
}

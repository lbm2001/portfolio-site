import Image from "next/image";
import Link from "next/link";
import { profile } from "@/lib/content";

// Home preview: short about block with a link to the full /about page.
export default function About() {
  return (
    <section id="about" className="section">
      <div className="section-head">
        <div className="label-mono">01 About</div>
        <Link href="/about">More →</Link>
      </div>
      <div className="about-head">
        <div className="about-photo">
          <Image src="/photo.jpg" alt={profile.name} fill sizes="112px" />
        </div>
        <p className="section-lead">{profile.lead}</p>
      </div>
      <div className="info-grid">
        <div className="info-cell">
          <div className="k">Field</div>
          <div className="v">{profile.field}</div>
        </div>
        <div className="info-cell">
          <div className="k">Focus</div>
          <div className="v">{profile.focus}</div>
        </div>
        <div className="info-cell">
          <div className="k">Location</div>
          <div className="v">{profile.location}</div>
        </div>
        <div className="info-cell">
          <div className="k">Email</div>
          <div className="v">{profile.email}</div>
        </div>
      </div>
    </section>
  );
}

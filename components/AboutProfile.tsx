import Image from "next/image";
import { profile } from "@/lib/content";

// Shared profile blocks used by both the home preview (components/About.tsx) and
// the full /about page (app/about/page.tsx), which rendered identical markup.

/** Portrait photo beside the multi-line lead paragraph. */
export function AboutHead() {
  return (
    <div className="about-head">
      <div className="about-photo about-photo-lg">
        <Image src="/photo.jpg" alt={profile.name} fill sizes="160px" />
      </div>
      <p className="section-lead text-pretty">
        {profile.lead.map((line, index) => (
          <span key={index}>
            {line}
            {index < profile.lead.length - 1 && <br />}
          </span>
        ))}
      </p>
    </div>
  );
}

/** Field / Focus / Location / Email fact grid. */
export function ProfileInfoGrid() {
  return (
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
        <div className="v">
          <a href={profile.links.email}>{profile.email}</a>
        </div>
      </div>
    </div>
  );
}

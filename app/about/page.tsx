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
          <p className="section-lead text-pretty">
            {profile.lead.map((line, index) => (
              <span key={index}>
                {line}
                {index < profile.lead.length - 1 && <br />}
              </span>
            ))}
          </p>
        </div>

        <div className="prose">
          <p>
            My first robot was a Lego Mindstorms EV3 I got when I was 13, and I still remember
            spending many hours getting it to drive toward the brightest spot in
            my room. That interest went quiet for a while. I started out studying
            Digital Business Management, until a data science and machine
            learning course during Covid pulled me back in. I realized I am more interested
            in mathematics and programming than business, and wanted a proper
            foundation to build on. Therefore, I started over with a second bachelor in
            computer science at TU Darmstadt.
          </p>
          <p>
            Since then I've focused on machine learning, both in my studies and
            as a working student, most recently at Compredict, building ML and
            data engineering tools, and before that at BioNTech and Fresenius. I
            think that "embodied AI" (robots) is the next big step after the current LLM wave.
            by bringing this kind of learning into the physical world. Therefore, that's the direction I want
            to keep pushing, currently by exploring Imitation Learning and VLAs. 
            Away from the screen you'll usually find me in the gym or on my bike.
          </p>
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

import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { AboutHead, ProfileInfoGrid } from "@/components/AboutProfile";
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

        <AboutHead />

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
            Since then I&apos;ve focused on machine learning, both in my studies and
            as a working student, most recently at Compredict, building ML and
            data engineering tools, and before that at BioNTech and Fresenius. I
            think that &quot;embodied AI&quot; (robots) is the next big step after the current LLM wave.
            Therefore, that&apos;s the direction I want
            to keep pushing, currently by exploring Imitation Learning and VLAs.
            Away from the screen you&apos;ll usually find me in the gym or on my bike.
          </p>
        </div>

        <ProfileInfoGrid />
      </article>
      <Footer />
    </main>
  );
}

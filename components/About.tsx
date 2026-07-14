import Link from "next/link";
import { AboutHead, ProfileInfoGrid } from "@/components/AboutProfile";

// Home preview: short about block with a link to the full /about page.
export default function About() {
  return (
    <section id="about" className="section">
      <div className="section-head">
        <div className="label-mono">01 About</div>
        <Link href="/about">More</Link>
      </div>
      <AboutHead />
      <ProfileInfoGrid />
    </section>
  );
}

import Link from "next/link";

// Home preview: résumé teaser linking to the full /resume page.
export default function ResumeStrip() {
  return (
    <section id="resume" className="resume">
      <div className="resume-left">
        <div>
          <div className="label-mono">02 Resume</div>
          <div className="resume-t">Education, Experience & Skills</div>
        </div>
      </div>
      <Link className="resume-btn" href="/resume">
        View Resume →
      </Link>
    </section>
  );
}

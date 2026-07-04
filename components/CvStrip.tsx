import Link from "next/link";

// Home preview: résumé teaser linking to the full /resume page.
export default function CvStrip() {
  return (
    <section id="resume" className="cv">
      <div className="cv-left">
        <div>
          <div className="label-mono">02 Resume</div>
          <div className="cv-t">Education, Experience & Skills</div>
        </div>
      </div>
      <Link className="cv-btn" href="/resume">
        View Resume →
      </Link>
    </section>
  );
}

import Link from "next/link";

// Home preview: CV teaser linking to the full /cv page.
export default function CvStrip() {
  return (
    <section id="cv" className="cv">
      <div className="cv-left">
        <span className="cv-num">04</span>
        <div>
          <div className="cv-k">Curriculum Vitae</div>
          <div className="cv-t">Education, experience & skills</div>
        </div>
      </div>
      <Link className="cv-btn" href="/cv">
        View CV →
      </Link>
    </section>
  );
}

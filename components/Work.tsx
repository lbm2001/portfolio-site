import { projects } from "@/lib/content";

export default function Work() {
  return (
    <section id="work" className="section">
      <div className="section-head">
        <div className="label-mono">02 — Selected Work</div>
        <a href="#">All projects →</a>
      </div>
      <div className="work-grid">
        {projects.map((p) => (
          <a key={p.idx} className="work-card" href="#">
            <span className="work-meta">
              <span className="work-idx">{p.idx}</span>
              <span className="work-venue">{p.venue}</span>
            </span>
            <span className="work-title">{p.title}</span>
            <span className="work-blurb">{p.blurb}</span>
            <span className="work-tags">
              {p.tags.map((t) => (
                <span key={t} className="tag">
                  {t}
                </span>
              ))}
            </span>
            <span className="work-links">
              {p.links.map((lk) => (
                <span key={lk.label} className="work-link">
                  {lk.label}
                </span>
              ))}
            </span>
          </a>
        ))}
      </div>
    </section>
  );
}

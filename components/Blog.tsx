import { posts } from "@/lib/content";

export default function Blog() {
  return (
    <section id="blog" className="section">
      <div className="label-mono" style={{ marginBottom: 24 }}>
        03 — From the Blog
      </div>
      {posts.map((b) => (
        <a key={b.title} className="post" href={b.href}>
          <span className="post-date">{b.date}</span>
          <span className="post-cat">{b.cat}</span>
          <span className="post-body">
            <span className="post-title">{b.title}</span>
            <span className="post-excerpt">{b.excerpt}</span>
          </span>
        </a>
      ))}
    </section>
  );
}

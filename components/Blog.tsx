import Link from "next/link";
import { posts } from "@/lib/content";

// Home preview: recent posts, each linking to its article page.
export default function Blog() {
  return (
    <section id="blog" className="section">
      <div className="section-head">
        <div className="label-mono">03 — From the Blog</div>
        <Link href="/blog">All posts →</Link>
      </div>
      {posts.map((b) => (
        <Link key={b.slug} className="post" href={`/blog/${b.slug}`}>
          <span className="post-date">{b.date}</span>
          <span className="post-cat">{b.cat}</span>
          <span className="post-body">
            <span className="post-title">{b.title}</span>
            <span className="post-excerpt">{b.excerpt}</span>
          </span>
        </Link>
      ))}
    </section>
  );
}

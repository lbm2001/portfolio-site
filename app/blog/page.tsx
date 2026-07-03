import Link from "next/link";
import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { posts, profile } from "@/lib/content";

export const metadata: Metadata = {
  title: `Blog · ${profile.name}`,
};

export default function BlogPage() {
  return (
    <main>
      <Nav />
      <article className="page">
        <div className="page-head">
          <div className="label-mono">Blog</div>
          <h1 className="page-title">Writing & Notes</h1>
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
      </article>
      <Footer />
    </main>
  );
}

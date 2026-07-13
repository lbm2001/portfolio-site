import Link from "next/link";
import { posts } from "@/lib/content";
import PostRow from "./PostRow";

// Home preview: recent posts, each linking to its article page.
export default function Blog() {
  return (
    <section id="blog" className="section">
      <div className="section-head">
        <div className="label-mono">04 Blog</div>
        <Link href="/blog">See all</Link>
      </div>
      {posts.length > 0 ? (
        posts.map((b) => <PostRow key={b.slug} post={b} />)
      ) : (
        <p className="empty-note">Writing Coming Soon</p>
      )}
    </section>
  );
}

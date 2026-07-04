import Link from "next/link";
import type { Post } from "@/lib/content";

// A single blog post row, linking to its article page.
// Shared by the home preview (Blog) and the blog index page.
export default function PostRow({ post: b }: { post: Post }) {
  return (
    <Link className="post" href={`/blog/${b.slug}`}>
      <span className="post-date">{b.date}</span>
      <span className="post-cat">{b.cat}</span>
      <span className="post-body">
        <span className="post-title">{b.title}</span>
        <span className="post-excerpt">{b.excerpt}</span>
      </span>
    </Link>
  );
}

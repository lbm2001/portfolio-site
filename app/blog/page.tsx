import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import PostRow from "@/components/PostRow";
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
          <div className="label-mono">Writing & Notes</div>
          <h1 className="page-title">Blog</h1>
        </div>

        {posts.length > 0 ? (
          posts.map((b) => <PostRow key={b.slug} post={b} />)
        ) : (
          <p className="empty-note">Writing Coming Soon</p>
        )}
      </article>
      <Footer />
    </main>
  );
}

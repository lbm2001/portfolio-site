import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { posts, getPost, profile } from "@/lib/content";

export function generateStaticParams() {
  return posts.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);
  return { title: post ? `${post.title} · ${profile.name}` : "Post" };
}

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) notFound();

  return (
    <main>
      <Nav />
      <article className="page page-narrow">
        <Link className="back-link" href="/blog">
          ← All posts
        </Link>
        <div className="page-head">
          <div className="post-meta-row">
            <span className="post-cat">{post.cat}</span>
            <span className="post-date">{post.date}</span>
          </div>
          <h1 className="page-title">{post.title}</h1>
        </div>

        <div className="prose">
          <p className="section-lead">{post.excerpt}</p>
          <p>
            Placeholder article body. Write the full post here — this is dummy
            copy standing in for the real content, which can be dropped in
            later.
          </p>
        </div>
      </article>
      <Footer />
    </main>
  );
}

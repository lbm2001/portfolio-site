import { profile } from "@/lib/content";

export default function Footer() {
  return (
    <footer className="footer">
      {/* stamped at build time; each deploy refreshes it, so it never goes stale */}
      <span className="footer-copy">© {new Date().getFullYear()} {profile.name}</span>
      <div className="footer-links">
        <a href={profile.links.github}>GitHub</a>
        <a href={profile.links.linkedin}>LinkedIn</a>
        <a href={profile.links.email}>Email</a>
      </div>
    </footer>
  );
}

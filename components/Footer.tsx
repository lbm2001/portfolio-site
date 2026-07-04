import { profile } from "@/lib/content";

export default function Footer() {
  return (
    <footer className="footer">
      <span className="footer-copy">© 2026 {profile.name}</span>
      <div className="footer-links">
        <a href={profile.links.github}>GitHub</a>
        <a href={profile.links.linkedin}>LinkedIn</a>
        <a href={profile.links.email}>Email</a>
      </div>
    </footer>
  );
}

"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { nav, profile } from "@/lib/content";

// Shared site header, used on every page. Hides on scroll-down, reveals on
// scroll-up, and gains a hairline border once the page is scrolled.
export default function Nav() {
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let last = 0;
    const onScroll = () => {
      const y = window.scrollY;
      const el = navRef.current;
      if (el) {
        if (y > last && y > 90) el.style.transform = "translateY(-118%)";
        else el.style.transform = "translateY(0)";
        el.style.borderBottomColor = y > 8 ? "rgba(0,0,0,0.10)" : "rgba(0,0,0,0)";
      }
      last = y;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav className="nav" ref={navRef}>
      <Link className="nav-brand" href="/">
        <span className="nav-name">{profile.name}</span>
      </Link>
      <div className="nav-links">
        {nav.map((l) => (
          <Link key={l.label} href={l.href}>
            {l.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}

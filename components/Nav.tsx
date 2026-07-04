"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { nav, profile } from "@/lib/content";

// Shared site header, used on every page. Hides on scroll-down, reveals on
// scroll-up, and gains a hairline border once the page is scrolled. On phones
// (≤720px) the links collapse behind a hamburger toggle.
export default function Nav() {
  const navRef = useRef<HTMLElement>(null);
  const [open, setOpen] = useState(false);
  const openRef = useRef(false);
  openRef.current = open;

  useEffect(() => {
    let last = 0;
    const onScroll = () => {
      const y = window.scrollY;
      const el = navRef.current;
      if (el) {
        // keep the bar (and the open menu) put while the menu is open
        if (!openRef.current && y > last && y > 90) el.style.transform = "translateY(-118%)";
        else el.style.transform = "translateY(0)";
        el.style.borderBottomColor = y > 8 ? "rgba(0,0,0,0.10)" : "rgba(0,0,0,0)";
      }
      last = y;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // while the mobile menu is open, close it on Escape or a tap outside the nav
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: PointerEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <nav className={`nav${open ? " nav-open" : ""}`} ref={navRef}>
      <Link className="nav-brand" href="/" onClick={() => setOpen(false)}>
        <span className="nav-name">{profile.name}</span>
      </Link>

      <button
        type="button"
        className="nav-burger"
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        aria-controls="nav-menu"
        onClick={() => setOpen((v) => !v)}
      >
        <span />
        <span />
      </button>

      <div className="nav-links" id="nav-menu">
        {nav.map((l) => (
          <Link key={l.label} href={l.href} onClick={() => setOpen(false)}>
            {l.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}

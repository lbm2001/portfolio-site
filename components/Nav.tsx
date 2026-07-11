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
  // Mirrored for the mount-once scroll listener below, which must not be torn
  // down and re-added every time the menu toggles. Written on commit rather
  // than during render (writing a ref while rendering is unsafe under
  // concurrent rendering); the listener only ever fires after a commit, so it
  // never observes a stale value.
  const openRef = useRef(false);
  useEffect(() => {
    openRef.current = open;
  }, [open]);

  useEffect(() => {
    let last = 0;
    const onScroll = () => {
      const y = window.scrollY;
      const el = navRef.current;
      if (el) {
        // keep the bar (and the open menu) put while the menu is open
        if (!openRef.current && y > last && y > 90) el.style.transform = "translateY(-118%)";
        else el.style.transform = "translateY(0)";
        // var(--border) so the scrolled hairline follows the active theme
        el.style.borderBottomColor = y > 8 ? "var(--border)" : "transparent";
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
        <ThemeToggle />
      </div>
    </nav>
  );
}

// Flips the theme to an explicit override. Reads the CURRENT resolved theme from
// the DOM at click time (the data-theme attribute if set, else the OS
// preference) and writes the opposite as a persisted override — so the first
// click always does the visible thing regardless of what the OS default was.
// Which icon shows is handled entirely in CSS (see .theme-toggle in
// globals.css), so this component holds no theme state and can't flash on load.
function ThemeToggle() {
  const toggle = () => {
    const root = document.documentElement;
    const current =
      root.getAttribute("data-theme") ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      // private mode / storage blocked — the in-session override still applies
    }
  };

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label="Toggle dark mode"
      title="Toggle dark mode"
    >
      {/* moon — shown in light mode (click → dark) */}
      <svg className="icon-moon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
      </svg>
      {/* sun — shown in dark mode (click → light) */}
      <svg
        className="icon-sun"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    </button>
  );
}

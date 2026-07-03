"use client";

import { useEffect, useRef } from "react";
import MiniDemo from "@/components/MiniDemo";
import MiniDemoNN from "@/components/MiniDemoNN";
import { nav, profile } from "@/lib/content";
import { WalkerEnv } from "@/lib/rl/env";
import { CartPoleEnv } from "@/lib/rl/cartpole";
import { PendulumEnv } from "@/lib/rl/pendulum";
import { ClassificationTask } from "@/lib/nn/classification";
import { CurveFitTask } from "@/lib/nn/regression";
import { DigitTask } from "@/lib/nn/digits";

// stable factory identities so the demo effects don't re-run on every render
const makeCartPole = () => new CartPoleEnv();
const makePendulum = () => new PendulumEnv();
const makeWalker = () => new WalkerEnv();

const makeClassification = () => new ClassificationTask();
const makeCurveFit = () => new CurveFitTask();
const makeDigit = () => new DigitTask();

export default function SiteHeader() {
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
    <>
      <nav className="nav" ref={navRef}>
        <a className="nav-brand" href="#">
          <span className="nav-logo">YN</span>
          <span className="nav-name">{profile.name}</span>
        </a>
        <div className="nav-links">
          {nav.map((l) => (
            <a key={l.label} href={l.href}>
              {l.label}
            </a>
          ))}
        </div>
      </nav>

      <header className="hero">
        {/* Six live learning demos ringed around the name — neural nets on the
            left arc, RL agents on the right. Drag them; click one to train. */}
        <MiniDemoNN make={makeClassification} corner="ul" />
        <MiniDemoNN make={makeCurveFit} corner="ll" />
        <MiniDemoNN make={makeDigit} corner="bot" />
        <MiniDemo make={makeCartPole} corner="top" />
        <MiniDemo make={makePendulum} corner="ur" />
        <MiniDemo make={makeWalker} corner="lr" displayDisturb={17} />

        <div className="hero-content">
          <div className="hero-tag">{profile.tagline}</div>
          <h1>{profile.name}</h1>
          <div className="hero-cta">
            <a className="btn-primary" href="#">
              Get in touch
            </a>
            <a className="btn-outline" href="#">
              Download CV
            </a>
          </div>
        </div>
      </header>
    </>
  );
}

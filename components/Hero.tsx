"use client";

import { useEffect, useRef, useState } from "react";
import MiniDemo from "@/components/MiniDemo";
import MiniDemoNN from "@/components/MiniDemoNN";
import { profile, resumeDownloadName } from "@/lib/content";
import { WalkerEnv } from "@/lib/rl/env";
import { CartPoleEnv } from "@/lib/rl/cartpole";
import { PendulumEnv } from "@/lib/rl/pendulum";
import { ClassificationTask } from "@/lib/nn/classification";
import { CurveFitTask } from "@/lib/nn/regression";
import { DigitTask } from "@/lib/nn/digits";
import type { EnvFactory } from "@/lib/rl/types";

// stable factory identities so the demo effects don't re-run on every render
const makeCartPole = () => new CartPoleEnv();
const makePendulum = () => new PendulumEnv();
const makeWalker = () => new WalkerEnv();

const makeClassification = () => new ClassificationTask();
const makeCurveFit = () => new CurveFitTask();
const makeDigit = () => new DigitTask();

// The mobile stand-in (shown below 1180px in place of the ring) is one of these,
// chosen at random on each page load. Both are RL/ARS envs, so they share the
// MiniDemo renderer and the .mini-mobile layout.
const MOBILE_DEMOS: {
  make: EnvFactory;
  showFall?: boolean;
  displayDisturb?: number;
}[] = [
  { make: makeCartPole, showFall: true },
  { make: makeWalker, displayDisturb: 17 },
];

export default function Hero() {
  const heroRef = useRef<HTMLElement>(null);
  const nameRef = useRef<HTMLHeadingElement>(null);
  const [demo, setDemo] = useState<(typeof MOBILE_DEMOS)[number] | null>(null);

  // Pick the mobile demo on the client only — a random pick during render would
  // differ between server and client and trip a hydration mismatch.
  useEffect(() => {
    setDemo(MOBILE_DEMOS[Math.floor(Math.random() * MOBILE_DEMOS.length)]);
  }, []);

  // Publish the rendered width of the name as --name-w so the mobile mini (and
  // the floor line drawn in it) can be exactly as wide as "Lukas Müller".
  useEffect(() => {
    const measure = () => {
      const w = nameRef.current?.getBoundingClientRect().width;
      if (w && heroRef.current) {
        heroRef.current.style.setProperty("--name-w", `${Math.round(w)}px`);
      }
    };
    measure();
    window.addEventListener("resize", measure);
    // web fonts can settle after first paint and change the measured width
    document.fonts?.ready.then(measure).catch(() => {});
    return () => window.removeEventListener("resize", measure);
  }, []);

  return (
    <header className="hero" ref={heroRef}>
      {/* Six live learning demos ringed around the name — neural nets on the
          left arc, RL agents on the right. Drag them; click one to train. */}
      <MiniDemoNN make={makeClassification} corner="ul" />
      <MiniDemoNN make={makeCurveFit} corner="ll" />
      <MiniDemoNN make={makeDigit} corner="bot" />
      <MiniDemo make={makeCartPole} corner="top" showFall />
      <MiniDemo make={makePendulum} corner="ur" />
      <MiniDemo make={makeWalker} corner="lr" displayDisturb={17} />

      {/* The floating ring is hidden below 1180px; on phones a single live RL
          demo — cartpole or walker, chosen at random on load — perches on top of
          the name, which acts as the ground it stands on (see .mini-mobile). */}
      {demo && (
        <MiniDemo
          make={demo.make}
          corner="top"
          variant="static"
          showFall={demo.showFall}
          displayDisturb={demo.displayDisturb}
        />
      )}

      <div className="hero-content">
        <h1 ref={nameRef}>{profile.name}</h1>
        <div className="hero-tag">{profile.field}</div>
        <div className="hero-cta">
          <a className="btn-primary" href={profile.links.email}>
            Contact
          </a>
          <a className="btn-outline" href="/resume.pdf" download={resumeDownloadName()}>
            Download Resume
          </a>
        </div>
      </div>
    </header>
  );
}

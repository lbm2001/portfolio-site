"use client";

import MiniDemo from "@/components/MiniDemo";
import MiniDemoNN from "@/components/MiniDemoNN";
import { profile, resumeDownloadName } from "@/lib/content";
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

export default function Hero() {
  return (
    <header className="hero">
      {/* Six live learning demos ringed around the name — neural nets on the
          left arc, RL agents on the right. Drag them; click one to train. */}
      <MiniDemoNN make={makeClassification} corner="ul" />
      <MiniDemoNN make={makeCurveFit} corner="ll" />
      <MiniDemoNN make={makeDigit} corner="bot" />
      <MiniDemo make={makeCartPole} corner="top" showFall />
      <MiniDemo make={makePendulum} corner="ur" />
      <MiniDemo make={makeWalker} corner="lr" displayDisturb={17} />

      <div className="hero-content">
        <h1>{profile.name}</h1>
        <div className="hero-tag">{profile.field}</div>
        <div className="hero-cta">
          <a className="btn-primary" href={profile.links.email}>
            Contact
          </a>
          <a className="btn-outline" href="/resume.pdf" download={resumeDownloadName()}>
            Download Resume
          </a>
        </div>

        {/* The floating ring is hidden below 1180px; on phones a single live
            cartpole demo stands in for it, in normal flow under the name. */}
        <MiniDemo make={makeCartPole} corner="top" variant="static" showFall />
      </div>
    </header>
  );
}

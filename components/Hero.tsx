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
      </div>

      {/* Six live learning demos — neural nets + RL agents. On desktop this
          wrapper is display:contents and the panels float on a ring around the
          name (drag them; click to train). On touch / narrow screens the wrapper
          becomes a grid and the panels sit as a tidy, tappable gallery below the
          name (see .mini-grid). */}
      <div className="mini-grid">
        <MiniDemoNN make={makeClassification} corner="ul" />
        <MiniDemoNN make={makeCurveFit} corner="ll" />
        <MiniDemoNN make={makeDigit} corner="bot" />
        <MiniDemo make={makeCartPole} corner="top" showFall />
        <MiniDemo make={makePendulum} corner="ur" />
        <MiniDemo make={makeWalker} corner="lr" displayDisturb={17} />
      </div>
    </header>
  );
}

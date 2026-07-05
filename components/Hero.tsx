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
import type { Corner } from "@/components/useFloatDrag";
import type { EnvFactory } from "@/lib/rl/types";
import type { NNTaskFactory } from "@/lib/nn/types";

// stable factory identities so the demo effects don't re-run on every render
const makeCartPole = () => new CartPoleEnv();
const makePendulum = () => new PendulumEnv();
const makeWalker = () => new WalkerEnv();

const makeClassification = () => new ClassificationTask();
const makeCurveFit = () => new CurveFitTask();
const makeDigit = () => new DigitTask();

// Order matters: on the small-mobile 4-demo layout, whichever two of these three
// survive the random 1-of-3 drop keep this array order when filling the two
// small-mobile "NN" slots (see SMALL_NN_SLOTS / .mini-sm-nn* in globals.css).
const NN_DEMOS: { corner: Corner; make: NNTaskFactory }[] = [
  { corner: "ul", make: makeClassification },
  { corner: "ll", make: makeCurveFit },
  { corner: "bot", make: makeDigit },
];
const RL_DEMOS: {
  corner: Corner;
  make: EnvFactory;
  showFall?: boolean;
  displayDisturb?: number;
}[] = [
  { corner: "top", make: makeCartPole, showFall: true },
  { corner: "ur", make: makePendulum },
  { corner: "lr", make: makeWalker, displayDisturb: 17 },
];

const SMALL_NN_SLOTS = ["nn1", "nn2"] as const;
const SMALL_RL_SLOTS = ["rl1", "rl2"] as const;

// Assigns each item a small-mobile slot name (skipping the dropped one), while
// leaving every item mounted — CSS alone decides *whether* the drop/slot classes
// have any effect, based on the current viewport (see globals.css), so rotating
// a tablet or resizing a window updates the layout without any JS involved.
function withSmallSlots<T>(items: T[], dropIndex: number | null, slots: readonly string[]) {
  let slot = 0;
  return items.map((item, i) => {
    const dropSmall = dropIndex === i;
    const smallSlot = dropSmall || dropIndex === null ? undefined : slots[slot++];
    return { item, dropSmall, smallSlot };
  });
}

export default function Hero() {
  const heroRef = useRef<HTMLElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  // Which one of each trio to drop on the small-mobile (4-demo) layout — decided
  // client-side (random) so SSR stays deterministic (no hydration mismatch) and
  // the desktop/tablet 6-demo ring is unaffected either way.
  const [dropNN, setDropNN] = useState<number | null>(null);
  const [dropRL, setDropRL] = useState<number | null>(null);

  useEffect(() => {
    setDropNN(Math.floor(Math.random() * NN_DEMOS.length));
    setDropRL(Math.floor(Math.random() * RL_DEMOS.length));
  }, []);

  // Publish the rendered half-height of the name/tag/CTA block as
  // --content-half-h, so the mobile/tablet ring (globals.css) can place its
  // seats a safe, measured distance above/below it — no collision regardless of
  // how the title wraps on a given device or font. Only height matters: seats
  // stack in bands above/below the content and are horizontally self-centred, so
  // they never depend on (and can't be clipped by) the content's width.
  useEffect(() => {
    const measure = () => {
      const r = contentRef.current?.getBoundingClientRect();
      if (r && heroRef.current) {
        heroRef.current.style.setProperty("--content-half-h", `${Math.round(r.height / 2)}px`);
      }
    };
    measure();
    window.addEventListener("resize", measure);
    // web fonts can settle after first paint and change the measured height
    document.fonts?.ready.then(measure).catch(() => {});
    return () => window.removeEventListener("resize", measure);
  }, []);

  const nnSlotted = withSmallSlots(NN_DEMOS, dropNN, SMALL_NN_SLOTS);
  const rlSlotted = withSmallSlots(RL_DEMOS, dropRL, SMALL_RL_SLOTS);

  return (
    <header className="hero" ref={heroRef}>
      {/* Six live learning demos ringed around the name — neural nets on the left
          arc, RL agents on the right. Drag them; click/tap to train. Below
          1180px (or on any device without real hover, e.g. a tablet) the ring
          scales down and re-forms as two bands (RL above the name, NN below)
          instead of disappearing. Below 600px width it further collapses to 4
          demos — one random NN and one random RL dropped — with the survivors
          shown larger. See the matching rules in globals.css. */}
      {rlSlotted.map(({ item, dropSmall, smallSlot }) => (
        <MiniDemo
          key={item.corner}
          make={item.make}
          corner={item.corner}
          showFall={item.showFall}
          displayDisturb={item.displayDisturb}
          dropSmall={dropSmall}
          smallSlot={smallSlot}
        />
      ))}
      {nnSlotted.map(({ item, dropSmall, smallSlot }) => (
        <MiniDemoNN
          key={item.corner}
          make={item.make}
          corner={item.corner}
          dropSmall={dropSmall}
          smallSlot={smallSlot}
        />
      ))}

      <div className="hero-content" ref={contentRef}>
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
    </header>
  );
}

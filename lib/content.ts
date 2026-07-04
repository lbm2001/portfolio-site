export interface NavLink {
  label: string;
  href: string;
}

export interface ProjectLink {
  label: string;
  href: string;
}

export interface ProjectFigure {
  id: string;
  src: string;
  alt: string;
  caption: string;
  width: number;
  height: number;
}

export interface Project {
  slug: string;
  idx: string;
  title: string;
  venue: string;
  blurb: string;
  tags: string[];
  links: ProjectLink[];
  /**
   * Full write-up shown on the detail page. Blank lines separate paragraphs.
   * Supports inline `$..$` and block `$$..$$` math (KaTeX), and a block
   * `{{figure:some-id}}` marker referencing an entry in `figures`.
   */
  body?: string;
  figures?: ProjectFigure[];
  /** Shows the AI-assisted-writing disclaimer on the detail page when true. */
  aiAssisted?: boolean;
}

export interface Post {
  slug: string;
  date: string;
  cat: string;
  title: string;
  excerpt: string;
  /** Full article body. Omit until the post is actually written. */
  body?: string;
  /** Shows the AI-assisted-writing disclaimer on the detail page when true. */
  aiAssisted?: boolean;
}

export const profile = {
  name: "Lukas Müller",
  lead: "I'm a computer science student at TU Darmstadt, currently writing my bachelor thesis on robot learning. I want to get machines to learn how to act, right now through imitation learning.",
  focus: "Imitation Learning",
  field: "Machine Learning & Robot Learning",
  location: "Frankfurt, Germany",
  email: "contact@lukasmueller.dev",
  links: {
    github: "https://github.com/lbm2001",
    linkedin: "https://www.linkedin.com/in/lukas-m-695b06195/",
    email: "mailto:contact@lukasmueller.dev",
  },
};

export const nav: NavLink[] = [
  { label: "About", href: "/about" },
  { label: "Resume", href: "/resume" },
  { label: "Projects", href: "/projects" },
  { label: "Blog", href: "/blog" },
];

export const projects: Project[] = [
  {
    slug: "ev-charging-mdp",
    idx: "01",
    title: "Online vs. Optimal Electrical Vehicle Charging",
    venue: "Aarhus University",
    blurb:
      "How close can a simple, computation-free charging rule get to the cost-optimal policy for an EV facing uncertain trips and fluctuating electricity prices?",
    tags: ["Markov Decision Processes", "Dynamic Programming", "EV Charging"],
    links: [{ label: "Code", href: "https://github.com/lbm2001/au-mdt" }],
    aiAssisted: true,
    figures: [
      {
        id: "heatmaps",
        src: "/projects/ev-charging-mdp/policy-heatmaps.png",
        alt: "Heatmaps of charging rate by hour of day and battery level, for the optimal policy and seven heuristics",
        caption:
          "Charging rate by hour and battery level. The optimal policy and the Departure Urgency heuristic (top row) share the same shape — charging swells ahead of each commute peak and the charge-or-wait line rises as departure nears. The simpler rules below only ever capture one side of that pattern.",
        width: 4200,
        height: 3360,
      },
      {
        id: "cost",
        src: "/projects/ev-charging-mdp/baseline-cost.png",
        alt: "Bar chart of expected total cost for the optimal policy and seven heuristic policies, log scale",
        caption:
          "Expected total cost per policy (log scale), split into charging cost (solid) and unserved-demand penalty (hatched). The urgency rules sit closest to the optimum; rules that under-charge are dominated by the penalty.",
        width: 4200,
        height: 1380,
      },
    ],
    body: `Electric vehicles are a fast-growing load on the grid, and how a single car is charged matters both for the driver's bill and for the system as a whole. The problem is a trade-off under uncertainty: the battery should be full whenever the driver needs it, but recharging is slow and electricity prices fluctuate over the day, so the car cannot simply be topped up right before departure. A good charging policy has to balance vehicle availability against energy cost when both the next trip and the price are unknown in advance.

I formulated the single-vehicle charging problem as a finite-horizon Markov Decision Process and solved it exactly by backward induction to obtain the cost-optimal policy, minimizing expected future cost $C_t$ at every state $s$ and time $t$ over the charging action $u$:

$$J_t(s) = \\min_{u \\in \\mathcal{U}} \\left[ C_t(s,u) + \\sum_{s'} P_t(s' \\mid s, u)\\, J_{t+1}(s') \\right]$$

That policy is not something you would run online, since it needs the full model and a heavy offline computation, so I used it purely as a reference. Against it I benchmarked seven heuristic rules that require no offline computation, computing the exact expected cost of each by policy evaluation on the same MDP.

{{figure:heatmaps}}

The central finding is that a computation-free rule can come close to the optimum, but only if it reasons over the right signals: the electricity price, the battery state of charge, and the time to the next departure. My best heuristic, Departure Urgency, folds all three into a single ratio comparing the energy still needed against the time left to the next expected departure $\\tau(m)$:

$$\\rho_t = \\min\\left\\{1, \\frac{e^{\\text{tgt}} - e_t}{u_{\\max}\\,\\eta_c\\,\\omega\\,\\tau(m)}\\right\\}$$

It charges whenever the price is cheap relative to $\\rho_t$, and reaches the optimum to within about 67% in the baseline while reproducing its structure: front-loading charge ahead of likely trips and treating price only as a secondary signal. This is essentially a systematic version of "charge before trips, prefer cheap hours". Rules that track only one side of the trade-off, or follow a fixed schedule, are worse by one to several orders of magnitude.

{{figure:cost}}

To test how far this holds, I added a negative-binomial trip-duration model for more realistic trips and a set of electricity-price models estimated from historical ENTSO-E data, then ran a sensitivity study across 28 configurations. The ranking is robust: Departure Urgency is the best heuristic in 22 of them, and the parameters that matter most are the penalty for unserved demand, the trip-duration model, and the departure profile. The takeaway is that for this model, a cheap online rule reasoning over price, charge, and time to the next departure captures much of what the exact optimal policy achieves.`,
  },
];

export const posts: Post[] = [
  // No posts written yet. Add entries here as you write them, e.g.:
  // {
  //   slug: "my-first-post",
  //   date: "Jul 2026",
  //   cat: "ML",
  //   title: "My First Post",
  //   excerpt: "One-line teaser shown in the list and atop the article.",
  //   body: `First paragraph.\n\nSecond paragraph — blank lines separate paragraphs.`,
  // },
];

export const getProject = (slug: string) => projects.find((p) => p.slug === slug);
export const getPost = (slug: string) => posts.find((p) => p.slug === slug);

// Save-as filename for the resume PDF download, stamped with the current
// month/year (e.g. resume_lukas_mueller_07_2026.pdf). The served asset stays
// /resume.pdf — the browser's `download` attribute renames it on save.
export function resumeDownloadName() {
  const d = new Date();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  return `resume_lukas_mueller_${month}_${d.getFullYear()}.pdf`;
}

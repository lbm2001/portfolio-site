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
  title: string;
  venue: string;
  /** Working period, shown in red in the meta line (e.g. "Apr–Sep 2025").
   *  Projects are listed newest→oldest by start date. */
  period?: string;
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
  lead: "I am studying computer science, currently writing my bachelor thesis on robot learning. I want to get machines that learn to act, right now through imitation learning.",
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
    title: "Online vs. Optimal Electrical Vehicle Charging",
    venue: "Aarhus University",
    period: "Jan–Jun 2026",
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
  {
    slug: "mujopy",
    title: "MuJoPy",
    venue: "Open Source",
    period: "Oct 2025",
    blurb:
      "A small Python package that gives pythonic, graph-structured access to MuJoCo models — the preprocessing layer from the robot-generation project, factored out into a standalone, pip-installable tool.",
    tags: ["MuJoCo", "Python", "Robotics Tooling"],
    links: [
      { label: "Code", href: "https://github.com/lbm2001/mujopy" },
      { label: "PyPI", href: "https://pypi.org/project/mujopy/" },
    ],
    aiAssisted: true,
    body: `MuJoPy spun out of the robot-generation project below. Working with MuJoCo models usually means writing custom parsing logic every time you need to reach the bodies, joints and geoms of a robot. MuJoPy wraps MuJoCo's low-level structs in dataclasses — Body, Joint, Geom, MuJoPyModel — that expose model fields as plain Python properties and derive a navigable graph of the robot, so downstream tasks like feature extraction no longer need bespoke parsing. Each wrapper still exposes the raw struct through a mujoco_view attribute for direct access when needed, and the property system is extensible: you can register your own read-only properties on the model.

On top of the wrapper, RobotGraph gives a concrete example of the abstraction the generation project relied on — it builds a NetworkX graph of the robot alongside a feature matrix driven by a feature-config file. This is exactly the layer that turned MuJoCo XML into the adjacency and feature matrices the VAE was trained on, pulled out into its own package so it can be reused on its own (pip install mujopy).`,
  },
  {
    slug: "large-scale-robot-generation",
    title: "Large Scale Robot Generation",
    venue: "Technical University of Darmstadt",
    period: "Apr–Sep 2025",
    blurb:
      "Can a variational autoencoder over robot graphs learn a continuous latent space of morphologies, so that thousands of diverse, physically plausible embodiments can be sampled directly from existing robot designs?",
    tags: [
      "Graph Neural Networks",
      "Variational Autoencoder",
      "Robot Morphology",
      "MuJoCo",
    ],
    links: [
      { label: "Code", href: "https://github.com/lbm2001/lascroge" },
      { label: "Paper", href: "/projects/large-scale-robot-generation/paper.pdf" },
    ],
    aiAssisted: true,
    body: `Cross-embodiment locomotion policies generalize better the more diverse the robots they are trained on, but the design space of embodiments grows exponentially with the number of components. Existing automated approaches are grammar-based: they assemble robots from a predefined library of parts, which requires prior knowledge to define that library, restricts designs to discrete combinations, and ignores the continuous physical parameters — mass, geometry, joint limits — of real systems. With Nurhak Yalcin, I built an end-to-end system that instead learns a generative model directly from existing robot designs: a variational autoencoder (VAE) implemented as a message-passing graph neural network (MPNN), so that novel, physically plausible embodiments can be sampled from a learned latent space.

A robot is abstracted as an acyclic, undirected graph $\\mathcal{G} = (V, E)$ — effectively a kinematic tree — whose nodes are body parts (links or joints) and whose edges are their connections, with the continuous attributes of each part stored in a feature matrix $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$. The encoder is an MPNN that runs $T$ steps of message passing, updating node messages and hidden states as

$$m_v^{t+1} = \\sum_{w \\in N(v)} M_t(h_v^t, h_w^t, e_{vw}), \\qquad h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

where in our case the learned functions $M_t$ and $U_t$ are realized by a GRU adapted to graphs. The graph representation $h_{\\mathcal{G}}$ — the sum over leaf-node states — is mapped by two linear layers to the mean $\\mu_{\\mathcal{G}}$ and variance $\\sigma_{\\mathcal{G}}^2$ of a latent distribution, from which we sample $z_{\\mathcal{G}} \\sim \\mathcal{N}(\\mu_{\\mathcal{G}}, \\sigma_{\\mathcal{G}}^2)$ using the reparametrization trick.

The decoder reconstructs the graph autoregressively from $z_{\\mathcal{G}}$, expanding depth-first from the root. At each step it decides whether to add a node with probability $p_t = \\sigma\\!\\left(u^c \\cdot \\tau(W_1^c x_{i_t} + W_2^c z_{\\mathcal{G}} + W_3^c \\sum_k h_{k,i_t})\\right)$ and — departing from classification-based methods that pick parts from a vocabulary — directly regresses the new node's continuous feature vector $f_j = U^l \\tau(W_1^l z_{\\mathcal{G}} + W_2^l h_{i,j})$ along with a link/joint type. Training minimizes the decoder's topology, feature-regression and type losses together with a KL term pulling the latent distribution toward a standard Gaussian prior:

$$\\mathcal{L}(\\mathcal{G}) = \\mathcal{L}_d(\\mathcal{G}) + \\beta\\, \\mathcal{L}_{KL}$$

We validated the system on synthetic graphs and on real robots from the MuJoCo Menagerie. Trained only on quadrupeds, the VAE reconstructed the unseen Unitree Go2 exactly; trained jointly on quadrupeds and humanoids it recovered a close but imperfect structure, with continuous features reconstructed well in both cases, and sampling a random latent vector yielded valid, novel graph structures. Scaling to 17 robots preserved the structure but left a higher feature-regression error that longer training should reduce. A postprocessing pipeline back to MuJoCo XML and validation via policy transfer are the main next steps — the goal being a task-agnostic generator of the diverse embodiments that morphology-agnostic policies need. This was joint work with Nurhak Yalcin for the Robot Learning: Integrated Project at TU Darmstadt.`,
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

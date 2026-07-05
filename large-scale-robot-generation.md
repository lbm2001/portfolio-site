---
title: Large Scale Robot Generation
venue: Technical University of Darmstadt
period: Apr–Sep 2025
blurb: Can a variational autoencoder over robot graphs learn a continuous latent space of morphologies, so that thousands of diverse, physically plausible embodiments can be sampled directly from existing robot designs?
aiAssisted: true
tags:
  - Graph Neural Networks
  - Variational Autoencoder
  - Robot Morphology
  - MuJoCo
links:
  - label: Paper
    href: paper.pdf
---

Cross-embodiment locomotion policies generalize better the more diverse the robots they are trained on, but the design space of embodiments grows exponentially with the number of components. Existing automated approaches are grammar-based: they assemble robots from a predefined library of parts, which requires prior knowledge to define that library, restricts designs to discrete combinations, and ignores the continuous physical parameters — mass, geometry, joint limits — of real systems. With Nurhak Yalcin, I built an end-to-end system that instead learns a generative model directly from existing robot designs: a variational autoencoder (VAE) implemented as a message-passing graph neural network (MPNN), so that novel, physically plausible embodiments can be sampled from a learned latent space.

A robot is abstracted as an acyclic, undirected graph $\mathcal{G} = (V, E)$ — effectively a kinematic tree — whose nodes are body parts (links or joints) and whose edges are their connections, with the continuous attributes of each part stored in a feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$. The encoder is an MPNN that runs $T$ steps of message passing, updating node messages and hidden states as

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}), \qquad h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

where in our case the learned functions $M_t$ and $U_t$ are realized by a GRU adapted to graphs. The graph representation $h_{\mathcal{G}}$ — the sum over leaf-node states — is mapped by two linear layers to the mean $\mu_{\mathcal{G}}$ and variance $\sigma_{\mathcal{G}}^2$ of a latent distribution, from which we sample $z_{\mathcal{G}} \sim \mathcal{N}(\mu_{\mathcal{G}}, \sigma_{\mathcal{G}}^2)$ using the reparametrization trick.

The decoder reconstructs the graph autoregressively from $z_{\mathcal{G}}$, expanding depth-first from the root. At each step it decides whether to add a node with probability $p_t = \sigma\!\left(u^c \cdot \tau(W_1^c x_{i_t} + W_2^c z_{\mathcal{G}} + W_3^c \sum_k h_{k,i_t})\right)$ and — departing from classification-based methods that pick parts from a vocabulary — directly regresses the new node's continuous feature vector $f_j = U^l \tau(W_1^l z_{\mathcal{G}} + W_2^l h_{i,j})$ along with a link/joint type. Training minimizes the decoder's topology, feature-regression and type losses together with a KL term pulling the latent distribution toward a standard Gaussian prior:

$$\mathcal{L}(\mathcal{G}) = \mathcal{L}_d(\mathcal{G}) + \beta\, \mathcal{L}_{KL}$$

We validated the system on synthetic graphs and on real robots from the MuJoCo Menagerie. Trained only on quadrupeds, the VAE reconstructed the unseen Unitree Go2 exactly; trained jointly on quadrupeds and humanoids it recovered a close but imperfect structure, with continuous features reconstructed well in both cases, and sampling a random latent vector yielded valid, novel graph structures. Scaling to 17 robots preserved the structure but left a higher feature-regression error that longer training should reduce. A postprocessing pipeline back to MuJoCo XML and validation via policy transfer are the main next steps — the goal being a task-agnostic generator of the diverse embodiments that morphology-agnostic policies need. This was joint work with Nurhak Yalcin for the Robot Learning: Integrated Project at TU Darmstadt.

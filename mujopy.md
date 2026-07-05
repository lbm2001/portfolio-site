---
title: MuJoPy
venue: Open Source
period: Oct 2025
blurb: A small Python package that gives pythonic, graph-structured access to MuJoCo models — the preprocessing layer from the robot-generation project, factored out into a standalone, pip-installable tool.
aiAssisted: true
tags:
  - MuJoCo
  - Python
  - Robotics Tooling
links:
  - label: PyPI
    href: https://pypi.org/project/mujopy/
---

MuJoPy spun out of the robot-generation project below. Working with MuJoCo models usually means writing custom parsing logic every time you need to reach the bodies, joints and geoms of a robot. MuJoPy wraps MuJoCo's low-level structs in dataclasses — Body, Joint, Geom, MuJoPyModel — that expose model fields as plain Python properties and derive a navigable graph of the robot, so downstream tasks like feature extraction no longer need bespoke parsing. Each wrapper still exposes the raw struct through a mujoco_view attribute for direct access when needed, and the property system is extensible: you can register your own read-only properties on the model.

On top of the wrapper, RobotGraph gives a concrete example of the abstraction the generation project relied on — it builds a NetworkX graph of the robot alongside a feature matrix driven by a feature-config file. This is exactly the layer that turned MuJoCo XML into the adjacency and feature matrices the VAE was trained on, pulled out into its own package so it can be reused on its own (pip install mujopy).

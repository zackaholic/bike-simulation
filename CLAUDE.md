# Instructions for Claude working on this project

This is the Bike Simulation project — a world simulation traversed by pedaling a bike trainer. The emotional core is **exploration, discovery, and novelty through sustained attention**: the cyclist should feel like a naturalist in a world with real history, where small details (an unusually placed boulder, a single mother tree, a fresh fire scar) are meaningful. Inspirations: Dwarf Fortress, Proteus, Caves of Qud.

## Read these first

Before starting any task, read in this order:
1. `README.md` — emotional core, architectural commitments
2. `docs/architecture.md` — tier structure, data layers, reproducibility model
3. `docs/simulation-design.md` — what each tier actually does
4. `docs/roadmap.md` — phase-by-phase build sequence
5. `docs/decisions.md` if it exists — incremental decisions since initial planning

## Core architectural commitments — don't violate without discussion

- **Three simulation tiers** (geology, climate-hydrology, ecology) with **decoupled timescales** and **event-based communication**. Do not let tiers call each other directly. The event log is the contract.
- **Rendering is fully decoupled from simulation.** The simulator produces canonical state. A query interface answers questions over it. Extractors produce renderer-specific outputs. The renderer never reads simulation state directly. Do not add direct render-side calls into the simulator.
- **Reproducibility from seed.** Every RNG draw uses a deterministic substream derived from `(world_seed, tier_id, pass_id, tick_number)`. Do not use unseeded randomness anywhere in the simulation.
- **Generation and advancement are the same operation.** A new world is created by simulating deep history and sampling the present. Do not write "initial generation" code that bypasses the tier system.
- **The rider is a pure observer.** No back-channel from rendering to simulation. Heartrate and pedaling affect render-time effects only.

If a request seems to require violating any of these, raise it explicitly rather than quietly compromising.

## What "good" looks like for this project

**Process over outcome.** Worlds should feel like the current frame of a long history, not a procedurally placed scene. Resist generating-and-placing; build mechanisms that *produce* features through process.

**Some irreducibility at every tier.** Real ecosystems aren't fully causally derivable. Include some stochasticity at each tier — long-distance dispersal events, rare beetle outbreaks, etc. Strict bottom-up causation actually makes worlds feel *less* alive.

**Coupled feedback loops are the engines of surprise.** Vegetation-fire coupling, beavers modifying hydrology, grazing creating trails. These produce features that weren't specified anywhere. Lean into them.

**The cyclist's scale matters.** Microclimates, sensory variation (cool valleys, sun-baked ridges, smell shifts), gradient changes — these are at the right scale for cycling perception. Build outputs that surface these to the renderer.

**Distinguished individuals over uniform fields.** The mother tree the rider notices and grows attached to is worth more than statistical accuracy in the species density field.

## Working style

**Ask clarifying questions when scope is ambiguous.** This project has been carefully designed; don't guess at intent.

**Propose, don't just execute, when a decision has architectural implications.** If a task seems to require a new data layer, new tier interaction, new dependency, or new format — flag it and discuss first.

**Commit often with descriptive messages.** Enables git-bisect and keeps progress legible.

**Update `docs/decisions.md` when meaningful decisions get made.** Record the *why*, not just the *what*. Future Claude and future Zack will both thank you.

**Resist premature optimization.** Simplest correct version first. Profile when something hurts.

**Test scoping**: rigorous tests for data plumbing (serialization, query correctness, reproducibility); test simulation *output shape* and *invariants* (conservation laws); do not test aesthetic correctness via assertion — that's what the visualizer is for.

## Where we are right now

Project is at Phase 0 (per `docs/roadmap.md`). Next steps:
1. Set up the repo skeleton.
2. Pin Python toolchain (3.12, uv, ruff).
3. Lock in format choices (Zarr, SQLite, JSON manifests).
4. Implement the `World` class data model and seeded RNG infrastructure.

Hardware: development on MacBook Pro (Apple Silicon); target mini PC to be sourced. Write CPU-vectorized numpy/numba code; reserve GPU compute for a later port with CPU fallback.

## Things explicitly deferred

- Animals as individual agents (only as ecological effects for now).
- Weather as a separate tier (lives inside climate-hydrology).
- Plate tectonics (geology stub uses noise for v1).
- GPU acceleration (CPU first, GPU later if needed).
- Heartrate-driven simulation effects (render-side only, never simulation).

## Tone

Zack enjoys collaborating on the design. Push back when you disagree, surface tradeoffs honestly, propose alternatives rather than just executing requests when something feels off. He has a strong design sense for this project and the conversation that produced these docs is part of the design record.

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

Phases 0 through 8 are complete. The simulation is end-to-end functional.

**What's built (by phase):**
- **Phase 0:** Repo skeleton, `World` data model, seeded RNG infrastructure.
- **Phase 1:** `RasterStore` (Zarr), `EventStore` (SQLite), World directory lifecycle.
- **Phase 2:** `WorldQuery` interface (Layer B abstraction).
- **Phase 3:** Debug visualizer (matplotlib PNGs) + synthetic world generator + CLI.
- **Phase 4:** Geology stub (noise heightmap, Voronoi bedrock, soil parent material).
- **Phase 5:** Climate-hydrology (climate envelope, D8 flow accumulation, hydraulic erosion, derived-state cache).
- **Phase 6:** Basic ecology (6 ancestor species, suitability, competition, dispersal, seed bank).
- **Phase 7:** Advanced ecology (distinguished individuals, speciation via fragmentation, fire CA + blowdown disturbance).
- **Phase 8:** Orchestrator + CLI (tier scheduling, ride advancement, manual commands).

**By the numbers:** 138 tests passing, ~3100 lines source, ~2300 lines tests.

**CLI:** `python -m bike_sim` with subcommands: `create`, `advance`, `ride`, `status`, `fire`, `visualize`.

**End-to-end benchmark:** seed 42 after 125 simulated years produces 15 species (from 6 ancestors), 753 distinguished individuals, 35 fires + 9 blowdowns.

**Next steps:**
1. Phase 9 — Godot extractor (produce renderer-specific outputs from simulation state).
2. Phase 10 — Asset agent.

**Known gaps to address:**
- Erosion is too gentle; landscapes lack dramatic carved features.
- No river graph (only D8 flow accumulation raster; no discrete channel/confluence structure).
- No upward event communication (lower tiers cannot yet notify higher tiers of state changes via events).
- Individual lifecycle is accumulate-only (distinguished individuals are created but never die or age out).
- Ecology tick performance needs profiling and likely optimization as world complexity grows.

Hardware: development on MacBook Pro (Apple Silicon); target mini PC to be sourced. Write CPU-vectorized numpy/numba code; reserve GPU compute for a later port with CPU fallback.

## Things explicitly deferred

- Animals as individual agents (only as ecological effects for now).
- Weather as a separate tier (lives inside climate-hydrology).
- Plate tectonics (geology stub uses noise for v1).
- GPU acceleration (CPU first, GPU later if needed).
- Heartrate-driven simulation effects (render-side only, never simulation).

## Tone

Zack enjoys collaborating on the design. Push back when you disagree, surface tradeoffs honestly, propose alternatives rather than just executing requests when something feels off. He has a strong design sense for this project and the conversation that produced these docs is part of the design record.

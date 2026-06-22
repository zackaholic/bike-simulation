# Instructions for Claude working on this project

This is the Bike Simulation project — a world simulation traversed by pedaling a bike trainer. The emotional core is **exploration, discovery, and novelty through sustained attention**: the cyclist should feel like a naturalist in a world with real history, where small details (an unusually placed boulder, a single mother tree, a fresh fire scar) are meaningful. Inspirations: Dwarf Fortress, Proteus, Caves of Qud.

## Read these first

Before starting any task, read in this order:
1. `README.md` — emotional core, architectural commitments
2. `docs/architecture.md` — tier structure, data layers, reproducibility model
3. `docs/simulation-design.md` — what each tier actually does
4. `docs/ecology-tick-refactor.md` — current ecology design (supersedes the older ecology in roadmap Phases 6–7)
5. `docs/roadmap.md` — phase-by-phase build sequence (original plan; ecology since revised)
6. `docs/decisions.md` if it exists — incremental decisions since initial planning

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

Phases 0–8 are complete (full infrastructure + all three tiers + orchestrator/CLI). The **ecology tier was then substantially rewritten** (June 2026) and is the current focus. Read `docs/ecology-tick-refactor.md` and the Ecology section of `docs/simulation-design.md` for the current design — the older "Phase 6/7" ecology is superseded.

**Infrastructure (stable):** Zarr `RasterStore` + SQLite `EventStore` + versioned `World`; `WorldQuery` (Layer B); debug2d + webview extractors (Layer C); geology stub; climate-hydrology (climate envelope, D8 flow accumulation, hydraulic erosion, derived-state cache); orchestrator + CLI.

**Ecology (current model):** single unified per-tick rule — logistic growth `density·growth_rate·(K_eff−load)/K`, fixed-rate mortality `density/(lifespan·4)`, dispersal every tick. 6-trait genome. Absolute suitability normalization with references fit to the world's climate envelope. `COMPETITION_BASELINE` (competition between dissimilar species) is what creates biome structure. Refugium floor lets species survive unfavorable phases. **No seed banks, no biotic pressure, no mast seeding.** Extinction, speciation, and distinguished-individual promotion exist but are **toggleable and currently deferred** (see refactor doc).

**By the numbers:** 333 tests, ~7700 lines source.

**CLI:** `python -m bike_sim` with subcommands: `create`, `advance`, `ride`, `status`, `fire`, `visualize`, `ride-experience`, `ride-compare`.

**Testing methodology (core dev workflow):** `scripts/test_ecology.py` — `equilibrium` (freeze slow climate drift, cycle 4 seasonal snapshots, run to a stable state) then `perturb` (temperature/precipitation/fire/species-removal) and verify the response matches an ecological prediction. Validated: equilibrium produces biome structure; a wet perturbation makes dry species retreat and wet/mid species rebound from refugia (biome migration).

**Next steps:**
1. More perturbation tests (temperature shift, fire, species removal).
2. Re-enable climate cycling and watch biome migration happen naturally over the cycle (the original goal — species ranges shifting in *phase*, not just amplitude).
3. Later: dedicated extinction/speciation design session; warm-wet genome redistribution (see known gaps); then Phase 9 (Godot extractor) and Phase 10 (asset agent).

**Known gaps / open threads:**
- Erosion is too gentle; landscapes lack dramatic carved features, and terrain is jagged/high (rideability work deferred to pre-render phase).
- No river graph (only D8 flow accumulation raster; no discrete channel/confluence structure).
- No upward event communication (lower tiers can't yet notify higher tiers via events).
- **Warm-wet species stranding — addressed (2026-06-21)**: this world's temp/precip are anticorrelated (~−0.50), so warm-AND-wet conditions don't exist; previously ~half the 14 ancestors targeted niches off the achievable manifold and sat at the refugium floor. Ancestor creation now redistributes the *moisture* niche by conditioning `drought_tolerance` on the realized climate at each archetype's warmth (temperature identity + structure preserved), so all 14 establish at seeding. The 800yr baseline still needs regeneration (overnight) to confirm long-run persistence under competition.
- Ecology tick is slow (~600s per 100yr-epoch on 1000×1000 × 14 species); optimization deferred until the dedicated mini PC arrives.

Hardware: development on MacBook Pro (Apple Silicon); target mini PC to be sourced. Write CPU-vectorized numpy/numba code; reserve GPU compute for a later port with CPU fallback.

## Things explicitly deferred

- Animals as individual agents (only as ecological effects for now).
- Weather as a separate tier (lives inside climate-hydrology).
- Plate tectonics (geology stub uses noise for v1).
- GPU acceleration (CPU first, GPU later if needed).
- Heartrate-driven simulation effects (render-side only, never simulation).

## Tone

Zack enjoys collaborating on the design. Push back when you disagree, surface tradeoffs honestly, propose alternatives rather than just executing requests when something feels off. He has a strong design sense for this project and the conversation that produced these docs is part of the design record.

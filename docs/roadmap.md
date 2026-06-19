# Roadmap

Build order is **bottom-up with cross-cutting infrastructure pulled forward**. The discipline is to have *visible feedback* available before there's anything real to look at, so each subsequent phase has immediate verification.

> **Status note (June 2026):** Phases 0–8 are complete. The ecology built in
> Phases 6–7 was substantially **rewritten** afterward (simpler unified tick,
> 6-trait genome, no seed banks, baseline competition, deferred
> extinction/speciation) — see `ecology-tick-refactor.md` and
> `simulation-design.md` for the current design. The phase descriptions below
> reflect the *original* build sequence, not the current ecology internals.

## Phase 0: Skeleton and decisions

- Set up repo structure (see `architecture.md`).
- Pin Python version (3.12), package manager (uv), linting/formatting (ruff).
- Lock in format choices: Zarr for rasters, SQLite for individuals/events, JSON/MessagePack for manifests.
- Define `World` class skeleton — data model only, no behavior.
- Write seed/RNG infrastructure: deterministic substreams per `(world_seed, tier_id, pass_id, tick_number)`. **Get this right early.**

## Phase 1: Canonical state I/O

- `RasterStore`: named layers, memory-mapped read, atomic versioned write, chunked access.
- SQLite schemas: species, distinguished individuals, events. Spatial and temporal indices.
- `World` object as unifier: knows seed, tier clocks, version; can save/load.
- Round-trip tests through save/load.

## Phase 2: Query interface

- Point queries, region queries, individual lookups, temporal queries.
- Thin wrapper at first: raster sampling + SQL.
- Exists *as an abstraction layer*; both simulation tiers and extractors use it.

## Phase 3: Top-down debug visualizer

- Standalone Python script (matplotlib or pillow).
- Reads a `World`; produces PNGs of various layers.
- Configurable: single layer, composite, region zoom.
- Ugly-but-informative, not pretty. **This is your eyes for the rest of the project.**
- Verify by populating world with synthetic data (Perlin heightmap, fake species, hand-placed individuals).

**Checkpoint**: scaffolding + eyes. Zero simulation. The next phases add one tier at a time, each immediately visible.

## Phase 4: Geology stub + initial topography

- Noise-based heightmap (architected as a swap-in for tectonics later).
- Bedrock assignment (Voronoi regions with rock-type labels).
- Soil parent material derivation.
- Geology tier ticks exist as plumbing; current tick is mostly a no-op.

## Phase 5: Climate-hydrology — first real simulation tier

- Climate envelope: latitudinal temperature, prevailing winds, base precipitation, orographic effects, rain shadows.
- Watershed and flow accumulation.
- River graph construction (confluences, reaches, lakes).
- Hydraulic erosion: pure CPU (numpy/numba) particle-based or grid-based. **This is the pass that carves the heightmap into something that looks real.**
- Sediment deposition.
- Soil moisture derived field.
- Derived-state cache populated.

**Checkpoint**: simulation transforms world from "noise" to "alive." Top-down viewer shows real-looking terrain with rivers, watersheds, valleys. Run "advance 1000 years" and see further erosion. Verify reproducibility from seed.

## Phase 6: Ecology — basic

- Species data model: genome, range, history.
- Small ancestor pool: 5-10 species, widely separated traits.
- Establishment suitability (Liebig minimum).
- Competition term.
- Population dynamics: growth, mortality, dispersal.
- Simple succession.

**Deferred to Phase 7**: speciation, distinguished individuals, disturbance.

**Checkpoint**: run "advance 100 years" and see species distributions shift across map. First phase that produces *the magic*.

## Phase 7: Ecology — advanced

- Distinguished individuals: promotion rules, persistence, event accumulation, post-mortem decay (snag → log → mound).
- Speciation via population fragmentation and divergence clocks.
- Disturbance regimes: fire CA (with hydrology coupling for post-fire erosion), blowdowns, floods.
- Mutation pressure under climate change.
- (Keystone species can come later if scope is tight.)

**Checkpoint**: simulation is *fully alive*. Run for a few million simulated years and observe radiated lineages, landmark trees, fire history, the works.

## Phase 8: Orchestrator and ride-triggered advancement

- Orchestrator: schedule tier ticks during a given advance.
- CLI entry point: detect new ride log, compute time to advance, run sim, write new state.
- Manual commands: introduce fire, introduce supervolcano, advance N years, rebuild.
- Reproducibility verification: replay from seed, verify bit-identical state.

## Phase 9: Godot extractor and chunks

- Bike route selection over current world state (cyclability constraints, interesting terrain).
- Chunk generation along route.
- Manifest production.
- Godot side: consume chunks (existing renderer adapted to new format).

## Phase 10: Asset agent integration

- Species → visual description prompt construction.
- Asset library management.
- Regeneration triggers on species drift.
- Distinguished individuals get individual generations.
- Caching strategy.

## Estimated timeline

Sessions of ~1-2 hours, working with Claude Code, with engaged steering:
- ~25-40 sessions to end of Phase 7 (simulation does the cool thing).
- ~10-15 more sessions to end of Phase 10 (working bike ride through an evolved world).

Roughly 2-3 months to MVP simulation, 4-5 months to working bike ride. Faster if pushed, slower if breathing.

## Process recommendations

**Tests scoped specifically.** Test data plumbing (serialization, query correctness, reproducibility) rigorously. Test simulation pass *output shape* and *invariants* (conservation laws like total water doesn't change), not aesthetic correctness. The visualizer is for aesthetic verification.

**Commit often, with descriptive messages.** Enables git-bisect when regressions sneak in. Claude Code works better with frequent commits.

**Keep a design log.** `docs/decisions.md` should record *why* decisions were made, not just what. Decisions made in this initial conversation are captured in the existing docs; future decisions should append there.

**Resist optimization until something is slow.** First version of every pass = simplest correct version. Profile when something hurts. Premature optimization locks in wrong data layouts.

## Hardware

Development: MacBook Pro (Apple Silicon).
Target: Mini PC, to be sourced. Specs to look for:
- CPU: Ryzen 7 7840HS / 8845HS or Intel Core Ultra 7 155H tier (6-8 modern cores).
- RAM: 32 GB minimum, 64 GB if headroom desired. DDR5.
- Storage: 1 TB NVMe.
- iGPU with compute support (Radeon 780M or Intel Arc) — useful for later GPU passes, not required.
- Target models around $600-800: Beelink SER8, Minisforum UM790/UM890.

Cross-platform compatibility: write CPU-vectorized numpy/numba first; port hot passes to GPU compute later with CPU fallback for the Mac. JAX is an option if abstraction over both becomes important.

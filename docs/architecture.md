# Architecture and Design

This document captures the design decisions made during initial planning. The *why* matters as much as the *what*; future changes should engage with the reasoning, not just the structure.

## The three-tier simulation

### Tier 1: Geology
- **Timestep**: 10K-100K years per tick
- **State**: bedrock type map, fault/joint structure, mineral distribution, large-scale base topography
- **Representation**: coarse grid (~500m-1km cells) plus vector features (faults, intrusions, plate boundaries)
- **Frequency**: Ticked rarely — maybe once per several months of real play
- **v1 approach**: Stub with noise-based heightmap and simple bedrock assignment. Architected so plate tectonics can swap in later.

### Tier 2: Climate-hydrology
- **Timestep**: 10-100 years per tick
- **State**: climate envelope (winds, temperature gradients, precipitation), river graph (nodes = confluences, edges = reaches with flow rates), watershed polygons, lakes, soil moisture, sediment
- **Representation**: climate on coarse grid (~200-500m); rivers as graph; watersheds as polygons; derived fields at the 50m main resolution
- **Frequency**: Most ride-triggered advances tick this 0-2 times
- **Key insight**: contains a slow *climate envelope* (centuries-scale drift) and produces *weather realizations* on demand (sampled from envelope, possibly cached). This means we don't need a separate weather tier.

### Tier 3: Ecology
- **Timestep**: 1 season per tick (4 ticks/year)
- **State**: species (6-trait genome + range + history), distinguished individuals, disturbance events
- **Representation**: species density fields at 50m; distinguished individuals as point data with full records
- **Frequency**: Ticked on every ride-log advance, multiple times per advance
- **Note**: the ecology tick was rewritten (June 2026) to a single unified grow/compete/disperse rule; see `simulation-design.md` and `ecology-tick-refactor.md`. Seed banks were removed in favor of dispersal + a refugium floor.

## Tier communication

The pattern is **read-only downward references** plus **batched event emission upward**.

- Ecology reads continuously from the climate-hydrology derived-state cache (cheap lookups).
- Ecology emits events upward (fire scar at X, beaver dam placed at Y) that climate-hydrology consumes at its next tick.
- Climate-hydrology reads continuously from geology; emits sediment-deposition and incision events upward.
- Slow tiers emit *state changes* downward (glacier advance wipes ecology); fast tiers emit *events* upward.

Tiers do not call each other directly. The event log is the contract.

## The derived-state cache

The performance trick that makes rich ecology tractable on a mini PC: climate-hydrology maintains a cache of derived fields at ecology's resolution. These fields are precomputed at each climate-hydrology tick:

- Effective soil moisture (summer / winter)
- Frost pocket likelihood
- Growing degree days
- Flood return interval
- Solar insolation accounting for slope and aspect
- Distance to permanent water

Ecology reads from this cache constantly but cheaply. The cache only recomputes when climate-hydrology ticks (rarely).

## Spatial extent and resolution

- **World extent**: 50 km × 50 km
- **Boundary model**: Bounded with implied larger context. Climate flows in from outside (prevailing wind direction). Species can occasionally immigrate from "beyond the edge." Not toroidal.
- **Main raster resolution**: 50m cells (1M cells per layer)
- **Geology resolution**: 500m-1km cells
- **Climate envelope resolution**: 200-500m cells
- **Distinguished individuals**: stored as floating-point coordinates (sub-cell precision)
- **Renderer mesh resolution**: higher than 50m, achieved by sampling/interpolation at extraction time

## Data layer

### Canonical state (Layer A)
- **Rasters**: Zarr arrays, memory-mapped, chunked, compressed. Versioned per simulation tick (immutable per tick). One Zarr group per tier.
- **Individuals and events**: SQLite database, append-mostly. Schemas for species, distinguished individuals, events with spatial and temporal indices.
- **Geological vectors**: stored as GeoJSON or in SQLite with WKB geometry.
- **The `World` object** ties all of this together; knows seed, tier clocks, version; serializable.

### Query interface (Layer B)
Public API that simulation tiers and extractors both use. Operations:
- Point queries: "what's at (x, y) at time t?"
- Region queries: "species composition in this polygon"
- Individual lookups: "distinguished individuals within radius R of point P"
- Temporal queries: "events that occurred here in the last 100 years"

Implemented as an in-process Python library initially. Same API would work as an HTTP/gRPC service if needed later.

### Extractors (Layer C)
Renderer-specific code that consumes the query interface and produces output for a specific renderer.
- `extract/godot/`: produces chunks-and-manifest format for the Godot bike renderer.
- `extract/debug2d/`: produces top-down PNGs for development feedback.
- Future: could add JSON dumps, LLM-friendly region descriptions, alternative renderers.

**Critical discipline**: the simulation never produces a chunk file directly. Extraction is a separate pass that runs after the simulation tick completes.

## Bike route selection

Route selection is part of *extraction*, not simulation. Given a current world state, the route extractor picks a 25-30 km loop (or other shape) through interesting terrain that obeys cyclability constraints (no cliffs, manageable gradients). This means:

- Different rides through the same simulated world are possible without re-simulating.
- Route selection logic is isolated and replaceable.
- The route depends on the current state of the world, so route choice naturally varies as the world evolves.

## Reproducibility

Every tier and every pass gets a deterministic RNG substream derived from `(world_seed, tier_id, pass_id, tick_number)`. Replay from seed plus the log of advance triggers must produce bit-identical state.

This costs nothing at runtime and provides:
- Cheap "rewind" capability
- Robust debugging (replay to any past state)
- Insurance against state corruption

## Asset agent integration

The asset agent is a sidecar process that:
- Watches for new species and significantly mutated existing species.
- Generates art from the species genome and morphological description.
- Caches assets per species ID; all individuals of a species look like siblings.
- Generates individual assets for distinguished individuals on promotion (with their specific traits — "this oak, age 340, lightning-struck, leaning").

The species' genome is the canonical input. Mutation tracking determines when regeneration is warranted.

## Things explicitly out of scope (for now)

- Animals (beyond ecological effects of generic herbivores / engineers; no individual creatures yet).
- Weather as a separate tier (handled inside climate-hydrology as realizations from envelope).
- Player effects on the world (rider is pure observer).
- Multiplayer or shared worlds.
- Heartrate-driven simulation effects (heartrate is render-side only).

## Open questions to revisit later

- Whether to GPU-accelerate hydraulic erosion. CPU-first; revisit if too slow.
- Whether to add animals as a first-class agent layer or keep them as effects (grazing pressure as a field, etc.).
- How to handle "world is getting stale" detection automatically.
- How to surface the world's history to the player. Names? Visible markers? Or strictly through the landscape itself?

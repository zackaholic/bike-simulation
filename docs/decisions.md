# Decision Log

Records significant design and implementation decisions. Append entries chronologically. Capture the *why*, not just the *what*.

---

## Initial planning (foundation)

All entries below come from the initial planning conversation; they represent the starting point.

### Three-tier architecture (geology / climate-hydrology / ecology)
**Decision**: Three tiers, not two. Climate-hydrology is its own tier between geology and ecology.

**Why**: Climate-hydrology genuinely occupies a different timescale (slower than ecology, faster than tectonics) and folding it into either neighbor creates problems. Folded into geology, it becomes a frozen backdrop. Folded into ecology, it becomes too expensive at ecology's resolution.

### Climate and hydrology in one tier (not separated)
**Decision**: Keep climate and hydrology unified. Internally distinguish slow climate envelope from fast weather realizations.

**Why**: Climate and hydrology are mutually defining at their natural timescales. Separating them creates a constantly-crossed boundary. The split that *does* pay off — slow envelope vs. fast realizations — sits inside this tier rather than between it and another.

**Considered alternative**: separate climate and hydrology tiers (Zack initially leaned this way for flexibility). Pushed back; agreed unified is right.

### Event-based tier communication
**Decision**: Tiers communicate via read-only downward references and batched upward events. No direct tier-to-tier calls.

**Why**: Keeps coupling explicit and decoupled timing tractable. Matches the actor-model pattern; scales as we add complexity.

### Decoupled rendering via three layers
**Decision**: Layer A (canonical simulation state) → Layer B (query interface) → Layer C (renderer-specific extractors). Simulator never writes renderer formats directly.

**Why**: Lets us change renderers without re-simulating. Route selection becomes part of extraction (different routes through same world without sim work). Multiple extractors (Godot, 2D debug, JSON, etc.) all work over the same query API.

### Reproducibility from seed
**Decision**: All randomness uses deterministic substreams derived from `(world_seed, tier_id, pass_id, tick_number)`.

**Why**: Free cheap rewind, robust debugging, insurance against corruption. Costs nothing at runtime. Painful to retrofit.

### 50 km × 50 km world at 50m resolution
**Decision**: Bounded world (with implied larger context for climate boundaries and immigration), 50m main raster cells.

**Why**: 45-min ride at 25-30 km/hr covers ~20-25 km; 50 km square gives loop flexibility. 50m resolution = 1M cells per layer, comfortable on a mini PC. Climate envelope coarser (200-500m); geology coarser still (500m-1km).

### Generation = advancement at large scale
**Decision**: New worlds are created by simulating deep history (~200M years) and sampling the present. There is no separate "initial generation" code path.

**Why**: Architectural elegance. The "advance overnight" feature is just stepping the simulation forward at a smaller scale. Worlds arrive already old.

### Rider is pure observer
**Decision**: No back-channel from rendering to simulation. Heartrate and pedaling affect render-time effects only.

**Why**: Preserves the meditative Proteus quality. Avoids gamification creep. Keeps architecture simpler.

### Ride duration → simulation time
**Decision**: Each ride contributes simulation time to overnight advancement, with a cap. Suggested starting point: 20-50 years per ride.

**Why**: Couples world's continued existence to the player's engagement, which has emotional weight. Cap prevents jarring "came back to unfamiliar world" after a long ride. Skip a day → world waits.

### Species-and-individuals model (not plant-by-plant)
**Decision**: Two-resolution ecology. Species/population layer with genome + density field. Distinguished individuals layer for the ~thousands of plants the cyclist might form relationships with. Most plants are stochastically sampled by the renderer from the density field, not persistent.

**Why**: Plant-by-plant doesn't scale and isn't necessary. Cyclist forms relationships with landmark trees, not with random shrubs. Species-level continuity with individual-level anchors is the right granularity.

### Speciation via fragmentation + divergence clock
**Decision**: Track each species' range as a graph of connected populations. When isolated, start a divergence clock; traits drift directionally toward local conditions; eventually speciate.

**Why**: Produces real evolutionary radiation traceable to specific geographic/climatic events. Player can sense lineages over time. Better than curated species lists.

### Start small, let radiation happen
**Decision**: Begin worlds with 5-10 ancestor species with widely separated traits. Don't pre-populate with curated flora.

**Why**: Avoids unrealistic uniform mixing. Each world develops its own flora belonging to its specific history. Mass extinction events become meaningful (survivors radiate into empty niches).

### Stochasticity at every tier
**Decision**: Don't enforce strict bottom-up causation. Each tier has its own internal stochasticity (long-distance dispersal, beetle outbreaks, etc.).

**Why**: Real ecosystems aren't fully causally derivable. Strict causation makes worlds feel *less* alive. "Huh, what's that doing here?" moments need some genuine irreducibility.

### Worlds can die
**Decision**: Terminal trajectories (snowball earth, runaway desertification) are intentional features, not bugs. Player can watch them play out or manually intervene (introduce fire, supervolcano, etc.).

**Why**: Worlds that can die are worlds that matter. Rebirth after devastation is emotionally significant.

### Build order: bottom-up with eyes pulled forward
**Decision**: Phase 0-2 builds scaffolding (state, query). Phase 3 builds the top-down debug visualizer *before* any simulation logic. Then build tiers in order of experiential impact (climate-hydrology before geology depth, ecology after).

**Why**: Need visible feedback before any tier exists, so subsequent phases have immediate verification. Climate-hydrology transforms world from "noise" to "alive" faster than geology depth does.

### Tech stack: Python + numpy/numba CPU first
**Decision**: Python 3.12, uv, ruff. numpy/numba for vectorized passes. Zarr for rasters. SQLite for individuals/events. GPU later if needed.

**Why**: Matches Claude Code's strengths and Zack's tooling. Cross-platform between MacBook Pro development and mini PC target. GPU port has a clean place to slot in (hot passes only, CPU fallback).

---

## Subsequent decisions go below this line

### Zarr v3 API adaptation
**Decision**: Use Zarr v3 API (can't pass both `data` and `dtype` to create_array). Let Zarr infer dtype from data.

**Why**: We pinned zarr>=3.0 and the v3 API changed parameter validation. Simpler to let it infer.

### list_layers returns sorted results
**Decision**: RasterStore.list_layers() returns sorted layer names.

**Why**: Zarr group key iteration order is not guaranteed to be deterministic across store instances. Sorting ensures reproducible iteration for tests and deterministic behavior.

### EventStore.list_species returns dicts not strings
**Decision**: Changed list_species() to return list of dicts (with species_id, parent_id, appeared_year) instead of list of strings.

**Why**: Ecology tier needs parent_id for speciation logic, and iterating species as dicts is more natural than fetching each separately.

### Climate-hydrology tick threshold logic
**Decision**: Climate ticks when `ecology_simulated_year // 1000 > climate_tick_number - 1`, subtracting the bootstrap tick from create_world().

**Why**: create_world() ticks climate once to bootstrap derived-state cache. This initial tick shouldn't count toward the ecology-driven threshold. The subtraction ensures the first ecology-triggered climate tick happens at year 1000, not year 0.

### Speciation via downsampled connected components
**Decision**: Run connected component analysis on a 10x downsampled grid (100x100) rather than the full 1000x1000 grid.

**Why**: BFS flood fill on 1M cells in pure Python is too slow. 100x100 (10K cells) is fast and still detects landscape-scale fragmentation. Fine-grained fragmentation at the cell level isn't meaningful for speciation anyway.

### Disturbance skipped on tick 0
**Decision**: Fire and blowdown disturbance is skipped on the first ecology tick.

**Why**: Ensures all disturbance events have simulated_year > 0, which tests rely on and which makes logical sense (no vegetation to burn before initial establishment).

### Ride duration mapping
**Decision**: 1 simulated year per minute of riding, capped at 50 years.

**Why**: Starting point from design docs. Couples world existence to rider engagement (emotional weight) while preventing jarring fast-forwards. Cap ensures the rider doesn't return to an unrecognizable world after a long ride.

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

## Versioning architecture for observability

**Context**: Phases 1-8 are complete. Building the webview observability tool requires reading historical world states, not just the current one. This forced a real design pass on how versions are stored, indexed, and queried.

### Decision: copy-on-write per-version Zarr groups

**Decision**: Each world version (snapshot after an advance) gets its own Zarr group. Only layers that were actually written during that advance are stored in the version's group. Layers that didn't change are not duplicated — they remain in the version that last wrote them.

**Why**: Geology and climate layers change rarely (most advances tick only ecology). Naively snapshotting every layer per version would waste enormous storage; mutating layers in place would destroy historical inspection. Copy-on-write gives both: history is preserved, storage scales with actual change.

**Considered alternative**: Snapshot every layer every version. Rejected on storage grounds (~hundreds of MB per advance, gigabytes per month).

### Decision: explicit layer-ownership index per version, not backward-walking lookups

**Decision**: Each version's manifest stores an explicit `{layer_name: owning_version_id}` map covering *all* layers, not just the ones it wrote. Writing a new version copies the previous manifest and overrides entries for layers that changed.

**Why**: Initial plan was to "walk backward through versions to find which one owns a layer." This works but creates several problems: O(versions) lookup at read time, dependency on filesystem state for correctness, harder to debug, harder to implement future "consolidate old versions" operations. An explicit index makes lookups O(1), makes the manifest a complete description of the version, and makes `cat manifest.json` a real debugging tool. Storage cost of the redundant index entries is negligible (small JSON blob per version).

**This was a course-correction from the initial plan**: the original proposal had implicit backward-walking; Claude flagged it and we switched to explicit indexing before implementation.

### Decision: distinguish "exists at version V" from "alive at version V"

**Decision**: The SQLite schema and query API must distinguish entities that existed in the historical record by time V from entities that were currently alive at time V. Distinguished individuals and species need both `appeared_year` and `died_year` (or equivalent) fields. Queries express the distinction explicitly.

**Why**: Filtering only by `appeared_year <= V.year` returns everything that ever existed by then, including entities that died centuries ago. Point inspection in the webview shows *currently alive* vegetation; lineage browsing shows *all species that ever existed*. These are different queries against the same data and the schema must support both.

**Risk if missed**: webview would show "ghost trees" — long-dead individuals appearing as alive at past versions where they had in fact already died.

### Decision: per-tier clocks in version reconstruction

**Decision**: A world version stores *all* tier clocks (geology, climate-hydrology, ecology), not a single "simulated_year." A `version_for_tier(version_id, tier)` helper returns the right clock value for the right tier. Queries against tier-specific data use the tier-specific clock.

**Why**: Tier clocks tick independently and at vastly different rates. Geology might be at year 1,500,000 while ecology is at year 1500 within the same version. Conflating them into a single "current year" loses information and causes subtle bugs when querying across tiers. Tier clocks should be first-class in the version manifest.

### Decision: atomic manifest writes

**Decision**: Manifest writes go through a temp-file-then-rename pattern with `fsync` between. Optionally keep the previous N manifests as backups.

**Why**: The manifest is the single point of failure for the versioned history. Zarr data is recoverable from any working manifest; a corrupted manifest makes the whole version graph unreadable. Atomic writes are cheap insurance.

### Decision: defer storage optimizations

**Decision**: Don't optimize species density storage (lower precision, delta encoding, hashing-to-detect-unchanged) until storage actually becomes a problem.

**Why**: Estimated growth is ~10-30 MB per advance for ecology layers after Zarr compression. On a 1 TB SSD, that's hundreds of advances before storage matters at all. Premature optimization here locks in data layouts and complicates the simple case. The levers exist (uint8 with scale factor, delta encoding, content hashing) and can be pulled later if needed.

### Decision: don't build migration tooling, don't worry about concurrent writes, don't build old-version deletion

**Decision**: These are all deferred indefinitely.

**Why**: Single user, single writer (the simulator), no production deployment. Schema changes are handled by regenerating worlds. Storage is cheap enough that old versions accumulating is fine. Building any of this now is solving non-problems.

### Process note

This conversation was a good example of *why* the decision log matters. The initial plan was structurally sound but had two real issues (backward-walking lookups, alive-vs-existed ambiguity) that would have caused problems later. Catching them before implementation cost an hour of design discussion; catching them after would have meant a schema migration. The pattern of "Claude Code proposes a concrete plan, Claude (chat) sanity-checks it" is working well and should continue for architectural decisions in subsequent phases.

## Particle-based hydraulic erosion (gap #1)

**Context**: The original erosion implementation was a placeholder — 5 passes of grid-based diffusive smoothing producing only cosmetic terrain changes. This was the first of five known gaps to address, chosen first because terrain shape is foundational to everything downstream (river graph, ecology, rendering).

### Decision: particle-based over enhanced grid-based

**Decision**: Replace the grid diffusion placeholder with particle-based hydraulic erosion (virtual water droplets flowing downhill, eroding and depositing sediment).

**Why**: Particle-based erosion naturally produces the features that make terrain feel water-carved — V-shaped valleys, alluvial fans, differential channel depths. Grid-based approaches can be enhanced but tend toward uniform smoothing rather than channelised carving. The overnight compute budget makes particle simulation tractable even at 200K particles per tick.

**Considered alternative**: Keeping grid-based but running many more passes with a stream-power formula. Rejected because it still lacks sediment transport and produces broad smoothing rather than discrete channels.

### Decision: separate erosion module (`erosion.py`)

**Decision**: Core erosion algorithms live in `src/bike_sim/tiers/erosion.py`, a pure-numpy module with no bike_sim imports. `climate_hydrology.py` calls into it.

**Why**: The particle simulation is 400+ lines of self-contained numerical code. Keeping it in `climate_hydrology.py` would blur two concerns (climate envelope vs. terrain carving). The pure-numpy design means the inner particle loop (`_simulate_particle`) can be wrapped with `@numba.njit` later without restructuring.

### Decision: simple scalar bedrock erodibility

**Decision**: Each of the 6 bedrock types maps to a single erodibility float (0.3 for granite to 1.2 for shale). Erosion erodes sediment first; only after sediment is depleted does it cut bedrock, scaled by erodibility.

**Why**: Produces visible differential erosion (hard ridges, soft valleys) with minimal complexity. The geology tier already has type labels; this adds properties without changing the geology interface.

**Note for future**: Could expand to `{hardness, solubility, fracture_tendency}` per rock type for more nuanced weathering. The current scalar approach doesn't block this.

### Decision: thermal erosion as companion pass

**Decision**: After hydraulic particle erosion, run a vectorised thermal erosion pass (talus creep) that redistributes material on slopes exceeding the angle of repose.

**Why**: Particle erosion alone creates sharp channels but unrealistic cliff faces. Thermal erosion softens these into scree slopes and fills narrow gullies — the complementary weathering process that makes landscapes look physical rather than algorithmically carved.

### Decision: sediment as a separate layer, not merged into heightmap

**Decision**: Erosion produces two outputs: `eroded_heightmap` (bedrock surface) and `sediment_depth` (unconsolidated material on top). The combined surface is `eroded_heightmap + sediment_depth`.

**Why**: Sediment and bedrock have different erodibility (sediment is always easy to erode; bedrock depends on type). Tracking them separately means particles naturally cut through sediment deposits before hitting bedrock. The sediment layer also feeds the derived-state cache (thick sediment increases soil moisture) and will be visible to ecology.

### Decision: post-erosion flow accumulation recomputation

**Decision**: Flow accumulation is computed twice per tick — once pre-erosion (for particle spawn weighting) and once post-erosion (on the combined surface, for the derived-state cache).

**Why**: Erosion changes the terrain, which changes where water flows. The derived cache (especially `distance_to_water`) must reflect the post-erosion landscape, not the pre-erosion one. The extra flow computation is ~2s and worth the accuracy.

### Decision: erosion scope separate from river graph

**Decision**: This work produces carved terrain + sediment. Discrete river channels, lakes, meanders, and springs are deferred to river graph work (gap #2).

**Why**: Pragmatic separation of concerns. The particle erosion naturally produces channel-like carving in the heightmap that the river graph extractor can later identify and formalise. Bundling everything would make the scope unmanageable.

**Note**: The design docs (`simulation-design.md:42-48`) arguably bundle sediment transport with river graph construction. If this separation causes problems (e.g., realistic deltas need sediment flowing along discrete channels), we may need to revisit.

### Decision: literature defaults for parameters, tune visually

**Decision**: Start with well-known defaults from terrain erosion literature (inertia 0.1, capacity factor 6.0, deposition/erosion rate 0.3, etc.). Tune using the webview inspector.

**Why**: These parameters interact nonlinearly; theoretical derivation is less useful than visual tuning against the project's aesthetic goals. The webview gives us the inspection tools to iterate.

### Scale calibration note

Each tick represents 1000 simulated years. 200K particles per tick, each representing aggregate storm erosion. Starting estimate: this produces 10-50m of carving in major drainage paths. Primary tuning knobs if terrain looks wrong: `num_particles`, `erosion_rate`, `capacity_factor`.
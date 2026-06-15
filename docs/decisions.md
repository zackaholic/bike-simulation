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

Each tick represents 1000 simulated years. 70K particles per tick, each representing aggregate storm erosion. After initial tuning: ~11m mean erosion per climate tick, with 99th percentile carving ~50m in major drainage paths. Primary tuning knobs if terrain looks wrong: `num_particles`, `erosion_rate`, `capacity_factor`.

### Decision: erosion accumulates across climate ticks

**Decision**: On tick > 0, erosion reads the previously eroded heightmap rather than the geology baseline. Each climate tick deepens channels carved by previous passes.

**Why**: Caught during testing — the original implementation read the geology heightmap every tick, discarding prior bedrock carving. This made erosion non-cumulative (each tick was independent). Cumulative erosion is the whole point: the second pass preferentially deepens existing channels because flow accumulation concentrates in them, producing increasingly realistic drainage networks through process rather than placement.

**Detail**: Climate envelope (temperature, precipitation) still uses the geology heightmap for lapse rate and orographic effects, since those depend on bedrock structure. Only the erosion and flow accumulation inputs use the evolving surface.

### Future enhancement: intermediate version snapshots during advance

**Decision (deferred)**: Currently `advance(N)` commits one version at the end, regardless of how many tier ticks occurred. A 1050-year advance produces one snapshot, not intermediate views at each ecology or climate tick.

**Why this matters**: The world is experienced as snapshots — a ride at one point in time, then another ride days later at a later world date. To tune how aggressively the world changes per tick, or how much to advance between rides, you need to *see* intermediate states, not just before and after a large advance. Without intermediate snapshots, there's no way to evaluate whether 50 years of ecology produces the right amount of visible change for one ride-to-ride interval.

**Future approach**: Add a `snapshot_interval` option to `advance()` (or to the Orchestrator) that commits a version every N ecology ticks or whenever a slow tier ticks. This would let the webview show the world unfolding in fine steps. The version picker and diff tooling already support arbitrary numbers of versions — this is purely an orchestrator change.

## Experience-first design principles (course correction)

**Context**: The original design was simulation-first — tier timescales were chosen to match geological/ecological realism, with the rider experience assumed to follow. This produced a structural problem: the climate-hydrology tier ticks every 1,000 simulated years, but a typical ride advances ~50 years. The rider either sees no landscape change for ~20 rides, then a sudden lurch (11m of mean erosion, completely recomputed drainage), or the landscape is effectively static. Neither serves the project's emotional core.

This is a set of foundational principles that refine (not replace) the original design. They represent a shift from "simulate realistically, then render" to "design the experience, then choose what to simulate."

### Principle: Experience drives simulation design, not the reverse

**Decision**: Every simulation decision — what to model, at what resolution, at what timescale — should be evaluated against what the rider will actually perceive and feel. Unlimited overnight compute is a resource to be spent where it most directly serves the ride experience, not on physical realism for its own sake.

**Why**: Hours of compute on plant interactions that produce visible succession, shifting treelines, recovery from fire — time well spent. The same hours realistically eroding a mountain peak the rider will never perceive changing — wasted. The compute budget is unlimited but the rider's attention is not.

### Principle: Change should be continuous and proportional, not stepped and lurching

**Decision**: The tick system must not create artificial discontinuities. If a ride advances 50 simulated years, the world should reflect 50 years of gradual change, not zero change or 1,000 years of change applied in a lump. Big visible changes should only occur when grounded in world events (a major flood, a devastating fire season) — never as an artifact of tick boundaries.

**Why**: The rider builds a relationship with a world that feels alive. Sudden unexplained lurches break that relationship. Gradual change rewards sustained attention — "is that streambed a little deeper?" is the right feeling.

### Principle: Perturbations push toward interesting states, not forced novelty

**Decision**: The simulation should allow (not force) unusual states: a species dominating for centuries, a wet period ravaging a landscape, a dry season producing devastating fires. Weather/climate cycles are a key mechanism for this. World rules should permit recovery from extreme states without mandating it — some worlds may enter terminal trajectories, and that's meaningful.

**Why**: Forced novelty feels arbitrary. Emergent novelty from perturbation + recovery feels earned and produces "what happened here?" moments. The rider becomes a naturalist reading a landscape shaped by events, not a consumer of procedurally varied content.

### Principle: Rider's perceptual scale is the resolution target

**Decision**: Simulation complexity should concentrate at the scale the rider perceives: ground-level, cycling-speed, across 25-30km of terrain. Microclimates, vegetation shifts, water level changes, individual trees — these register. Continental-scale processes matter only insofar as they produce ground-level effects.

**Why**: A 0.5m streambed change is noticeable at cycling speed. A 0.5m change on a distant peak is not. Compute and design effort should follow perception.

### Principle: Balance consistency (relationship) with change (exploration)

**Decision**: Some world features must be anchors — major landforms, geology, distinguished individuals — providing continuity so the rider recognizes "their" world. Other features — vegetation, water, weather, disturbance scars — are the source of novelty. The tick system and advancement rate must be tuned so both are present: enough stability to build attachment, enough change to sustain curiosity.

**Why**: A world that changes too fast doesn't reward revisiting. A world that changes too slowly doesn't reward persistence. The rider should notice what's different against a familiar backdrop — spatial novelty (new routes) and temporal novelty (familiar places changed) are complementary.

## Tick system redesign: seasonal core loop with continuous hydrology

**Context**: Following the experience-first principles above, we worked through what the rider actually perceives changing and how the tick system should serve that. The key insight: the rider reads geology and hydrology *through vegetation*, just like a real naturalist. A hillside stripped of nutrients by back-to-back floods is experienced as "that species is suddenly out-competing here," not as "the sediment layer changed." Simulation design should follow the same logic.

**Why these decisions are a package, not independent choices**: The original design had three tiers ticking at vastly different rates (5 years, 1000 years, 100K years). The experience problem was that climate-hydrology's 1000-year tick meant either no landscape change for ~20 rides or a sudden lurch. The fix isn't just "tick climate more often" — that would be expensive and still wouldn't produce the *kind* of change that matters to the rider. Instead: (1) make ecology seasonal so it can express seasonal strategies and respond to weather variation — this is where the rider's attention lives; (2) add a weather cycle system so ecology has something interesting to respond to — overlapping cycles produce emergent extremes without scripting; (3) make erosion a continuous side-effect of weather rather than a batched pass — this eliminates lurches and couples terrain change to the same weather that drives ecology. Each piece enables the others: seasonal ecology needs seasonal weather to be interesting; weather cycles need seasonal ecology to have visible effects; continuous erosion needs weather-driven intensity to avoid being either too uniform or too lurchy.

### Decision: Seasonal ecology ticks (replacing 5-year ticks)

**Decision**: Ecology ticks at seasonal resolution (~4 ticks/year, ~200 ticks per 50-year ride advance) instead of the current 5-year ticks.

**Why**: Seasonal resolution isn't just more granularity — it makes the ecology model fundamentally richer. Species can have seasonal strategies (spring ephemerals, summer-drought grasses, frost-hardy evergreens). Mortality becomes seasonal (winter kill, summer drought stress). Seed bank timing matters (fire after seed set vs. before). A bad winter followed by a wet spring is a different event than the reverse. These distinctions produce the subtle ride-to-ride variation that sustains attention.

**Implication**: The ecology tier needs significant rework — dormancy, phenology, seasonal growth rates, winter kill as distinct processes. This is a model change, not a parameter change.

### Decision: Weather system with overlapping cycles (new subsystem)

**Decision**: Replace the static climate envelope with a weather system that generates season-by-season conditions from overlapping deterministic oscillatory cycles. Cycle periods derived from world seed: short cycles (3-7 years), medium cycles (30-50 years), long cycles (200-500 years), and possibly longer.

**Why**: Overlapping cycles with different frequencies produce beat patterns — rare alignments that create conditions no single cycle produces alone. A 7-year wet cycle and a 23-year warm cycle align once every ~161 years. This is the engine for world-unique emergent events: superblooms, mass die-offs, species explosions into marginal habitat. None of these are scripted; they emerge from cycle interference. Each world gets its own characteristic rhythms (from seed), so each world has its own signature rare events. The rider discovers these rhythms over months of riding — enormous emotional payoff from pure rule-following.

### Decision: Hydrology as continuous weather-driven side effect (replacing batched 1000-year erosion)

**Decision**: Erosion and sediment transport become a lightweight per-season operation driven by that season's weather, replacing the current 70K-particle batch pass every 1000 simulated years. Each season's rainfall drives proportional erosion. Wet years erode more; dry years almost nothing. Extreme precipitation events carve noticeably.

**Why**: This eliminates the artificial lurch problem (no change for 20 rides, then sudden 11m of erosion). It couples hydrology to weather naturally — the flood that erodes a bank also kills riparian vegetation and deposits sediment that changes soil downstream. The rider sees recovery from these events over subsequent rides. Computationally, a lightweight per-season flow-and-erode step along the drainage network is much cheaper than 70K particles over the whole terrain, and the total erosion over 1000 years of seasonal application should be comparable.

### Decision: Large landform changes via trivial diffusion (replacing thermal erosion passes)

**Decision**: Mountain weathering and hill flattening become a tiny diffusion applied each ecology tick — essentially imperceptible per ride, subtly visible over a year of riding. Replaces the batched thermal erosion pass.

**Why**: The rider never directly perceives a mountain losing 0.5m. But over a year of riding, the skyline subtly shifting contributes to the sense of deep time passing. This costs essentially nothing computationally and doesn't need physical accuracy — it's experiential, not geological.

### Decision: The three-tier architecture persists but boundaries shift

**Decision**: The tier structure remains, but what each tier does changes:
- **Geology**: Unchanged — static substrate, rarely ticks. Provides bedrock, base heightmap, soil parent material.
- **Climate-hydrology**: Splits internally. The slow *climate envelope* still drifts on century timescales. A new *weather system* generates seasonal conditions from overlapping cycles. Per-season lightweight erosion replaces batched passes. The derived-state cache becomes "current seasonal conditions" updated each season.
- **Ecology**: Seasonal ticks. Reads current weather conditions. Experiences hydrology effects through changed soil/water/sediment. This is where the bulk of overnight compute goes.

**Why**: The tier boundaries still represent real causal separation. What changes is the granularity of interaction — ecology and weather talk every season instead of ecology reading a static cache that updates every 1000 years. The event-based communication model still works; it just happens more frequently at the ecology-weather boundary.

## Species genome redesign: functional + morphological traits

**Context**: The seasonal ecology redesign requires expanding the species genome. The original 7-trait genome was mostly load-bearing but lacked seasonal strategy traits and had no visual representation pathway. The asset agent (Phase 10) needs enough morphological information to generate distinctive, consistent visual assets per species — and speciation needs to produce *visibly* different species, not just numerically different ones.

### Decision: Split genome into functional traits (simulation) and morphological traits (visual)

**Decision**: The genome contains two categories of traits. Functional traits affect simulation dynamics. Morphological traits affect visual asset generation and carry through speciation but have zero simulation cost. Some morphological traits are softly coupled to functional traits at initialization but can diverge independently during speciation.

**Why**: Without morphological traits, the simulation produces rich ecological dynamics that are invisible to the rider. Species could have fascinating distribution patterns, but if they all look the same, the rider can't perceive them. The asset agent needs a species description to generate visuals, and that description needs to come from the genome — not be hand-authored — so that speciation automatically produces visibly distinct daughter species. Morphological traits drift with higher variance than functional traits during speciation because visual distinctiveness is the rider's primary tool for reading the ecology. The soft coupling to functional traits ensures species *look* ecologically plausible (drought-adapted plants have small leaves) without locking visuals to function (two drought-adapted sister species can still have different flower colors).

### Functional traits (10 floats, all load-bearing)

| Trait | Range | Role |
|-------|-------|------|
| `drought_tolerance` | [0, 1] | Suitability in low soil moisture. Trades off against growth rate in moist conditions. |
| `frost_tolerance` | [0, 1] | Winter survival threshold. Drives warmth preference (inverse). |
| `shade_tolerance` | [0, 1] | Competitive ability under canopy. Trades off against growth rate in full sun. |
| `growth_rate` | [0, 1] | Speed of biomass accumulation. Trades off against lifespan and drought tolerance. |
| `seed_mass` | [0, 1] | High = large seeds (shorter dispersal, longer seed bank, larger seedlings). Low = small seeds (wider dispersal, shorter bank, wind-dispersed). |
| `max_height` | [0, ∞) | Canopy competition trait. Taller species cast shade on shorter ones. **Currently unused — becomes load-bearing in seasonal model.** |
| `lifespan` | [1, ∞) | Drives base mortality rate. Long-lived species resist displacement (landscape inertia). |
| `phenological_aggressiveness` | [0, 1] | **New.** How early the species leafs out relative to frost-safe conditions. High = spring ephemeral strategy (gains light, risks late frost). Low = conservative. |
| `evergreenness` | [0, 1] | **New.** Fraction of leaf area retained through winter. 1.0 = full evergreen (no spring startup cost, lower peak growth). 0.0 = full deciduous. |
| `mast_interval` | [1, 7] | **New.** Years between heavy seed crops. 1 = annual seeder. 5-7 = mast species (oaks). Creates pulsed recruitment cohorts. |

### Morphological traits (7 floats, visual-only, zero simulation cost)

| Trait | Range | Soft coupling | Notes |
|-------|-------|---------------|-------|
| `growth_form` | enum (0-4): tree, shrub, herb, grass, cushion | Derived from `max_height` + `lifespan` at creation | Primary visual category. Can drift during speciation. |
| `leaf_size` | [0, 1] | Loosely correlated with `shade_tolerance` (+) and `drought_tolerance` (−) | Small leaves = drought/wind adapted. Large = shade adapted. |
| `leaf_shape` | [0, 1] (needle → broad) | Correlated with `evergreenness` (evergreen trends needle) | Drives visual texture of vegetation areas. |
| `flower_color` | [0, 1] (mapped to hue wheel) | Fully independent | **The single most visible speciation marker.** Irrelevant when `flower_size` ≈ 0 (non-flowering species). |
| `flower_size` | [0, 1] (inconspicuous → showy) | Inversely correlated with `seed_mass` (wind-dispersed = small flowers); low for high `evergreenness` + needle-like species | At 0 = effectively non-flowering (grasses, conifers, ferns). Asset agent omits flowers entirely. |
| `bark_texture` | [0, 1] (smooth → rough) | Correlated with `lifespan` | Long-lived species develop thicker, rougher bark. |
| `stem_woodiness` | [0, 1] (herbaceous → woody) | Correlated with `growth_rate` (−) and `lifespan` (+) | Fast-growing short-lived = herbaceous. Slow long-lived = woody. |

### Initialization and speciation behavior

**Initialization**: Morphological traits are derived from functional traits via soft coupling formulas plus a random offset. A drought-tolerant species *tends* toward small leaves but might not be. Non-flowering species emerge naturally: high `evergreenness` + needle-like `leaf_shape` → low `flower_size`.

**Speciation drift**: Morphological traits mutate with **higher variance** than functional traits. Two daughter species may converge on similar functional niches (same altitude, same moisture) but diverge visually. The rider sees "those are clearly related but different" — purple flowers on one side of the ridge, yellow on the other — which is how real speciation looks and is the key to the rider perceiving ecological dynamics.

**Asset agent input**: The full trait vector becomes the species description prompt. Example: *"A medium shrub (1.2m), semi-evergreen, small needle-like leaves, showy purple flowers, rough bark, moderately woody stems. Grows in cold dry highlands."* When a species speciates, the child's shifted morphological traits produce a visibly related but distinct asset.

### New per-cell state buffers

| Buffer | Type | Purpose |
|--------|------|---------|
| Cumulative drought stress | float per cell | Rolling ~3-year water deficit. Multi-year droughts produce gradual forest decline, not instant switching. |
| Biomass/establishment age | float per species per cell | How long a population has been established. Older stands resist displacement — landscape inertia. Makes old-growth feel anchored. |
| Seed bank | float per species per cell | Already exists. Gains seasonal timing: seeds produced in fruiting season, not continuously. |

### What's discarded from current model

- 5-year tick structure and all rates calibrated to it
- Static climate cache model (replaced by per-season weather)
- Stochastic noise addition (0-0.001 per occupied cell — does nothing)
- `distance_to_water` in suitability (loaded but never used; should become load-bearing in the new model)

### Decision: Ride mapping — 1 season per minute

**Decision**: Each minute of riding advances 1 simulated season (0.25 years). A 30-minute ride advances ~7.5 years. Configurable via `seasons_per_minute` setting for frequent tuning.

**Why**: Much more gradual than the previous 1-year-per-minute mapping. The rider sees subtle changes per ride, meaningful shifts over weeks of riding, and long-cycle effects over months. Varying ride lengths mean the rider doesn't always land on the same season. The slower pace supports the continuity-with-change balance.

### Decision: Two-phase world creation (deep history + seasonal recent history)

**Decision**: World creation uses coarse ticks (existing batched system) for deep history (~200M years geology, ~200K years climate, ~10K years coarse ecology), then switches to seasonal ticks for the final ~1,000 years. The seasonal system is only used for the part of history the rider actually experiences evolving.

**Why**: 200M years at 4 ticks/year = 800M ticks, which is impossible. Deep history just needs to produce plausible terrain and species radiation — it doesn't need seasonal fidelity. The transition to seasonal ticks for recent history ensures the world enters play with realistic seasonal patterns, seed banks, and weather-cycle-driven vegetation.

**Full implementation spec**: `docs/seasonal-redesign-spec.md`

## Ecology stabilization: self-correcting feedback loops

**Context**: After the seasonal ecology redesign, calibration runs revealed two critical problems. First, 3 of 6 ancestor species couldn't establish because initial population seeding used winter weather (tick 0 = winter), giving warm-adapted species zero suitability. Second, once ancestor templates were tuned and seeding was fixed, a single ancestor (lowland herb) exploded to 184 species in 850 years, producing an O(n²) competition matrix that took 8+ hours and consumed 6.7GB of memory. The system lacked any mechanism to check runaway dominance or prevent cascade micro-speciation.

These problems prompted a broader discussion about what "balance" means for the project. The goal is not to engineer a particular equilibrium, but to build a system with self-correcting tendencies that can recover from dramatic perturbations. A lowland herb bloom is fine — it's the Carboniferous fern forests — as long as the world has mechanisms to eventually check it. "A world that can die is one that matters" (from initial design docs), and a world out of balance is interesting as long as it can recover.

### Decision: Annual-mean seeding for initial populations

**Decision**: Initial population placement uses the base climate layers (annual mean temperature and precipitation) rather than the current season's weather.

**Why**: Tick 0 is winter. Winter conditions give warm-loving species (lowland herb, valley tree) zero suitability across the entire grid. They're seeded with zero density and immediately fail the MVP check. Using annual means represents "where could this species potentially persist over a full year" — the ecologically correct question for initial placement.

### Decision: Ancestor template tuning to match environmental range

**Decision**: Adjusted three ancestor templates whose trait optima fell outside the world's actual environmental range:
- Lowland herb: drought_tolerance 0.15 → 0.30
- Valley tree: drought_tolerance 0.15 → 0.30
- Alpine cushion: drought_tolerance 0.85 → 0.65, frost_tolerance 0.95 → 0.85

**Why**: Diagnostic analysis showed the world's drought stress range is 0.30–0.68 and temp norm range is 0.05–0.80. Species with optima outside these ranges get near-zero suitability everywhere. The adjustments are minimal — each species remains the most extreme specialist on its axis — but now falls within the range where viable habitat exists.

**Note**: Alpine cushion still goes extinct in calibration runs. Its niche (cold + dry) is the smallest habitat type in this world. This may be acceptable — not every ancestor needs to succeed in every world — or may indicate the world generation needs more extreme alpine habitat.

### Decision: Seed-mass-driven long-distance dispersal

**Decision**: Replace the flat 5% long-distance dispersal chance with a seed_mass-scaled mechanism. Light seeds (low seed_mass) get frequent long-range jumps (up to 50% of grid); heavy seeds get rare short jumps. Seeds are sourced from high-density cells and deposited proportionally.

**Why**: The primary cause of cascade speciation was false fragmentation — a species covering 80% of the map has dozens of tiny gaps that register as separate connected components, each becoming a new species. Long-distance dispersal bridges these gaps by maintaining gene flow across habitat discontinuities. When speciation does happen, it means populations are genuinely isolated by inhospitable terrain, not just separated by a few empty cells. This also makes the `seed_mass` trait load-bearing — it now creates a real evolutionary tradeoff between dispersal range and establishment success.

### Decision: Janzen-Connell biotic pressure (oscillating top-down mortality)

**Decision**: Add a hidden "biotic pressure" variable per species that serves as a proxy for pathogens, herbivores, and fungi. Pressure accumulates when a species is abundant (total density > baseline), decays when rare, and is shared among close genetic relatives (genome distance < threshold). Applied as density-dependent mortality each tick.

**Why**: The simulation lacked any top-down pressure on dominant species. In real ecosystems, monocultures attract specialized pathogens and herbivores that check their growth (the Janzen-Connell effect). Without this, a fast-growing generalist can fill every niche and fragment into hundreds of near-clone species. The hidden variable approach avoids modeling explicit herbivore agents while producing the key dynamic: boom-bust oscillations where dominant species build pressure, crash, then recover as pressure decays. Close relatives share pressure (representing shared pathogen pools), creating selection pressure for genetic divergence — species that diverge escape the pathogen shadow.

**Key design choice**: Pressure is not a constant — it oscillates. This was a deliberate decision after discussing that top-down forces in nature have their own dynamics. A constant Janzen-Connell multiplier would produce stable equilibria; the oscillating hidden variable produces the cycles of dominance and decline that make ecosystems dynamic.

**Parameters (tuned for 1000×1000 grid)**:
- `baseline_density = 20,000` — pressure only builds above this (a healthy species has 10K–50K total density)
- `growth_k = 0.0005` — slow accumulation to prevent immediate kills
- `decay_rate = 0.95` — 5% decay per tick when species is sparse
- `mortality_strength = 0.08` — max 8% mortality per tick at full pressure
- `relatedness_threshold = 0.5` — genome distance below which pressure is shared

**Storage**: One float per species in a `biotic_pressure` table in EventStore. Negligible cost.

### Decision: Speciation threshold tuning

**Decision**: Raised speciation thresholds significantly:
- Minimum parent occupied cells: 100 → 500
- Minimum fragment size: 50 → 200 (downsampled cells)
- Speciation probability per eligible fragment: 0.3 → 0.15
- Added 100-year cooldown: species must be ≥100 years old to speciate

**Why**: The previous thresholds allowed a species to fragment into 10+ new species per check, each of which could fragment again 50 years later. The 100-year cooldown prevents cascade speciation — young species need time to establish their identity before fragmenting again. Higher minimum sizes ensure speciation events represent meaningful population isolation, not noise.

**Result**: Calibration runs produce ~40 species over 1000 years (vs 244 previously), with healthy turnover (extinctions + new speciations). Runtime dropped from 8+ hours to ~90 minutes.

### Future directions discussed

Several ideas were discussed for future implementation but not yet built:

1. **Climate-responsive speciation rates**: Lower speciation thresholds during extreme climate events, producing burst speciation when cycles align to create dramatic conditions. The weather system's overlapping cycles would naturally produce these rare windows.

2. **Climate-responsive biotic pressure**: Modulate pressure growth/decay based on weather conditions (warm wet = pathogen-friendly = faster pressure buildup; cold dry = pressure decay). Creates windows where monocultures are temporarily safe, followed by crash periods.

3. **Rare extreme climate configurations**: The current weather system uses regular sinusoidal cycles that don't produce truly extreme events. Adding long-period cycles (500-1000 years) with different phases would create rare constructive interference events — the engine for mass extinctions and speciation booms.

4. **Lineage age effects on pressure**: Young species haven't accumulated their full pathogen/herbivore load ("enemy release" hypothesis). Could give new species a natural honeymoon period.

These form a coherent system: rare extreme climate → habitat fragmentation + pressure collapse → speciation burst → new species fill altered landscape → pressure rebuilds → new equilibrium. The full cycle of disruption and recovery that makes worlds feel alive.

## Fractal climate noise + climate-responsive biotic pressure

**Context**: The sinusoidal weather cycle system produced periodic, predictable oscillations — every rare event recurred on a fixed schedule. Extended calibration runs (24+ hours, 508 species) confirmed that the system lacked the climate variability needed to drive meaningful perturbation/recovery dynamics. Additionally, biotic pressure used fixed parameters regardless of environmental conditions.

### Decision: Replace sinusoidal cycles with 1D fractional Brownian motion (fBm)

**Decision**: The weather system now evaluates 7 octaves of 1D value noise instead of 8-9 overlapping sine waves. Two independent noise streams (temperature, precipitation) are seeded from the world seed via hash-based deterministic noise.

**Why**: Fractal noise produces aperiodic variability with a red-noise power spectrum — low frequencies dominate (century-scale trends), with progressively smaller high-frequency variation overlaid. This means:
- Every climate epoch is unique; nothing repeats
- "Trends within trends" — zoom into any 200-year window and it shows its own internal structure
- Rare extreme alignments emerge naturally from the mathematics, without being engineered
- The power spectrum matches real climate data far better than superposed sinusoids

Real climate data shows fractal structure, not discrete frequency peaks. No matter how many sine waves are overlaid, the result repeats at the LCM of all periods. The fractal approach eliminates this fundamental limitation.

**Parameters**: base_freq=1/800 (lowest octave ~800yr), lacunarity=2.0, persistence=0.55, base_amp_temp=3.0°C, base_amp_precip=0.15 (log-space). Safety caps widened to ±8°C temp, [0.25, 3.0] precip multiplier (rarely hit — natural falloff from persistence keeps anomalies bounded).

**Verification**: 5000-year trace for seed 42 shows mean temp anomaly -1.26°C (std 1.59), max year-to-year change 0.043°C. Each 200-year window has distinct character. Caps never hit.

### Decision: Climate-responsive biotic pressure

**Decision**: The Janzen-Connell biotic pressure system now responds to weather conditions. A "pathogen favorability" score (0-1) is computed from mean temperature × mean precipitation each tick. This modulates `growth_k` (50-150% of base) and `decay_rate` (faster decay in cold/dry conditions).

**Why**: In real ecosystems, pathogen and herbivore pressure varies with climate. Warm wet conditions favour disease; cold dry conditions suppress it. With fixed pressure parameters, the system reaches a steady oscillation. With climate-responsive parameters, the dynamics become richer:
- **Warm wet period**: pressure builds faster → dominant species crash sooner → more turnover
- **Cold dry period**: pressure decays → dominant species get a reprieve → temporary expansion
- **Cold→warm transition**: species that expanded during cold window suddenly face pressure → potential crash
- **Warm→cold transition**: pressure drops but species already depleted → slow recovery

This creates asymmetric dynamics around climate transitions — perturbation followed by recovery into a new equilibrium, which is the core dynamic the project is designed to produce.

## Speciation rate-limiting: genetic divergence + niche saturation

**Context**: Extended calibration revealed a fundamental structural problem with fragmentation-based speciation. A widespread species has tiny gaps in its distribution on the 1000×1000 grid; each gap registers as a separate fragment; each fragment that passes the probability check becomes a new species; each new species fragments again. This cascade produced 508 species from 6 ancestors in a 24-hour run, with O(n²) competition scaling making each tick progressively slower. The speciation threshold tuning from the ecology stabilization work (min fragment 200, min occupied 500, 15% prob, 100yr cooldown) slowed the cascade but didn't address the root cause: spatial separation alone, without meaningful genetic divergence, shouldn't produce new species.

### Decision: Minimum genetic divergence threshold for speciation

**Decision**: After computing a new genome for a candidate daughter species, measure the Euclidean genome distance (in functional trait space) between the daughter and parent genomes. If the distance is below a threshold (0.15), reject the speciation — the fragment stays part of the parent species.

**Why**: This is the most mechanistically honest solution. Geographic separation alone doesn't drive speciation in nature — it's geographic separation *plus different selection pressures* that does it. A fragment on a cold ridge diverges from a fragment in a warm valley because their genomes are being pulled in different directions by local conditions. The existing adaptive drift mechanism already biases genome mutation toward local environmental optima (strength 0.3). This means:

- Fragments in environments *similar* to the parent's range get small, similar mutations → low genome distance → speciation rejected → fragment stays as part of parent population
- Fragments in *different* environments get directional mutations toward local optima → higher genome distance → speciation accepted → genuinely distinct species

**Critical interaction with fractal climate**: Climate disruptions become the primary speciation engine through two mechanisms:
1. **Creating separation**: A harsh climate epoch kills a species in marginal habitat, splitting one continuous population into isolated fragments
2. **Driving divergence**: Fragments experience different local conditions (temperature, drought, frost), so their genomes drift apart through differential selection

When conditions improve, the now-distinct species re-expand and overlap — real sympatric diversity that emerged from process, not from a probability roll. This means speciation rate naturally accelerates during and after climate disruptions (exactly when it should) and naturally slows during stable periods (when populations are continuous and fragments don't diverge).

**Threshold choice (0.15)**: At the current drift sigma of 0.08 per functional trait (7 traits), a single speciation event produces an expected genome distance of roughly `sqrt(7) * 0.08 ≈ 0.21`. But this is the *average* — many events will produce less divergence. The 0.15 threshold rejects the bottom ~30% of speciation attempts where the daughter barely differs from the parent. Fragments under genuinely different selection pressure (where adaptive bias adds to random drift) easily clear the threshold. This is tuneable.

**World filling dynamics**: This threshold naturally allows rapid speciation during early world history (when habitats are empty, fragments colonize diverse environments and diverge quickly) and slows speciation as niches fill (when most fragments are in environments similar to the parent's optimum). This addresses the concern about needing fast speciation during world creation but not wanting runaway speciation during normal play.

### Decision: Soft niche saturation scaling

**Decision**: Speciation probability scales inversely with total alive species count: `prob *= max(0.1, 1.0 - alive_count / 100.0)`. At 50 species, probability is halved. At 90+, it's 10% of base rate. Never reaches zero.

**Why**: Complements the genetic divergence threshold with a system-level brake. Even with the divergence requirement, a sufficiently heterogeneous landscape could sustain high speciation rates indefinitely. Niche saturation represents the ecological reality that finite landscapes have finite niche space — as species count rises, the remaining unfilled niches are smaller and more marginal. The soft scaling (never reaching zero) preserves the possibility of speciation even in a "full" world — if conditions are extreme enough to create genuinely novel habitat, a new species can still emerge.

**Not a hard cap**: A global maximum species count was considered and rejected as too artificial. The saturation scaling achieves the same practical effect (asymptotic slowdown) while remaining responsive to conditions. A world with 80 diverse species in genuinely distinct niches can still speciate; a world with 80 near-clone species finds it very difficult.

**Combined effect**: The two mechanisms work at different levels. Genetic divergence is a per-event gate (is this particular fragment genuinely distinct?). Niche saturation is a population-level brake (how full is the world?). Together they produce the desired behavior: rapid early radiation → gradual slowdown → climate-disruption-driven bursts → new equilibrium.

### Outcome: rate-limiting alone was insufficient

Calibration v5 (1050 years) showed that genetic divergence + niche saturation slowed speciation but didn't change the fundamental dynamic. Lowland herb still reached 67 of 92 species, with speciation *accelerating* (13.5/century by year 1000). The herb evolved to escape our pressure strategy: each daughter sat just below the baseline_density threshold (~18.7K vs 20K baseline), avoiding pressure accumulation. Pressure inheritance (v6, 60% of parent pressure inherited by daughter) also failed to change the trajectory.

The root cause was structural: the fragmentation mechanic rewarded spatial dominance with more speciation opportunities. A generalist covering 83% of the map always has spatial gaps — but those gaps are suitable habitat, not barriers. Success → fragmentation → speciation was backwards.

## Barrier-based speciation: environment-driven isolation

**Context**: After three calibration runs showed that parameter tuning couldn't fix the speciation cascade, we stepped back to rethink the mechanic itself. The core insight: geographic separation alone doesn't drive speciation — it requires prolonged isolation across an environmental barrier under different selection pressures.

### Decision: Require inhospitable terrain between fragment and main population

**Decision**: After finding a fragment via connected components, compute suitability for the species on the downsampled grid. BFS flood fill from the fragment through hospitable cells (suitability > 0.2). If the flood reaches the main population, reject the speciation — the gap is traversable habitat, not a barrier.

**Why**: A species covering 80% of the map has dozens of tiny gaps that register as fragments. But those gaps are perfectly good habitat the species just hasn't filled yet — dispersal will bridge them. A real barrier is a mountain ridge with near-zero suitability, a drought zone, a frost belt. These are environmental features that the species genuinely cannot cross. By requiring barriers, speciation becomes driven by the environment (terrain, climate) rather than by spatial noise in the density grid.

**Critical interaction with fractal climate**: This makes climate the primary speciation engine:
1. Wet period → species expands continuously across lowlands → no barriers → no speciation
2. Dry period → drought zone emerges, splitting habitat → populations isolate behind barrier
3. Centuries of isolation under different conditions → genomes diverge
4. Wet period returns → barrier dissolves → two now-distinct species overlap and compete

The fractal noise ensures these barrier-creating events are aperiodic and unique — every world's evolutionary history reflects its specific climate epochs.

**Emergent seed_mass tradeoff**: Light-seeded species (herbs, grasses) have high long-distance dispersal that bridges barriers, naturally *preventing* their speciation. Heavy-seeded species (trees) can't jump barriers and are more likely to genuinely isolate. This produces realistic patterns without parameter tuning — wind-dispersed species have wider ranges and lower speciation rates, while heavy-seeded species show more local endemism.

### Calibration v7 results (seed 42, 4000 years)

| Year | Species | Herb | Tree | Others |
|------|---------|------|------|--------|
| 400 | 8 | 3 | 1 | 4 |
| 800 | 11 | 4 | 3 | 4 |
| 2000 | 15 | 8 | 3 | 4 |
| 2400-2800 | 15 | 8 | 3 | 4 |
| 4000 | 18 | 8 | 6 | 4 |

- **Zero extinctions** across 4000 years. All 6 ancestors have living descendants including alpine cushion.
- **Herb stabilized at 8 species** since year 2000 — no runaway.
- **Valley tree** is the late-game speciation story (1→3→6), driven by real terrain barriers. Heavy seeds can't bridge gaps, so tree populations genuinely isolate.
- **Speciation rate**: 8 in first 800yr, then 1/400yr, then 0 for 800yr, then a burst of 2 trees at year 4000. Exactly the pattern of asymptotic decline with rare climate-driven bursts.
- **Runtime**: 7.2 hours for 4000 years (vs 5+ hours for 1000 years in v5 with 89 species).

Compare v5 (no barrier check): 89 species at year 1050, 67 herbs, accelerating speciation, 508 species after 24 hours of compute with no sign of stabilizing.

### Gene flow / species reabsorption (implemented 2026-06-14)

**Decision**: When two closely-related species overlap spatially and no barrier separates them, the smaller population is absorbed into the larger one with gene flow. This is the inverse of speciation.

**Why**: The v7 calibration (4000yr) showed zero extinctions and 8 frozen herb species — "zombie pairs" that were functionally identical but grinding through competitive exclusion indefinitely. Lotka-Volterra competitive exclusion takes 50-100 years for near-identical species, which is too slow when climate shifts, disturbance, and LDD keep resetting the contest. Without reabsorption, species accumulate monotonically.

**Mechanism**:
- Checked every 200 ticks (50 years), same cadence as speciation, run immediately after
- **Trigger conditions** (all must be true):
  - Genome distance < 0.25 (below niche_width of 0.3, strong competition)
  - Spatial overlap > 20% of the smaller species' range
  - No barrier separating them (BFS through hospitable terrain, same check as speciation)
  - Both species at least 100 years old (prevents immediate post-speciation reabsorption)
- **Merge mechanics**:
  - Smaller population absorbed into larger
  - Densities added together in all cells
  - Absorber's genome shifts toward absorbed species, weighted by population ratio × 0.5 (gene flow / adaptive introgression)
  - Biotic pressure merged as weighted average
  - Absorbed species marked extinct, density and seed bank zeroed
- **Emergent properties**:
  - Fires more often for herbs (light seeds bridge barriers via LDD → overlap → reabsorption)
  - Fires less for trees (heavy seeds stay isolated → barriers persist → genuine speciation)
  - Makes speciation a two-way door: briefly isolated populations merge back with slight adaptation
  - Only populations that diverge past 0.25 genome distance become permanent species

### Climate-ecology coupling audit and fixes (2026-06-14)

**Decision**: Strengthen climate→ecology signal by fixing three independent damping points identified by a deep audit of the v7 calibration data.

**Why**: An audit of the 4000-year v7 calibration revealed that the climate system was essentially decorative. All species converged to ~20,000 total density (coefficient of variation <4%) regardless of climate conditions. Five independent damping points each suppressed the climate signal. Three were fixable without adding new mechanics.

**Fixes applied**:

1. **Fixed drought stress self-cancelling baseline (R4)**: Drought deficit was computed as `1 - precip / precip.mean()`, but since the fractal multiplier shifts all cells equally, the mean IS the shifted value — deficit was always ~zero. Fixed by using the base climate precipitation from world creation (the terrain-modulated envelope without weather anomalies) as the reference.

2. **Climate-modulated biotic pressure baseline (R1)**: The fixed `baseline_density = 20000` acted as a thermostat pinning every species at the same density regardless of climate. Replaced with a per-species baseline that scales with mean suitability under current weather: `baseline = 10000 + 20000 * mean_suitability`. In favorable climate, species can grow larger before pathogens build up; in unfavorable climate, pathogens kick in earlier. This makes biotic pressure amplify climate signal rather than override it.

3. **Narrowed suitability Gaussian sigma (R3)**: Reduced σ from 0.2 to 0.14 for drought and temperature match functions. At σ=0.2, a century-scale climate swing (4.5°C) only dropped suitability by ~25% — too gentle to drive range shifts. At σ=0.14, the same swing drops suitability to ~7%, making climate variability a dominant force in species dynamics.

**Rejected alternatives**:
- **R2 (spatial fractal noise)**: Terrain already provides spatial differentiation (valley amplification, orographic precip, rain shadows). The issue was sensitivity, not spatial structure. Adding artificial spatial noise would violate the "process over placement" principle.
- **R5 (discrete climate catastrophe events)**: The existing frost kill, drought mortality, and spring frost damage mechanics should produce visible events naturally once sensitivity is high enough. Adding a separate event system would be artificial.

**Design principle**: These fixes unblock existing mechanisms rather than adding new ones. The climate system, terrain modulation, suitability computation, and biotic pressure were all correctly designed — they just had parameter/baseline choices that independently suppressed the signal at each stage.

## Two-layer ecology and spatial climate variation (2026-06-15)

**Context**: Calibration v9 (4000 years) revealed that while climate-ecology coupling was working (genuine boom-bust cycles), the world was almost completely empty. Dense pockets of vegetation in a mostly barren landscape. The colonization model — 6 point-source species expanding via dispersal — could never fill a world because suitability filtering kills propagules faster than dispersal establishes them. Real Earth had billions of years for colonization; simulating thousands and wondering why it's empty is a structural mismatch.

Additionally, the world lacked spatial climate variation beyond elevation. Every valley felt the same. Climate cycles shifted all cells uniformly, preventing range migration and the compelling narratives (species bridging barriers, getting cleaved by returning deserts) that make worlds feel alive.

### Decision: Spatial climate bias fields

**Decision**: Generate two static 2D noise fields at world creation — moisture_bias [0.5, 2.0] and continentality [0.0, 1.0]. These multiply/scale the dynamic weather output spatially. Moisture_bias correlates with elevation (lower = wetter) for natural rain shadow effects.

**Why**: Creates distinct wet/dry and maritime/continental regions without hard biome boundaries. The fields are continuous, so region borders are fuzzy ecotones that shift with global climate cycles. During a wet period, dry-biased areas may become temporarily habitable; during drought, they become barriers. This enables the desert-bridge narrative to emerge from rules rather than being scripted.

**R2 revisited**: The earlier audit rejected spatial noise because "terrain already provides differentiation." Extended calibration proved this insufficient — terrain modulates climate but doesn't create the distinct regional character needed for a BotW-like landscape. The spatial bias fields are not artificial noise added to existing signals; they represent real geographic features (distance from coast, mountain sheltering, prevailing wind effects) that the simplified terrain model doesn't capture.

### Decision: Two-layer ecology (ground cover + canopy)

**Decision**: Split ecology into two layers. Ground cover (grass, moss, lichen, bare soil) is computed directly from current weather conditions — always present, changes with seasons, zero simulation cost. Canopy (trees, shrubs, structural plants) is the existing competitive ecology with dispersal, speciation, and population dynamics.

**Why**: The world must never be empty — every cell needs vegetation for visual quality. Making ground cover a simulation output that can be zero created a problem that didn't need to exist. Ground cover in real ecosystems is a fast-responding function of conditions, not a dispersal-limited population. Separating it from canopy ecology lets us guarantee full coverage while preserving the interesting dynamics for structural plants.

**Rendering alignment**: Ground cover is a texture (free). Canopy is geometry (expensive). The simulation should match the rendering cost model.

### Decision: 14 ancestral species with broad placement (replacing 6 point-sources)

**Decision**: Expanded from 6 to 14 ancestors across 4 structural roles: 3 canopy trees, 2 understory trees, 4 tall shrubs, 5 low shrubs/forbs. Species are placed broadly at creation using suitability × species-specific noise, not from point sources.

**Why**: With 6 point-sources, the world started empty and required thousands of simulated years to fill — which it never fully did because suitability filtering prevented colonization of marginal habitat. With 14 species scattered across suitable habitat, the world starts 100% covered. Structural roles ensure different rendering costs and ecological strategies coexist. Species-specific noise creates natural clustering (groves, patches) without placement.

**"Generation = advancement" preserved**: The same dispersal/competition rules apply during both world creation settling and ride-time play. The broad initial scatter is the equivalent of saying "this world has been inhabited for millions of years." The settling period lets competition sort out overlaps, producing a world that's near equilibrium but with enough roughness for the rider to witness ongoing dynamics.

### Decision: Speciation as rare narrative event (not niche-filling mechanism)

**Decision**: Speciation is now rarer (~500yr average), more dramatic, and role-dependent. Cooldown raised from 100yr to 300yr. Base probability 0.15 → 0.08. Trees speciate at 2x rate, herbs at 0.5x. Morphological drift increased (σ 0.15 → 0.25) so daughter species are visibly distinct.

**Why**: In the old model, speciation was load-bearing infrastructure — the only way to fill niches. With broad placement and 14 ancestors, that function is no longer needed. Speciation becomes a world event the rider might witness over months of riding: "there's a new kind of tree on the far side of that ridge." Trees speciate most readily because their heavy seeds can't bridge barriers; herbs rarely speciate because wind dispersal maintains gene flow.

### Decision: Deterministic ghost species extinction

**Decision**: Species with zero density AND zero seed bank are immediately marked extinct, rather than relying on probabilistic MVP checks.

**Why**: v9 calibration showed 8 "alive" species with zero density — ghosts that survived probabilistic extinction checks indefinitely. Species that are truly gone should be formally marked extinct.

### Full design document

See `docs/ground-cover-and-spatial-climate.md` for the complete design including implementation phases, open questions, and rendering considerations.
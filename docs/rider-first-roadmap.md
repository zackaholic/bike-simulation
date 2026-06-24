# Rider-First Roadmap

The simulation ecology has reached a good place. But the world it runs on is
barely rideable — 0–2000m elevation range, jagged terrain, no carved valleys.
The A* pathfinder can find routes, but they're circuitous detours around
impassable slopes. More fundamentally, **altitude is the only source of
ecological variation**, so experiencing different biomes requires changing
elevation — which means steep grades — which means unrideable.

This roadmap reorients the project around the rider experience: get a
minimum viable ride working in the renderer, then iterate on terrain and
simulation to make that ride ecologically rich.

## The Core Problem

The climate pipeline derives everything from altitude:
- Temperature = latitude gradient + lapse rate (6.5°C/1000m)
- Precipitation = base + orographic effects (altitude-driven)
- Result: temp/precip correlation is ~−0.50 (cold = wet, warm = dry)

This means biome transitions require elevation changes. But elevation changes
= steep grades = unrideable. **Rideability and ecological diversity are in
direct conflict.**

### The Fix: Decouple Climate from Altitude

Promote 2D noise fields to the primary climate drivers. Altitude becomes a
subtle secondary modifier.

- **Temperature**: primary driver = low-frequency 2D noise field ("climate
  zones"), lapse rate reduced to ~1–2°C/1000m (flavor, not structure)
- **Precipitation**: primary driver = separate 2D noise field, orographic
  effect reduced to a gentle boost
- The existing `moisture_bias` and `continentality` spatial fields are
  already the right shape — promote them from modifiers to primary drivers

This lets flat terrain have rich ecological variation. A rider on rolling hills
passes through distinct climate zones → distinct species → biome transitions,
without needing to climb mountains. Mountains still exist (pretty, ecologically
interesting skyline features) but aren't required for diversity.

## Reference: Previous Godot Prototype

The previous project (`bike-trainer-godot/`) had a working chunk-based
renderer with terrain, cel-shaded vegetation, and camera path following. Its
terrain was too flat (amplitudes 25/8/2m, ~35m total range) — the opposite
of the current sim's problem. The sweet spot is between the two.

### Reusable patterns and hard-won fixes

These save significant implementation time:

- **Chunk binary format** (`.chunk` files + `manifest.json`): header, terrain
  mesh (vertices/normals/UVs/colors/indices), MultiMesh asset transforms,
  JSON metadata. Custom `ResourceFormatLoader` in Godot parses them.
- **Chunk streaming**: async loading with 300m lookahead / 200m cull,
  `ResourceLoader.load_threaded_request()` with per-frame polling.
- **Camera grounding**: raycast downward + exponential smoothing
  (`1 - exp(-speed * delta)` for frame-rate independence), 1.5m eye height,
  0.25m hard floor to prevent terrain clip.
- **Arc-length parameterized Bezier curves**: `BakedCurve` class matching
  Godot's `Curve3D` convention exactly. Constant-velocity sampling avoids
  speed-dependent camera jitter.
- **Parametric asset builder**: assets as JSON specs (cylinder primitives
  with radius/height/color/rotation), assembled from `CylinderMesh` at load
  time. Guarantees correct normals, winding, and caps by construction.
- **Vertex-color splatmap**: RGBA encodes surface type blend weights
  (R=rock, G=dry scrub, B=wet grass, A=temperature tint).
- **Poisson disk placement**: quality-map-filtered sampling with size-class
  ordering, cross-asset spacing, and dominance falloff.

### Critical gotchas (documented solutions)

| Issue | Symptom | Fix |
|-------|---------|-----|
| AABB culling on anchor-relative meshes | Terrain vanishes at horizon | Set `custom_aabb` on BOTH `MeshInstance3D` node AND `ArrayMesh` |
| Triangle winding order | Terrain invisible from above, visible from below | CCW winding when viewed from above: `(r,c), (r+1,c), (r,c+1)` |
| Inverted-hull outlines on hollow geometry | Objects appear solid black | Always use `CylinderMesh` with caps, never hand-built hollow shapes |
| Float32 precision on long paths | Geometry jitters far from origin | Store vertices relative to chunk anchor |

### What doesn't carry over

- The specific terrain noise config (too flat)
- The cel-shading aesthetic (undecided — may stay, may go)
- The static biome/placement rules (replaced by ecology-driven placement)
- The world builder pipeline (replaced by bike-sim extractor reading
  `WorldQuery`)

### Godot docs MCP server

An FTS5 search tool over the full Godot 4 documentation, served as an MCP
server. Invaluable for rendering/performance research questions. Config added
to `.mcp.json` in this project, pointing to the existing server at
`/Users/zack/Projects/Bike-Trainer/godot_docs/mcp_server.py`.

## Phase Sequence

### Phase R1: Terrain in Godot (no sim changes)

**Goal**: See the current world from cycling perspective. Understand what's
wrong viscerally, not analytically. Terrain only — no plants.

**Python extractor** (new module in `src/bike_sim/extract/godot/`):
- Reads heightmap from `WorldQuery`
- Generates bike path (reuse `ride_experience.py` A* pathfinder)
- Builds chunk meshes along the path: terrain grid with vertex colors from
  elevation (simple gradient — green low, brown high, grey peaks)
- Outputs `.chunk` binary files + `manifest.json` in the format the previous
  Godot project already consumes

**Godot scene** (adapted from previous project):
- Chunk streamer loads terrain chunks
- Camera follows path via `PathFollow3D` + ground raycast
- No vegetation, no water — terrain + sky only

**Done when**: You can "ride" through the current world and feel the grades,
see where the path detours, understand what needs to change.

### Phase R2: Rideable Terrain

**Goal**: Iterate on terrain generation until a 30km loop feels good at
cycling speed.

**Approach**: Change geology noise parameters. The current heightmap uses 6
octaves with weights [32,16,8,4,2,1] normalized to 0–2000m. Directions to
explore:

- **Lower base elevation range** (0–400m for rideable terrain) with sparse
  dramatic features (mountain ridges/peaks reaching 800–1200m via a separate
  noise layer or threshold)
- **Nonlinear transfer function** on one heightmap — compress the middle
  range (where riding happens), stretch the peaks
- **Different noise profile** — gentle rolling base with occasional sharp
  ridgeline features, rather than uniform octave weight distribution

The ride path sticks to the lowlands; mountains are scenery.

**Iteration loop**: Change noise params → export heightmap → ride in Godot →
adjust. Fast cycle, no ecology involved. The existing A* pathfinder gives
feedback on path quality.

**Done when**: A 30km loop with <8% average grade, <15% max grade, with
visible mountains on the horizon and gentle rolling terrain underwheel.

**Note**: R2 and R3 may need some back-and-forth iteration, since the "right"
terrain depends partly on what climate variation is achievable without altitude,
and vice versa.

### Phase R3: Climate Decoupling

**Goal**: Ecological variation without altitude dependence.

**Changes to the climate pipeline**:
- Temperature: 2D noise field as primary driver, lapse rate reduced to
  ~1–2°C/1000m
- Precipitation: 2D noise field as primary driver, orographic effect reduced
  to gentle boost
- The temp/precip anticorrelation breaks — warm-wet and cold-dry regions
  exist naturally
- The climate space becomes genuinely 2D, not a forced 1D diagonal

**Ecology recalibration** (rules unchanged, parameters refit):
- Suitability references refit to the new climate envelope
- Genome redistribution simplifies (manifold is 2D, all niches reachable)
- Carrying capacity formula adjusted (currently moisture × elevation →
  moisture-based or soil-quality noise)
- Rerun equilibrium + perturbation tests on the new landscape
- Competition, dispersal, speciation, biotic pressure mechanics: unchanged

**Done when**: 14 species establish across the terrain with visible biome
boundaries that a rider crosses during a 30km loop.

### Phase R4: Species in Godot (colored cylinders)

**Goal**: See ecological structure from cycling perspective.

**Rendering approach**:
- **Ground cover** (grass, moss, lichen, bare soil): terrain texture blending
  driven by ground cover type/vigor from the sim. Ground cover is always
  present (the two-layer ecology ensures no bare terrain).
- **Canopy species** (trees, shrubs, structural plants): colored cylinders.
  Each species gets one distinct color. Cylinder height = `max_height` genome
  trait, width proportional. Positions stochastically sampled from density
  field per cell. This gives a feel for what the terrain will look like to
  pass through — dense tall cylinders feel like forest, sparse short ones
  feel like scrubland.

**Extractor additions**:
- Sample species density in a corridor around the ride path (visible range)
- Per species: Poisson disk → cylinder transforms at density-appropriate
  spacing. Reuse the placement engine pattern from the previous project.
- Pack cylinder transforms as MultiMesh data in chunk format

**Godot additions**:
- One `MultiMeshInstance3D` per species per chunk (colored cylinder mesh,
  trivially cheap — one-primitive parametric asset)
- Ground cover as shader-driven texture blending on terrain mesh

**Done when**: Riding through the world, biome transitions are visible as
color/height changes in the cylinder forest. Dense tall green cylinders give
way to sparse short red ones as climate zones shift. The experience of passing
through a dense stand vs open ground is legible at cycling speed.

### Phase R5: Replace Cylinders with Plant Assets

L-system / asset agent work. Out of scope for this plan but slots in cleanly:
the cylinder positions and density sampling pipeline are the same code that
feeds real meshes later. The visual genome (morphological traits) drives
L-system generation, producing species-specific assets that replace the
placeholder cylinders.

## What Gets Deprioritized

- **Hydraulic erosion**: Not producing visible results. With terrain shaped by
  noise for rideability rather than erosion for realism, the erosion module
  becomes optional. Keep as a light smoothing pass if helpful, remove if not.
- **Flow accumulation / river graph**: Water features can be placed at
  extraction time from the heightmap (stream meshes in low paths between
  ridges). Not a simulation concern for now.
- **Climate-hydrology tick timing**: The 1000-year lurch problem dissolves.
  Climate is 2D noise fields modulated by per-season fractal weather. The
  slow-tier discontinuity goes away.
- **The current heightmap and calibration against it**: The world gets
  regenerated. Ecology rules survive; specific parameter values don't.

## What Survives Intact

- All ecology mechanics (growth, competition, dispersal, speciation, biotic
  pressure, gene flow/reabsorption, Allee threshold)
- The perturbation testing framework (equilibrium + perturbation validation)
- The three-layer architecture (canonical state / query / extractors)
- Event-based tier communication
- Reproducibility from seed
- The webview inspector (consumes new data)
- The ride-experience tool (A* pathfinding, strip sampling)
- The weather system (fractal noise modulation of climate over time)

## Relationship to Existing Roadmap

This plan replaces the original Phase 9 (Godot extractor) and Phase 10 (asset
agent) with a more iterative, experience-driven sequence. The original phases
assumed the simulation would be "done" before rendering started. This plan
recognizes that **rendering is the feedback loop that tells us whether the
simulation is producing the right thing** — the terrain, the climate model,
and the ecology parameters all need tuning against what feels right at cycling
speed.

The original Phases 0–8 (infrastructure + all three tiers + orchestrator) are
complete and remain the foundation. The ecology refactor (June 2026) is stable
and its mechanics carry forward unchanged.

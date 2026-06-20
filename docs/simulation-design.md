# Simulation Design

Detailed notes on what each simulation tier actually does. These are the design intentions; implementation can simplify any of this for v1, but new implementations should engage with these intentions rather than ignore them.

## Geology

### What it produces
- Heightmap (base topography, before erosion).
- Bedrock type map (rock classes with weathering/erosion properties).
- Soil parent material (derived from bedrock + exposure history).
- Fault and joint structure (vector features that influence later erosion patterns).

### Deep-time vision (post-v1)
- Plate tectonics via clustered-convection-style simulation on Voronoi mesh.
- Orogeny producing aligned mountain ranges with consistent strike directions.
- Volcanic provinces leaving distinctive bedrock signatures.
- Glacial cycles scouring high-latitude/high-altitude terrain.

### v1 stub
- Multi-octave noise heightmap.
- Bedrock assignment via Voronoi regions with random rock-type labels.
- Soil parent material assigned by simple rules.
- Plumbing for "geology ticks" exists but is mostly a no-op.

## Climate-hydrology

### Climate envelope (slow, persistent)
- Latitude-based base temperature gradient.
- Prevailing winds (and seasonal variation).
- Base precipitation field, modified by topography (orographic effects, rain shadows).
- Storm tracks.
- Decadal-to-centennial oscillations drift this envelope over time.

### Weather realizations (fast, sampled)
- Sampled from envelope when ecology or other consumers need them.
- "What did year 1247's growing season look like at point P?" — drawn from distributions in the envelope, possibly with autocorrelation across years.
- Cached only short-term; not part of persistent state.

### Hydrology
- **River graph**: confluences as nodes, reaches as edges with flow rates. Lakes as polygons in basins.
- **Flow accumulation**: every cell's water goes somewhere; compute drainage networks from the heightmap.
- **Hydraulic erosion**: particle-based or grid-based. CPU-vectorized first (numpy/numba); GPU later if needed. This is what carves the heightmap into something that *looks like* it was carved by water.
- **Sediment**: carried by water, deposited where flow slows, forming deltas and floodplains.
- **Soil moisture**: derived from precipitation, drainage, topography, soil properties.
- **Springs**: emerge where water table meets surface (faults, geological contacts).
- **Lakes**: fill basins until they overflow; overflow paths carve.
- **Meander migration**: rivers move laterally over decades; oxbow lakes form when meanders cut off.

### Derived-state cache (for ecology)
Recomputed at each climate-hydrology tick:
- Effective soil moisture (summer / winter).
- Frost pocket likelihood (cold air pooling in concave terrain).
- Growing degree days.
- Flood return interval per cell.
- Solar insolation (slope + aspect + latitude).
- Distance to permanent water.

## Ecology

> The ecology tier was substantially rewritten in June 2026. The guiding
> principle is **resolution over coefficients**: a simple rule applied at high
> spatial/temporal granularity (1M cells × thousands of ticks × 14 species)
> produces emergent, legible behavior; drama comes from perturbing the
> *environment*, not from stacking tick complexity. See
> `ecology-tick-refactor.md` for the full design and rationale, and
> `decisions.md` for the incremental decisions.

### Species data model

Each species has:
- **Genome**: 6 traits, each mapping to exactly one mechanism — `drought_tolerance`,
  `frost_tolerance`, `growth_rate`, `max_height` (competition / shading),
  `lifespan` (base mortality), `dispersal_range`. Additional traits
  (shade_tolerance, mast_interval, morphology for rendering) are deferred and
  added back only when they earn rider-visible effects.
- **Range**: density field across the map (50m resolution). Density is continuous
  biomass per cell, not an individual count.
- **History**: when it appeared, lineage (parent species ID if speciated).

### Suitability (absolute normalization)

For each species at each cell, suitability is a **product** of Gaussian
factor-matches (Liebig's law of the minimum — a severe mismatch on any axis
knocks suitability toward zero):
- Drought stress (from precipitation) vs. `drought_tolerance`.
- Temperature vs. warmth preference (`1 - frost_tolerance`).

Normalization uses **fixed reference ranges fit to the world's climate
envelope** (e.g. precip 350–2650mm, temp −10/30°C), held constant rather than
recomputed per tick. This is load-bearing: a per-tick/relative normalization
erases the temporal climate signal, so a wet-vs-dry century looks identical and
species ranges never migrate. Fixed references let a global climate shift
actually change suitability, which is what drives biome migration.

### The tick (one unified rule, every season)

There is **no season-specific branching** in the core math. Each tick, for each
species at each cell:

- **Logistic growth**: `density × growth_rate × (K_eff − load)/K × 0.25`, where
  `K_eff = K × suitability` is the species' per-cell carrying capacity and
  `load` is the competition-weighted sum of all species' densities. The
  `(K_eff − load)/K` term goes **negative** when overcrowded — this negative
  feedback is what keeps density from overshooting carrying capacity.
- **Mortality**: a fixed base rate `density / (lifespan × 4)`. Mortality does
  *not* scale with suitability — suitability already governs growth, and making
  mortality climate-dependent too created a contraction ratchet (species dying
  in their own suitable habitat).
- **Carrying capacity** `K` varies by terrain (~5 on dry ridges to ~20 in wet
  lowlands), from moisture × elevation.

Breakeven suitability is therefore analytically predictable:
`suit > 1 / (growth_rate × lifespan × 4)` — very low, so suitability alone
doesn't draw range boundaries. **Competition does** (next).

### Competition draws the boundaries

A Lotka-Volterra alpha matrix weights how strongly species compete by genome
distance, but with a **baseline floor** (`COMPETITION_BASELINE`): even
functionally dissimilar species compete for shared physical space. In each cell
the species with the highest local `K_eff` (best-suited) suppresses the others,
so biome boundaries emerge where the suitability ranking flips between species —
and *move* as climate shifts. Without the baseline, every species survives
everywhere it's remotely suited and the world becomes a uniform soup.

### Dispersal and refugia

- **Dispersal** runs every tick from living density: a local distance-weighted
  kernel (radius = `dispersal_range`) plus rare long-distance jumps. No seed
  bank — recolonization happens from range edges, where empty cells have zero
  competition so arriving propagules grow fast.
- **Refugium floor** (optional): a species retains a trace presence in its single
  most-suitable cell so it can wait out an unfavorable phase and rebound when
  its conditions return. This is how a wet specialist survives dry centuries to
  flourish again at the next wet peak — the soft alternative to a seed bank.

### Deferred mechanics (toggleable)

These exist in code but are gated behind toggles and currently disabled while the
core dynamics are validated:
- **Distinguished individuals**: prominent trees promoted to named, tracked
  individuals (snag → log → mound after death). Restricted to trees.
- **Extinction & speciation**: deferred together to a dedicated design session —
  extinction without reliable speciation collapses diversity, and speciation has
  been hard to get right. Likely to return as a paired mechanic.

### Disturbance regimes

Disturbance is the primary *environmental perturbation* — discrete events, not
per-tick drain:
- **Fire**: cellular-automaton spread driven by dryness and storm intensity;
  clears density in the burned area. Recovery is natural (dispersal from
  unburned edges into now-uncompeted cells).
- **Blowdown**: storm-triggered windthrow in exposed positions; opens canopy.
- **Flood / disease / insect outbreaks**: future stochastic perturbations.

### Keystone species (post-v1)
A small number of species with outsized impact (beaver-analog with
hydrology-write permission, large grazers suppressing trees, fungal symbionts).
These create feedback loops that produce surprises.

### Stochasticity by design

Don't make every event causally derivable from the tiers below — include
irreducibility at each tier (long-distance dispersal, rare outbreaks). The
"huh, what's that doing here?" moments are the payoff.

### Testing methodology

The ecology is validated by a falsifiable test loop (`scripts/test_ecology.py`):
run to **equilibrium** under a frozen-drift seasonal cycle (slow climate held
constant, 4 seasonal snapshots cycled), then apply a single **perturbation**
(temperature/precipitation shift, fire, species removal) and verify the response
matches an ecological prediction — e.g. "a wet shift makes dry species retreat
and wet species rebound from their refugia." This is the core development
workflow for tuning ecology parameters.

## Overnight advancement

### Trigger
- A new ride log appears in the configured directory.
- The CLI entry point reads the log, computes time to advance, kicks off the simulation.

### Time budget
- Each ride contributes some amount of simulation time. Tentatively: capped, so an extreme ride doesn't fast-forward through years of state the player loses connection to.
- Mapping function: ride duration → simulated years advanced. Suggested starting point: 20-50 years per ride, with some variation based on duration but capped.
- Skip a day: world waits.
- Manual override: "introduce fire" / "introduce supervolcano" / "advance 100 years" / "rebuild from scratch" as CLI commands.

### Pass ordering during advance
1. Compute total simulated time to advance.
2. Tick ecology multiple times across that period.
3. Tick climate-hydrology a few times (with batched events from ecology).
4. Tick geology only if a long advance has passed since last geology tick.
5. Apply state changes downward where slow tiers ticked.
6. Update derived-state cache.
7. Run extractor (route selection + chunk generation for next ride).

### Mortality of worlds
- Worlds can enter terminal trajectories (snowball earth, runaway desertification). This is *intentional*, not a failure mode.
- Player can choose to watch terminal trajectories play out or intervene.
- Periodic "is the world stale?" check; surface to the player but don't auto-reset.

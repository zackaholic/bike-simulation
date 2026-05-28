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

### Species data model

Each species has:
- **Genome**: 15-25 dimensional trait vector
  - Continuous traits: leaf area, root depth, max height, lifespan, drought tolerance, frost tolerance, shade tolerance, seed mass, growth rate, etc.
  - Categorical traits: woody vs. herbaceous, evergreen vs. deciduous, reproduction mode, dispersal mode.
  - Linked tradeoffs: high growth rate ↔ short lifespan; large seeds ↔ low seed count; drought tolerance ↔ growth rate in moist conditions.
  - Traits constrained to fall mostly on the Grime CSR manifold (competitor / stress-tolerator / ruderal).
- **Range**: density field across the map (50m resolution).
- **History**: when it appeared, lineage (parent species ID), notable events (range fragmentation, near-extinction recoveries, etc.).

### Establishment suitability

For each species at each cell, compute a product of factor-matches:
- Drought tolerance vs. summer soil moisture.
- Frost tolerance vs. winter minimum temperature.
- Shade tolerance vs. existing canopy.
- Soil type tolerance.
- etc.

Each factor is a Gaussian-ish match function. **Product**, not sum (Liebig's law of the minimum: severe mismatch in any factor knocks suitability to ~zero).

Then add a competition term: existing residents consume light, water, nutrients in proportion to their density and traits. New establishers must find under-exploited niches.

### Population dynamics
- Reproduction: existing populations produce seed input to neighboring cells (dispersal kernel depends on dispersal trait).
- Mortality: from old age, competition, stress, disturbance.
- Establishment: from seed input + seed bank, weighted by suitability and competition.

### Seed bank
- Each soil cell has a sparse map `{species_id: seed_density}`.
- Decay rate depends on species traits (some species: ~5 year half-life; others: ~200 year).
- After disturbance, establishment can draw from current seed input *or* from the seed bank.
- This produces "return-of-the-vanished" moments — burned hillsides come back as something suppressed for a century.

### Distinguished individuals

When a plant survives past an age threshold in a *prominent* position, it gets promoted:
- Unique ID, stable position, age, accumulated event log.
- Promotion triggers: local height maximum within radius, only specimen of species within radius, unusually old, adjacent to bike path, survivor of disturbance that killed neighbors.
- Persists across simulation ticks until it dies.
- After death: snag → log → mound, each with decreasing prominence over decades.

### Speciation

Each species' range is tracked as a graph of connected populations (flood-fill on cells where density > threshold).

When the graph fragments — population isolated by impassable barrier — start a divergence clock on the isolated piece. Traits drift directionally toward local conditions. When divergence clock + trait distance exceeds threshold, populations speciate.

Result: real evolutionary radiation. Start with 5-10 ancestors; after deep time, dozens of derived species traceable to specific geographic/climatic events.

### Disturbance regimes

- **Fire**: cellular automaton. Ignition from "lightning" (low base rate, modulated by climate). Spread depends on vegetation density, moisture, wind, slope. Fires bare-soil affected cells, kill above-ground biomass, trigger seed-bank germination cycles. Fire events emit "burned area" upward to climate-hydrology (erosion pulse follows).
- **Blowdown**: storm-triggered windthrow in exposed positions.
- **Flood**: from hydrology tier; scours riparian zones, deposits sediment.
- **Disease/insect outbreaks**: low-rate stochastic events affecting specific species or genome regions, can kill significant portions of a population.

### Keystone species (post-v1)
A small number of species with outsized impact:
- A beaver-analog: dams streams, floods forests (which die and become snags then meadows). Has hydrology-write permissions.
- Large grazers: maintain grasslands by suppressing tree establishment.
- Possibly: a fungal symbiont required for certain tree establishment.

These create the feedback loops that produce surprises.

### Stochasticity by design

Don't make every event causally derivable from the tiers below. Real ecosystems have:
- Long-distance dispersal founding new populations where they shouldn't quite be able to reach.
- Beetle outbreaks, seed-mast years, random rare events.
- Genetic drift untied to selection pressure.

Include some irreducibility at each tier. The "huh, what's that doing here?" moments are the payoff.

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

# Gap Analysis: Design Spec vs Implementation

Comprehensive comparison of what `simulation-design.md` and `architecture.md` specify vs what's actually built as of Phase 8. Organized by tier, then by subsystem.

## Geology

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Multi-octave noise heightmap | v1 stub | Done | 6 octaves, bilinear upsampling, 0-2000m |
| Voronoi bedrock regions | v1 stub | Done | 15-20 seed points, 6 rock types |
| Soil parent material | Derived from bedrock + exposure | Simplified | Hardcoded lookup, no exposure history |
| Fault/joint structure | Vector features influencing erosion | **Missing** | No GeoJSON, no vector storage |
| Bedrock weathering properties | Rock classes with erosion properties | **Missing** | Types exist but don't influence erosion |
| Plate tectonics | Post-v1 | Deferred | As planned |
| Glacial cycles | Post-v1 | Deferred | As planned |

**Key gap:** Geology produces no vector features (faults/joints) and bedrock types don't influence downstream erosion behavior. The heightmap is purely noise-derived with no geological process.

## Climate-Hydrology

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Latitude-based temperature | Specified | Done | 15C south to 5C north + lapse rate |
| Orographic precipitation | Specified | Done | Simplified: proportional to x-gradient |
| Prevailing winds | Specified | **Missing** | No wind field, no seasonal variation |
| Storm tracks | Specified | **Missing** | |
| Climate envelope drift | Decadal-centennial oscillations | **Missing** | Climate recomputed fresh each tick, no memory |
| Weather realizations | Sampled from envelope per-year | **Missing** | Only deterministic annual averages |
| D8 flow accumulation | Specified | Done | Pure Python loop (~5s), correct but slow |
| River graph (nodes/edges) | Confluences, reaches, flow rates | **Missing** | Only raster flow arrays, no graph structure |
| Hydraulic erosion | Particle or grid-based | Simplified | 5 passes grid-based, gentle, no physics coefficients |
| Sediment transport/deposition | Carried by water, deposited | **Missing** | Erosion applied but sediment not tracked |
| Lakes | Fill basins, overflow paths | Implicit | Top 0.5% flow = "water"; no volumes, no overflow carving |
| Springs | Water table meets surface | **Missing** | No water table simulation |
| Meander migration | Rivers move laterally | **Missing** | Rivers static |
| Frost pockets | Cold air pooling in concave terrain | Simplified | Global formula, no topographic concavity check |
| Flood return interval | Per-cell statistics | **Missing** | |
| Derived-state cache (6 layers) | Specified | Done | All 6 layers present |

**Key gaps:** Climate is a static snapshot with no drift, oscillations, or weather variability. Rivers exist as flow accumulation rasters but not as a proper graph. Sediment doesn't exist. Erosion is cosmetic (5 passes) rather than physically motivated. Lakes are implicit.

## Ecology

### Species Model

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| 15-25D genome trait vector | Specified | **7 traits only** | Missing: leaf area, root depth, categorical traits |
| Categorical traits | Woody/herbaceous, evergreen/deciduous, dispersal mode | **Missing** | Only continuous traits |
| Linked tradeoffs | Growth rate <-> lifespan, seed mass <-> count | **Missing** | Traits are independent |
| Grime CSR manifold constraint | Traits cluster on CSR triangle | **Missing** | No validation or constraint |
| Soil type tolerance | Match against geology soil_parent | **Missing** | No interaction with geology |
| Shade tolerance vs canopy | Existing canopy blocks light | Simplified | Uses static solar insolation, not dynamic canopy |

### Population Dynamics

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Suitability as product of Gaussians (Liebig) | Specified | Done | 4 factors: drought, frost, warmth, light |
| Competition via resource consumption | Density + traits consume light/water/nutrients | Simplified | Single carrying capacity scalar, no per-resource competition |
| Dispersal kernel from traits | Specified | Done | Radius from seed_mass |
| Long-distance dispersal | Specified | Done | 10% chance, 1-4 random placements |
| Seed bank with trait-dependent decay | Specified | Done | Half-life 5-200yr from seed_mass |
| Fire triggers seed bank germination | Specified | **Missing** | Fire kills density but doesn't trigger germination |
| Mortality from old age | Specified | Simplified | Base rate from lifespan, no cohort aging |

### Distinguished Individuals

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Promotion from density | Specified | Done | Above 80th percentile, 2-5 per species per tick |
| Local height maximum trigger | Specified | **Missing** | Uses density, not spatial prominence |
| Only specimen in radius | Specified | **Missing** | |
| Unusually old trigger | Specified | **Missing** | No age tracking |
| Adjacent to bike path | Specified | **Missing** | No bike path in simulation |
| Survivor of disturbance | Specified | **Missing** | No disturbance-survival detection |
| Accumulated event log | Specified | **Missing** | Only appeared_year stored |
| Post-mortem snag -> log -> mound | Specified | **Missing** | Individuals never die or decay |
| Death detection | Implied | **Missing** | No mechanism to kill individuals |

### Speciation

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Population graph via connected components | Specified | Done | Downsampled 10x for performance |
| Divergence clock on isolation | Specified | **Missing** | Fixed 30% probability, no time accumulation |
| Directional trait drift toward local conditions | Specified | **Missing** | Random walk, not adaptive |
| Trait distance threshold for speciation | Specified | **Missing** | Probability-based, not distance-based |

### Disturbance

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Fire CA (ignition + spread) | Specified | Done | BFS spread, moisture-modulated |
| Fire -> ecology coupling (kill biomass) | Specified | Done | 70-95% kill in burned cells |
| Fire -> climate coupling (erosion pulse) | Specified | **Missing** | Events recorded but not consumed |
| Blowdown | Specified | Done | Poisson 0.3/tick, elevation-biased |
| Flood disturbance from hydrology | Specified | **Missing** | No connection to flow/rivers |
| Disease/insect outbreaks | Specified | **Missing** | |
| Keystone species (beaver, grazers) | Specified | **Missing** | No feedback loops |

## Tier Communication

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Ecology reads climate cache | Read-only downward | Done | |
| Climate reads geology | Read-only downward | Done | |
| Ecology emits events upward | Batched event emission | Partial | Events recorded but not consumed by climate |
| Climate emits events to geology | Batched event emission | **Missing** | No sediment/incision events |
| Downward state changes (glacier wipes ecology) | Specified | **Missing** | No propagation mechanism |

## Orchestrator

| Feature | Design Spec | Status | Notes |
|---------|------------|--------|-------|
| Tier tick scheduling | Specified | Done | |
| Ride log file monitoring | Specified | **Missing** | Accepts years parameter directly |
| Ride duration -> sim years | 20-50 years/ride, varied | Simplified | Fixed 1 year/min, cap 50 |
| Manual commands | fire, supervolcano, advance N, rebuild | Partial | Only introduce_fire |
| Extractor call after advance | Specified | **Missing** | Extraction is separate |
| "Is the world stale?" detection | Specified | **Missing** | |
| World mortality (terminal trajectories) | Intentional feature | **Missing** | No detection or player notification |

## What's Solid

- **Three-tier architecture** is correctly implemented and extensible
- **Reproducibility from seed** works perfectly — tested and verified
- **Data layer** (Zarr + SQLite + World directory) is clean and reliable
- **Query interface** works for all current consumers
- **Debug visualizer** provides immediate visual feedback
- **CLI** makes the system usable
- **Population dynamics** produce ecologically plausible distributions
- **Speciation** works (6 -> 15 species) even if simplified
- **Disturbance** creates visible history

## Priority Gaps (most impact on the cyclist's experience)

1. **Distinguished individual lifecycle** — these are the emotional anchor points. They need age tracking, meaningful promotion triggers, death, and snag/log/mound decay.
2. **Fire -> seed bank germination** — the "return of the vanished" moment is a signature design goal and needs fire to actually trigger it.
3. **Upward event consumption** — fire should trigger erosion pulses in climate-hydrology. This is the simplest feedback loop to add.
4. **Deeper erosion** — 5 passes produces subtle changes. More passes (or particle-based) would give the terrain visible water-carved character.
5. **Richer trait model** — categorical traits and linked tradeoffs would make species feel more distinct and produce more surprising niche interactions.
6. **Climate drift** — without it, the world's climate is frozen. Species distributions can change but the backdrop doesn't.
7. **Speciation driven by conditions** — directional drift toward local conditions rather than random walk would make derived species feel like they belong to their landscape.

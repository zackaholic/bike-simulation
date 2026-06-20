# Two-Layer Ecology and Spatial Climate Variation

Design document for restructuring the ecology system to guarantee full world coverage, create distinct spatial regions, and separate ground cover from structural plant dynamics.

**Status**: Design phase. Not yet implemented.

**Motivation**: Calibration runs through v9 (4000 simulated years) revealed that the current colonization model cannot fill a world. Six point-source species expand through tick-by-tick dispersal, but suitability filtering kills propagules in marginal cells faster than dispersal can establish them. The result is dense islands of vegetation in a mostly barren landscape. No amount of sigma/kernel tuning fixes the structural mismatch: real Earth had billions of years for colonization; we simulate thousands.

Additionally, the world lacks spatial climate variation beyond elevation effects. Every valley feels the same as every other valley. The climate cycles we built (fractal noise temperature/precipitation) shift the whole map uniformly, so species oscillate in density but don't migrate between regions. The compelling narrative — species bridging a barrier during a wet period, establishing, then getting cleaved when the barrier re-emerges — requires spatially distinct climate zones that shift with global conditions.

---

## 1. Spatial Climate Variation

### The problem

Currently, weather varies with elevation (lapse rate, orographic precipitation) and time (fractal noise cycles), but has no horizontal spatial structure. A lowland cell on the east side of the map has identical climate to a lowland cell on the west side at the same elevation. This means:
- No distinct regions emerge from climate alone
- Climate cycles shift all cells equally, preventing range migration
- "Barrier" zones (deserts, frost belts) can't form or dissolve dynamically

### The solution: static bias fields modulated by dynamic weather

Add two 2D noise fields computed once at world creation from the world seed:

**Moisture bias field** (`moisture_bias`): Values in [0.5, 2.0]. Multiplied against precipitation. Creates structurally wetter and drier regions. A cell with `moisture_bias=0.7` is always drier than one with `moisture_bias=1.5`, but during a globally wet period even the dry cell may become habitable.

**Continentality field** (`continentality`): Values in [0.0, 1.0]. Scales temperature extremes (amplifies seasonal range). High continentality = hot summers, cold winters, higher frost risk. Low continentality = mild, maritime-like. Affects frost_tolerance requirements and growing season length.

These fields are:
- **Static**: computed once from world seed, never change. They represent geography (rain shadows, distance from implied coast, mountain sheltering).
- **Low-frequency**: generated with large-scale noise (wavelength ~30-50% of world size) so regions are landscape-scale, not cell-scale. Maybe 3-5 distinct "zones" emerge naturally.
- **Continuous**: no hard boundaries. Transitions are gradual, creating ecotones.

### How they interact with weather

The weather system currently produces `(temp_anomaly, precip_multiplier)` per tick from fractal noise. With spatial bias:

```
effective_precip[cell] = base_precip[cell] * precip_multiplier * moisture_bias[cell]
effective_temp_range[cell] = base_temp[cell] + temp_anomaly * (0.7 + 0.6 * continentality[cell])
```

During a globally wet period (`precip_multiplier=1.3`):
- Wet-biased cell (1.5): effective = base * 1.95 (very wet)
- Dry-biased cell (0.6): effective = base * 0.78 (marginal, but possibly habitable)

During a globally dry period (`precip_multiplier=0.7`):
- Wet-biased cell (1.5): effective = base * 1.05 (still okay)
- Dry-biased cell (0.6): effective = base * 0.42 (desert)

This naturally creates:
- **Persistent regions** that feel distinct (always-wet valleys, always-dry ridges)
- **Shifting boundaries** as global climate pushes marginal areas over thresholds
- **Barrier formation/dissolution** as dry zones expand and contract with climate cycles
- **The desert-bridge narrative**: dry-biased area becomes traversable during wet centuries, species cross, barrier reforms, populations cleave

### Noise parameters (starting point, will tune)

- Octaves: 2-3 (we want broad regions, not fine texture)
- Base frequency: ~1/400 cells (regions span ~20km at 50m resolution)
- Moisture bias range: [0.5, 2.0] (2x dry to 2x wet relative to base)
- Continentality range: [0.0, 1.0] (maritime to continental)

### Rendering implications

The moisture bias + continentality fields define the character of each area without naming it. The renderer can use these fields (plus elevation) to select environment-appropriate ground textures, ambient sounds, atmospheric haze, etc. No biome labels needed — the continuous fields ARE the biome definition.

---

## 2. Two-Layer Ecology

### The problem

The current ecology tier simulates all plant species with the same population dynamics model: suitability-filtered growth, competition, dispersal, mortality. This model is appropriate for trees and shrubs (structural plants with rendering cost and ecological drama) but is overkill for ground cover (grass, moss, lichen, bare soil) which:
- Must be present everywhere for visual quality
- Changes rapidly with conditions (seasonal, annual)
- Is cheap/free to render (texture, not geometry)
- Doesn't need population dynamics, dispersal, or speciation

### The solution: separate ground cover from canopy ecology

**Ground cover layer** (new, simple):
- A direct function of current conditions at each cell: temperature, precipitation, soil moisture, elevation, season
- Always produces a value — the world is never empty
- Output: categorical ground type + density/vigor scalar
- Changes every season automatically
- No dispersal, no competition, no population state
- Computed as part of derived climate state, not ecology tier

**Canopy layer** (existing ecology tier, restructured):
- Trees, shrubs, and tall structural plants
- The competitive, dispersal-driven ecology we've built
- Speciation, range shifts, DIs, climate coupling — all preserved
- Initialized broadly across suitable habitat (see Section 3)
- This is where rendering geometry lives (expensive)

### Ground cover types

Determined by climate conditions at each cell. Not a simulation — a lookup/interpolation:

| Conditions | Ground cover | Visual |
|-----------|-------------|--------|
| Warm + wet | Lush grass, dense herbs | Deep green, flowers in spring |
| Warm + dry | Dry grass, sparse scrub | Golden/brown, patchy |
| Cool + wet | Mossy meadow, ferns | Dark green, damp |
| Cool + dry | Hardy grass, lichen | Pale green/grey |
| Cold + wet | Alpine meadow | Short green, seasonal flowers |
| Cold + dry | Lichen, bare rock | Grey/brown, sparse |
| Very cold | Snow/ice, bare rock | White/grey |
| Very dry | Bare soil, sand | Tan/brown |

These are not discrete categories — they interpolate continuously based on temperature, moisture, and season. The table is illustrative of the target visual range.

### Seasonal ground cover variation

Ground cover responds to current season:
- **Spring**: green flush everywhere, flowers in meadows. Intensity proportional to moisture + warmth.
- **Summer**: peak biomass in wet areas, browning/drying in dry areas. Altitude gradient visible.
- **Autumn**: die-back begins at high elevation. Dry areas already brown.
- **Winter**: dormant everywhere except sheltered wet valleys. Snow at elevation.

Over years of riding, the rider notices: "the meadow near the ridge used to green up in spring but the last few years it's been dry and brown" — because the climate cycle shifted moisture away.

### Interaction between layers

Ground cover type influences canopy ecology:
- Dense grass may slow tree seedling establishment (competition for light at ground level)
- Bare/disturbed ground favors pioneer species establishment
- These are lightweight modifiers, not full simulation coupling

Canopy presence influences ground cover:
- Dense canopy = shade-tolerant ground cover (ferns, moss)
- Open canopy = sun-loving ground cover (grass, flowers)
- No canopy = ground cover determined purely by climate

---

## 3. Species Initialization (Broad Placement)

### The problem

Starting 6 species from point sources requires thousands of simulated years to fill the world, and with suitability filtering, they never actually fill it. The "generation and advancement are the same operation" principle was producing empty worlds.

### The solution: scatter species broadly at creation, let dynamics settle

**More ancestral species** (12-15 instead of 6), organized by structural role:

| Role | Count | Examples | Dispersal | Render cost |
|------|-------|----------|-----------|-------------|
| Canopy trees | 2-3 | Deciduous hardwood, evergreen conifer, tropical broadleaf | Slow, heavy seeds | High (geometry) |
| Understory/edge trees | 2-3 | Fast-growing gap filler, riparian specialist | Moderate | Moderate |
| Tall shrubs | 3-4 | Ridge scrub, valley thicket, heath | Moderate-fast | Moderate (billboard) |
| Low shrubs/forbs | 3-4 | Alpine cushion, flowering scrub, pioneer herb | Fast, light seeds | Low (sprite/texture) |

**Placement algorithm:**
1. Compute spatial climate fields (moisture bias + continentality)
2. Generate "year zero" weather
3. For each species:
   a. Compute suitability across the full map
   b. Generate a species-specific noise field (from world seed + species index) — this creates natural clustering (groves, patches)
   c. Initial density = `suitability * species_noise * max_density_for_role`
   d. Zero out cells below a survivability threshold
4. Run 50-100 years of settling simulation to let competition sort out overlaps

This produces a world that is:
- **Full**: every cell with habitable conditions has vegetation
- **Spatially structured**: species cluster naturally via noise, correlate with climate zones via suitability
- **Near equilibrium**: competition has resolved major overlaps
- **Not perfectly settled**: some species are in marginal territory, some clearings being colonized — the world has "recent history"

### Generation vs. exploration

The generation/exploration distinction becomes about **starting conditions**, not rules. The same dispersal, competition, and climate-response rules apply in both phases. During generation we run a fast settling period; during play the rider watches the same dynamics at ride pace. The rules don't change — the world just starts closer to steady state.

This preserves the design principle in spirit: there's no separate "placement" code path that bypasses ecology. The broad initial scatter is the equivalent of saying "this world has been inhabited for millions of years" — we're computing the implied steady state directly rather than simulating our way there.

---

## 4. Speciation Redesign

### Changed role

In the old model, speciation was load-bearing infrastructure — the only way to fill ecological niches. With broad placement and more ancestral species, speciation becomes a **narrative event**: rare, dramatic, driven by major climate disruptions.

### New design principles

**Rarer**: Once every few thousand years, not every few hundred. The world starts with enough species to cover its niches.

**More dramatic**: Daughter species should be visibly different. When speciation happens, the morphological drift should be large enough that the rider notices a new kind of plant.

**Geographically driven**: The trigger is prolonged isolation across a real environmental barrier. Climate cycles create and dissolve barriers; populations that are separated long enough under different conditions diverge.

**Structural-role dependent**: 
- Trees speciate most readily (heavy seeds, can't bridge barriers, long-lived populations persist in isolation)
- Shrubs occasionally (moderate dispersal)
- Herbs/forbs rarely (light seeds bridge most barriers via LDD, preventing true isolation)

### Specific changes to consider

- Raise minimum isolation time before speciation check (currently 100yr, target ~500yr average)
- Increase morphological drift during speciation (make daughters visually distinct)
- Consider whether genome divergence threshold should be higher (0.15 → 0.25?)
- Speciation probability could scale with structural role (trees > shrubs > herbs)
- The reabsorption mechanic remains important — briefly isolated populations merge back with gene flow

### What to preserve

- Barrier-based isolation check (BFS through suitability — this is the right mechanic)
- Genetic divergence threshold (prevents meaningless micro-speciation)
- Niche saturation scaling (prevents runaway at system level)
- Reabsorption as the inverse process

---

## 5. Ghost Species Bug

### Problem observed

v9 calibration (4000 years) shows 8 species listed as "alive" with zero density and zero occupied cells. They appear in the species table but have no ecological presence. These are species that went functionally extinct (density dropped to zero everywhere) but were never formally marked as extinct.

### Likely cause

The extinction check probably requires density to drop below some threshold that zero doesn't trigger correctly, or the seed bank is keeping them technically "alive." Need to audit the extinction logic.

### Fix

Add an explicit check: if a species has zero occupied cells (density > threshold) for N consecutive ticks, mark it extinct. This should be straightforward and independent of the other changes in this document.

---

## 6. Implementation Sequence

### Phase A: Spatial climate fields (foundation)
1. Add moisture_bias and continentality noise generation to world creation
2. Modify WeatherSystem to apply spatial bias to temperature and precipitation
3. Validate visually: the world should show distinct wet/dry regions
4. Run calibration to verify ecology responds to spatial variation

### Phase B: Ground cover layer
1. Define ground cover computation from climate conditions
2. Add as a derived layer (computed each ecology tick or each season)
3. Add to webview inspector for visual validation
4. No changes to ecology tier needed — this is purely additive

### Phase C: Species initialization redesign
1. Design 12-15 ancestral species across 4 structural roles
2. Implement broad placement algorithm (suitability * species noise)
3. Run settling period during world creation
4. Validate: world should be full, with species distributed according to climate zones
5. Calibrate settling duration

### Phase D: Speciation tuning
1. Adjust speciation parameters for the new regime (rarer, more dramatic)
2. Increase morphological drift
3. Consider role-dependent speciation rates
4. Fix ghost species bug
5. Calibration runs to validate

### Dependencies
- Phase A is the foundation — everything else depends on spatial climate variation
- Phase B is independent of C and D (can be done in parallel)
- Phase C depends on A (species placement needs spatial climate)
- Phase D depends on C (speciation tuning needs the new species set)

---

## 7. Open Questions

1. **How many distinct climate zones should emerge?** The noise parameters determine this. Too few (2-3) and the world feels binary. Too many (10+) and regions lose character. Target: 4-6 distinct "feels" with gradual transitions.

2. **Should moisture bias correlate with elevation?** Yes — scale moisture with elevation to create natural rain shadow effects. Try it and tune.

3. **Ground cover computation — how detailed?** The simplest version is a lookup table from (temp, precip, season). Could add soil type, canopy shade, recent disturbance. Start simple, add complexity where it produces visible results.

4. **Settling duration**: Start with 500 years. Will require experimentation and tuning.

5. **Rendering budget for structural plants**: How many tree/shrub instances can the renderer handle? This constrains species density and may affect how we map density values to placed instances. Flagged but not blocking — we design the simulation side now, tune density-to-instance mapping during extraction.

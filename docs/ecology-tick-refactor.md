# Ecology Tick Refactor

## Motivation

The current ecology tick has accumulated 7+ seasonal mortality/growth terms, each independently tuned across different design phases. The interaction between terms is unanalyzable — you can't look at a species' genome and predict whether it will grow or shrink at a given suitability. Diagnostic analysis (June 2026) confirmed:

- **Stress mortality double-dips**: suitability already controls growth, but `0.02 * (1 - suit)` also increases mortality. This makes the breakeven suitability surprisingly high (~0.50 for slow growers).
- **5:1 loss-to-gain ratio**: species lose cells 5x faster than they gain them, even in suitable habitat.
- **100% of cell losses occur in suitable territory**: species aren't dying at range margins, they're dying everywhere.
- **Ranges contract monotonically**: species consolidate into core areas and never expand, regardless of climate state.

The root insight: **resolution, not coefficients, should be the source of complexity**. Simple rules applied at high granularity (1M cells, thousands of ticks) produce emergent behavior that is legible and debuggable. Many interacting coefficients produce behavior that is neither.

## Design principles

1. **Simple, constant rules. Variable inputs.** The ecology math should be the same every tick. Seasonal variation comes from weather inputs (temperature, precipitation, frost), not from season-specific code paths.

2. **Single-purpose terms.** Each mechanism (growth, mortality, competition, dispersal) does one thing. No double-dipping: suitability controls growth potential, mortality is a fixed base rate, not climate-scaled.

3. **Equilibrium is the base state.** Under static conditions, species should reach a stable distribution that tracks suitability. Drama comes from perturbing the environment, not from tick mechanics.

4. **Every mechanic must earn its place.** New terms are added only when they produce rider-visible effects that simpler rules cannot. The perturbation testing framework validates this.

5. **Analytically tractable.** Given a genome and a suitability value, you should be able to compute the breakeven conditions on paper.

## The new tick

### Per-tick rule (every season, 4 ticks/year)

```
For each species at each cell:
    suit = suitability(genome, weather)
    
    growth = density * growth_rate * suit * (available / K)
    mortality = density * base_turnover
    
    new_density = density + growth - mortality
```

Where:
- `growth_rate` — genome trait, species' maximum growth rate
- `suit` — [0, 1], computed from weather (temperature, precipitation) and genome (drought_tolerance, frost_tolerance)
- `available / K` — logistic competition term; available = K - effective_load from all species weighted by competition coefficients
- `base_turnover` — derived from lifespan: `1.0 / (lifespan * ticks_per_year)`
- No stress mortality term. Low suitability already means low growth. Mortality doesn't need to scale with it.

**Breakeven suitability** (in an empty cell where available/K = 1):

```
growth > mortality
growth_rate * suit > base_turnover
suit > 1 / (growth_rate * lifespan * ticks_per_year)
```

For a tree with growth_rate=0.15, lifespan=500, 4 ticks/year:
```
suit > 1 / (0.15 * 500 * 4) = 0.0033
```

For a shrub with growth_rate=0.35, lifespan=40, 4 ticks/year:
```
suit > 1 / (0.35 * 40 * 4) = 0.018
```

Both species can sustain populations at very low suitability when uncrowded. Competition, not mortality, becomes the range limiter. This is ecologically correct — species ranges are set by competition at the margins, not by abiotic stress killing them outright.

### Suitability

Suitability is a pure function of weather and genome. No canopy shade interaction (that's competition, handled separately).

```
suit = gaussian(drought_stress, drought_tolerance, sigma) 
     * gaussian(temp_norm, warmth_preference, sigma)
```

**Critical change: absolute normalization.** Drought stress and temperature must use fixed reference ranges, not map-relative normalization. This ensures that a global precipitation shift from 1100mm to 1800mm actually changes suitability values.

```
drought_stress = 1.0 - clip(precipitation / PRECIP_REF_MAX, 0, 1)
temp_norm = clip((temperature - TEMP_REF_MIN) / (TEMP_REF_MAX - TEMP_REF_MIN), 0, 1)
```

Where `PRECIP_REF_MAX`, `TEMP_REF_MIN`, `TEMP_REF_MAX` are world-level constants set at creation based on the climate envelope (e.g., 3000mm, -10C, 30C). They don't change over time.

Sigma should be wide enough that most species are viable across a reasonable range. 0.25 worked previously; may need revisiting.

### Competition

Lotka-Volterra alpha matrix, as currently implemented. Species with similar genomes compete more strongly.

```
effective_load = sum(alpha[i,j] * density[j] for all species j)
available = max(0, K - effective_load)
```

Height-based dominance: taller species cast shade, reducing available light for shorter species. This is the one inter-species interaction that stays, because height is the primary competitive axis in plant ecology and produces the canopy/understory structure we want.

```
canopy_shade = sum(density[j] * height_factor[j] * shade_coeff for tall species j)
light_available = 1.0 - canopy_shade
```

For short species, suitability is additionally multiplied by `light_available` (or a function of it and their shade_tolerance). This keeps canopy structure meaningful without adding a separate mortality term.

### Dispersal

Dispersal happens every tick (not just fall). Keep the current two-mechanism approach:

1. **Local dispersal**: convolution kernel deposits a fraction of neighboring cell density. Radius determined by seed_mass trait (lighter seeds = wider kernel).
2. **Long-distance dispersal (LDD)**: random deposits from high-density cells into distant cells. Low probability, long range.

**No seed bank.** Dispersal is always from living density. Recovery after disturbance happens by recolonization from the edges — empty cells have zero effective_load, so growth is maximal when seeds arrive. This is the natural recovery mechanism and doesn't need a separate system.

If testing reveals that fire recovery is too slow without a seed bank, we can add a minimal version: fire converts density to a one-time "seed pulse" in neighboring cells, representing heat-triggered germination. But only if the simple model fails first.

### Disturbance events

Fire and blowdown remain as **discrete events**, not per-tick mortality. They are the primary perturbation mechanism:

- **Fire**: clears density in a burned area. Spread determined by moisture and fuel (total density). Recovery is natural via dispersal from unburned edges + reduced competition in empty cells.
- **Blowdown**: topples tall-species density in a patch. Opens canopy for understory/shade-intolerant species.

These are already roughly correct in the current code. May need minor adjustments to work with the simplified tick but the mechanic is sound.

### Density floor

Density below a threshold (0.001 or similar) snaps to zero. Prevents "dispersal dust" — trace populations that are ecologically meaningless but create visual noise and slow computation.

## Minimum viable genome

Six traits, each mapping to exactly one mechanism:

| Trait | Used by | Range |
|-------|---------|-------|
| `growth_rate` | Growth term | 0.1 - 0.5 |
| `lifespan` | Base turnover | 5 - 500 years |
| `drought_tolerance` | Suitability (moisture axis) | 0.0 - 1.0 |
| `frost_tolerance` | Suitability (temperature axis) | 0.0 - 1.0 |
| `max_height` | Competition (shade casting) | 0.3 - 30m |
| `dispersal_range` | Dispersal kernel width | 1 - 6 cells |

Deferred traits (add back only if they produce rider-visible effects):
- `seed_mass` — replaced by `dispersal_range` (more direct)
- `shade_tolerance` — may be needed for understory dynamics; test without first
- `phenological_aggressiveness` — was for spring frost risk; seasonal variation now comes from weather inputs
- `evergreenness` — visual trait for renderer, not simulation
- `mast_interval` — temporal texture, add back if boom/bust years are desired
- `biomass_age` tracking — incumbent advantage, add back if needed for stability

## Perturbation testing framework

A new development tool for validating ecology dynamics. The workflow:

1. **Create a world, run to equilibrium** under static weather (disable climate cycling). Verify all species reach stable ranges.
2. **Apply a single perturbation** and measure the response:
   - `--temperature +5`: raise temperature 5C. Cold-tolerant species should retreat, warm-tolerant should expand.
   - `--precipitation -500`: reduce precipitation. Drought-tolerant species should expand, wet species retreat.
   - `--fire-at 500,500 --radius 100`: clear a large area. Should recolonize from edges within ~50-100 years.
   - `--void-species anc_06`: remove one species entirely. Competitors should fill the gap.
3. **Validate using ride-compare**: the same path sampled before and after perturbation should show the expected shift.

Each perturbation test is a falsifiable claim:
- "Species track temperature shifts" — if they don't, the system is broken
- "Fire recovery happens in reasonable time" — if not, dispersal is too weak
- "Removing a dominant species allows competitors to expand" — if not, competition isn't working

This framework also validates new mechanics: add mast seeding, run the same perturbation tests, compare. If the tests still pass and the ride experience shows new texture, the mechanic earned its place.

### CLI interface

```
python -m bike_sim perturb <world_dir> --temperature +5
python -m bike_sim perturb <world_dir> --precipitation -500
python -m bike_sim perturb <world_dir> --fire-at 500,500 --radius 100
python -m bike_sim perturb <world_dir> --remove-species anc_06_valley_thicket
```

Each command snapshots the current state, applies the perturbation to the weather/ecology, then advances N years. The snapshot comparison tool shows the result.

## Migration plan

### Phase 1: Rewrite the tick
- Replace `_summer_growth_and_competition`, `_winter_mortality`, `_spring_leafout_and_frost`, `_spring_establishment`, `_summer_drought_mortality` with a single `_tick_ecology` method.
- Implement absolute suitability normalization.
- Strip genome to 6 traits.
- Remove seed bank system.
- Add density floor.
- Keep fire and blowdown as-is (minor interface adjustments).

### Phase 2: Validate
- Run to equilibrium under static conditions (disable weather cycling).
- Verify all species reach stable, spatially distinct ranges.
- Verify breakeven suitability matches analytical prediction.
- Verify competition creates range boundaries (not mortality).

### Phase 3: Perturbation tests
- Build the perturbation CLI.
- Run each perturbation test, verify expected responses.
- This becomes the regression test suite for ecology.

### Phase 4: Re-enable dynamics
- Turn weather cycling back on.
- Run calibration with ride-compare.
- Verify species ranges track climate shifts.
- Tune sigma, carrying capacity, dispersal rate as needed.

### Phase 5: Add flavor (only if earned)
- Consider shade_tolerance if canopy/understory structure needs work.
- Consider mast seeding if temporal texture is desired.
- Each addition goes through perturbation testing.

## Open questions

1. **Carrying capacity K**: currently 15.0 everywhere. Should it vary by terrain (higher in valleys, lower on ridges)? This would create natural density gradients without extra coefficients. But adds a coupling to geology that may not be needed yet.

2. **Suitability sigma**: 0.25 worked before but with the new tick math, species can survive at much lower suitability. May want to narrow sigma to create sharper niche boundaries. Test empirically.

3. **Dispersal every tick vs. less often**: dispersal every tick (4x/year) is more compute but creates smoother range dynamics. Could run dispersal every 4 ticks (annually) if performance matters. Test both.

4. **Shade tolerance**: the design above includes height-based competition (tall species shade short ones). But without a shade_tolerance trait, all short species are equally penalized. May need this trait for understory viability. Add in Phase 5 if understory species can't persist.

5. **Competition alpha niche_width**: currently 0.3. This controls how much similar species compete. Too narrow = every species is independent. Too wide = everyone competes with everyone. Needs empirical tuning but 0.3 is a reasonable starting point.

6. **What reference ranges for absolute normalization?** Options:
   - Fixed constants (PRECIP_REF_MAX=3000mm, TEMP_RANGE=-10 to 30C) — simple, predictable
   - Derived from world's geology/climate at creation — adapts to each world
   - Both feel reasonable; fixed constants are simpler to reason about

# Seasonal Redesign Specification

Implementation spec for the seasonal tick system, weather cycles, and related changes. This is a major restructuring of the simulation's temporal model, motivated by the experience-first design principles documented in `decisions.md`.

**Branch**: implement on a new branch off main.

**Scope**: weather system, seasonal ecology, per-season erosion, orchestrator, ride mapping. Does not include asset agent changes, renderer changes, or webview updates.

---

## Table of Contents

1. [Overview: what changes and why](#1-overview)
2. [Weather system](#2-weather-system)
3. [Seasonal ecology model](#3-seasonal-ecology-model)
4. [Per-season erosion](#4-per-season-erosion)
5. [Orchestrator and tick structure](#5-orchestrator-and-tick-structure)
6. [Deep history strategy](#6-deep-history-strategy)
7. [Ride mapping and config](#7-ride-mapping-and-config)
8. [Species genome changes](#8-species-genome-changes)
9. [Migration from current system](#9-migration-from-current-system)
10. [Implementation order](#10-implementation-order)
11. [Testing strategy](#11-testing-strategy)
12. [Open questions and tuning](#12-open-questions-and-tuning)

---

## 1. Overview

The current simulation ticks ecology every 5 years and climate-hydrology every 1,000 years. This creates artificial discontinuities: either no visible change for 20 rides, or a sudden lurch when a climate tick fires. The redesign switches to **seasonal ecology ticks** (~4/year) as the core simulation loop, with weather conditions generated per-season from overlapping deterministic cycles, and lightweight erosion applied continuously as a side effect of weather.

**Key numbers after redesign:**
- Ecology ticks per year: **4** (one per season)
- A 30-minute ride advances: **~30 seasons (~7.5 years)**
- Ecology ticks per ride: **~30**
- Weather cycle evaluations per tick: **~8 sinusoids** (negligible compute)
- Per-season erosion: **lightweight flow-proportional**, not 70K particles

---

## 2. Weather System

### 2.1 Architecture

New module: `src/bike_sim/weather.py`

```python
@dataclass
class WeatherCycle:
    period: float        # years
    amplitude: float     # strength (unitless multiplier or absolute offset)
    phase: float         # 0 to 2*pi, derived from seed
    target: str          # "temperature", "precipitation", or "both"
    correlation: float   # for "both": positive = warm&wet, negative = warm&dry

@dataclass
class SeasonalWeather:
    """Output for one season at one point in time."""
    temperature: np.ndarray      # (H, W) mean temp in C for this season
    precipitation: np.ndarray    # (H, W) total precip in mm for this season
    frost_severity: np.ndarray   # (H, W) frost damage potential [0, 1]
    storm_intensity: float       # world-level scalar, drives erosion
    season: int                  # 0=winter, 1=spring, 2=summer, 3=fall

class WeatherSystem:
    def __init__(self, world_seed: int, base_climate: BaseClimate):
        self.cycles = self._generate_cycles(world_seed)
        self.base = base_climate

    def generate(self, year: float, season: int) -> SeasonalWeather:
        """Deterministic weather for a given year and season."""
        ...
```

### 2.2 Cycle generation from seed

The world seed determines cycle parameters. Use `create_rng(seed, "weather", "cycles", 0)` to derive all cycle parameters deterministically.

**Cycle inventory (8 cycles per world):**

| Category | Count | Period range | Amplitude | Target | Purpose |
|----------|-------|-------------|-----------|--------|---------|
| Short precipitation | 1 | 3-7 years | +/-20-30% | precipitation | Year-to-year wet/dry texture |
| Short temperature | 1 | 4-8 years | +/-0.5-1.5 C | temperature | Year-to-year warm/cold texture |
| Medium precipitation | 1 | 25-50 years | +/-15-25% | precipitation | Ride-to-ride mood shifts |
| Medium temperature | 1 | 30-70 years | +/-1-2 C | temperature | Ride-to-ride mood shifts |
| Compound | 1-2 | 15-40 years | +/-10-15% / +/-1 C | both | World personality (warm=wet or warm=dry) |
| Long precipitation | 1 | 200-500 years | +/-10-20% | precipitation | Month-to-month drift |
| Long temperature | 1 | 300-800 years | +/-2-4 C | temperature | Month-to-month drift |
| Very long (drift) | 1 | 50,000-150,000 years | +/-3-6 C | both | Deep-time climate drift (ice ages, warm periods) |

Periods within each range are drawn uniformly. Amplitudes drawn uniformly within range. Phases drawn uniformly from [0, 2*pi]. Compound cycle correlation drawn from [-1, 1].

**Total: ~9 cycles.** All parameters are floats derived from seed. Stored in the World object so they persist across sessions.

### 2.3 Evaluating weather for a season

```python
def _evaluate_cycles(self, time: float) -> tuple[float, float]:
    """Sum cycle contributions at a given time point.

    Returns (temp_anomaly_C, precip_multiplier).
    Precipitation is multiplicative (can't go negative).
    """
    temp_anomaly = 0.0
    precip_log_anomaly = 0.0  # work in log space for multiplicative

    for cycle in self.cycles:
        value = cycle.amplitude * sin(2 * pi * time / cycle.period + cycle.phase)
        if cycle.target == "temperature":
            temp_anomaly += value
        elif cycle.target == "precipitation":
            precip_log_anomaly += value  # will exponentiate later
        elif cycle.target == "both":
            temp_anomaly += value
            precip_log_anomaly += value * cycle.correlation

    precip_multiplier = exp(precip_log_anomaly)  # always positive
    return temp_anomaly, precip_multiplier
```

### 2.4 Spatial application

The cycle anomalies are world-level scalars. Spatial variation comes from the existing terrain:

1. **Base temperature**: latitude gradient + elevation lapse rate (from current `_compute_climate`)
2. **Apply temperature anomaly**: `temperature = base_temp + temp_anomaly`
   - Valleys get amplified anomaly: `* (1 + valley_depth_factor * 0.3)` (cold air pooling)
3. **Base precipitation**: elevation + orographic effects (from current `_compute_climate`)
4. **Apply precipitation multiplier**: `precipitation = base_precip * precip_multiplier`
   - Mountains amplify: multiply the multiplier by `(1 + elevation_factor * 0.2)`
   - Rain shadows dampen: reduce multiplier on lee side
5. **Seasonal modulation**: base values are scaled by season:
   - Winter: temp * 0.6, precip * 0.8 (as snow)
   - Spring: temp * 0.85, precip * 1.1 (snowmelt adds moisture)
   - Summer: temp * 1.2, precip * 0.7 (warmer, drier)
   - Fall: temp * 0.9, precip * 1.0

These seasonal scalars are starting points for tuning.

### 2.5 Frost model

Late frost severity for spring:

1. Compute a "frost risk" from the spring temperature field: `frost_risk = sigmoid(frost_threshold - temperature)`
2. Add within-season variance: draw a frost event magnitude from `Normal(0, sigma)` where `sigma` increases when temperature anomaly is large (unsettled weather in unusual years)
3. Frost severity = `max(0, frost_risk + frost_event)`

Frost is spatially varied (higher elevation = more frost, valleys = cold pooling). Frost damage to plants is computed in the ecology tier based on phenological state.

### 2.6 Storm intensity

Per-season scalar derived from precipitation anomaly:

```python
storm_intensity = max(0, precip_multiplier - 1.0) * storm_scale_factor
```

When precipitation is above normal, storm intensity is positive. This drives per-season erosion magnitude. A season with 1.4x normal precipitation has significant storm-driven erosion; a 0.7x season has essentially none.

### 2.7 Persistence

Weather cycles (periods, amplitudes, phases) are stored in the `World` object alongside tier clocks. They are set once at world creation and never change. The very long cycle provides climate drift without modifying cycle parameters.

---

## 3. Seasonal Ecology Model

### 3.1 Season loop

Each ecology tick represents one season (0.25 years). The tick receives a `SeasonalWeather` object and performs season-appropriate operations:

```python
def tick(self, weather: SeasonalWeather) -> None:
    season = weather.season  # 0=winter, 1=spring, 2=summer, 3=fall

    if season == 0:  # Winter
        self._winter_mortality(weather)
    elif season == 1:  # Spring
        self._spring_leafout_and_frost(weather)
        self._spring_establishment(weather)
    elif season == 2:  # Summer
        self._summer_growth_and_competition(weather)
        self._summer_drought_mortality(weather)
        self._fire_disturbance(weather)
    elif season == 3:  # Fall
        self._seed_production_and_dispersal(weather)
        self._senescence_and_fuel(weather)
        self._blowdown_disturbance(weather)

    # Every tick:
    self._update_cumulative_drought(weather)
    self._update_biomass_age()
    self._promote_individuals(weather)
    self._check_speciation()  # only runs every N ticks
```

### 3.2 Winter tick

- **Dormancy check**: For each species, compute cold hardiness from `frost_tolerance` and `evergreenness`.
- **Winter kill**: Cells where `frost_severity > cold_hardiness` suffer mortality proportional to the excess. Kills a fraction of density, not all-or-nothing.
- **Evergreen activity**: Species with high `evergreenness` retain partial photosynthesis (slow growth even in winter, but at reduced rate).

### 3.3 Spring tick

- **Leaf-out timing**: Species with high `phenological_aggressiveness` leaf out in spring. Others wait until summer tick.
- **Late frost damage**: Species that leafed out (aggressive) take damage from `frost_severity`. Damage = `frost_severity * aggressiveness * (1 - frost_tolerance)`. This creates the gambling dynamic.
- **Spring establishment**: Recruitment from seed bank + current seed input, weighted by suitability under spring conditions. Early-leafing species get a competitive bonus (light access before canopy closes).

### 3.4 Summer tick

- **Main growth**: The primary competitive arena. Growth weighted by:
  - Suitability under summer conditions (moisture, temperature, light)
  - `growth_rate` trait
  - Available capacity (carrying capacity - total density)
  - Biomass/establishment age (established stands grow more efficiently)
- **Light competition**: `max_height` becomes load-bearing here. Taller species cast shade, reducing `solar_insolation` for shorter species in the same cell. Shade-tolerant species handle this; intolerant ones don't.
- **Drought mortality**: When summer precipitation is low, species with low `drought_tolerance` suffer. Mortality scales with cumulative drought stress (multi-year memory), not just this season's rainfall.
- **Fire**: Ignition probability scales with dryness (low precipitation, high temperature) and fuel load (from previous fall's senescence). Fire spread uses existing CA with moisture-dependent probability. Fire season is summer, matching the weather-ecology coupling.

### 3.5 Fall tick

- **Seed production**: Each species produces seeds proportional to density * growth_rate.
  - **Mast species** (`mast_interval` > 1): only produce seeds when `year % mast_interval == 0` (offset by species-specific phase from genome). In non-mast years, seed production is 10% of normal.
- **Dispersal**: Seeds spread via convolution kernel (radius from `seed_mass`). Long-distance dispersal events (low probability).
- **Seed bank input**: Fraction of produced seeds enters the bank. Bank decays each fall tick.
- **Senescence**: Deciduous species (`evergreenness` < threshold) drop leaves. Litter accumulates as fuel load for next summer's fire probability.
- **Blowdown**: Storm-driven windthrow. Probability scales with `storm_intensity` from weather. Exposed positions (ridges, edges) more vulnerable.

### 3.6 Cumulative drought stress

Per-cell float, updated every tick:

```python
def _update_cumulative_drought(self, weather):
    if weather.season == 2:  # Summer
        # Water deficit = how far below normal precipitation is
        deficit = max(0, normal_precip - weather.precipitation)
        self.drought_stress = self.drought_stress * 0.7 + deficit * 0.3
    else:
        # Slow recovery in non-summer seasons
        self.drought_stress *= 0.9
```

This creates a ~3-year rolling memory. One dry summer raises stress. Three in a row is devastating. A wet year brings it back down but slowly — the ecology "remembers" recent drought.

### 3.7 Biomass / establishment age

Per-species per-cell float. Increments each growth season:

```python
biomass_age[species] += density[species] * 0.25  # accumulates with density over time
```

Used as a competition modifier: established populations are harder to displace. Decays when density drops (disturbance resets it). This gives landscapes inertia — old-growth areas resist invasion, which is ecologically correct and makes the world feel like it has real history.

### 3.8 Carrying capacity

Remains at 15.0 (shared across all species per cell). This can be tuned later. The key constraint is that total density across all species in a cell cannot exceed this value, creating zero-sum competition.

### 3.9 Suitability model changes

The existing Gaussian match function stays. Changes:

- **Add `max_height` to light competition**: tall species reduce available light in a cell; shade-intolerant species penalized.
- **Use current season's weather** instead of static derived-state cache: `soil_moisture` derived from this season's precipitation + terrain + cumulative state, not a cached value from the last climate tick.
- **`distance_to_water`** becomes load-bearing: riparian species get a suitability bonus near water. Water cell identification updates with flow accumulation (recomputed periodically or after significant erosion events, not every tick — see section 4).
- **Frost tolerance interacts with phenological timing**: in spring, effective frost tolerance is reduced for species that already leafed out.

### 3.10 Distinguished individuals

Changes from current:

- **Lifecycle**: Individuals now age. Each seasonal tick increments their age by 0.25 years. When age exceeds the species' `lifespan` trait (with some stochastic variation), the individual dies. `died_year` gets set.
- **Post-mortem**: Dead individuals persist as snags for ~10 years, then logs for ~50 years, then mounds for ~200 years. Represented by a `state` field: alive -> snag -> log -> mound -> removed.
- **Disturbance interaction**: Fire and blowdown can kill individuals. A distinguished tree killed by fire becomes a fire-scarred snag — narrative-rich.
- **Promotion criteria**: Unchanged (high density, local prominence, disturbance survival). Frequency adjusted for seasonal ticks: check every 4 ticks (annually) instead of every tick.

### 3.11 Speciation

Timing: check every **200 ticks (50 years)**, matching the current effective rate. The connected-component analysis on the downsampled grid is unchanged. Genome drift now includes morphological traits (see section 8).

---

## 4. Per-Season Erosion

### 4.1 Approach

Replace the 70K-particle batch erosion with a lightweight per-season operation. The idea: each season's rainfall drives proportional erosion along existing drainage paths.

New function in `erosion.py`:

```python
def erode_seasonal(
    heightmap: np.ndarray,
    sediment: np.ndarray,
    flow_accumulation: np.ndarray,
    precipitation: np.ndarray,
    storm_intensity: float,
    bedrock_erodibility: np.ndarray,
    params: SeasonalErosionParams,
) -> None:
    """Lightweight per-season erosion. Modifies heightmap and sediment in-place."""
```

### 4.2 Algorithm

1. **Erosion potential** per cell = `flow_accumulation * precipitation * storm_intensity * slope * erodibility`
2. **Erode**: remove material proportional to erosion potential. Sediment first, then bedrock.
3. **Deposit**: material deposited where slope decreases (flow slows). Simple downhill redistribution.
4. **Scale factor**: calibrated so that ~1000 years of seasonal erosion (4000 ticks) produces comparable total erosion to the current 1-tick batch system (~11m mean). This means each seasonal tick erodes on the order of ~3mm mean — imperceptible per ride, cumulative over months.

### 4.3 Flow accumulation updates

Flow accumulation is expensive (the D8 sort-and-accumulate pass). Don't recompute every season.

- **Recompute every N ticks** (e.g., every 40 ticks = 10 years) or when cumulative erosion since last recomputation exceeds a threshold.
- **Between recomputations**, use the cached flow accumulation. Drainage patterns change slowly enough that stale-by-a-few-years flow data is fine.
- **After major events** (large flood, significant erosion from extreme storm), trigger an immediate recomputation.

### 4.4 Thermal diffusion

The slow landform smoothing. Applied every tick as a trivial operation:

```python
# Tiny diffusion: ~0.001mm per tick, imperceptible per ride
heightmap += laplacian(heightmap) * diffusion_rate
```

Over 1000 years (4000 ticks), this produces subtle smoothing of sharp features. Over 100K years, visible skyline changes. Costs essentially nothing.

---

## 5. Orchestrator and Tick Structure

### 5.1 New orchestrator loop

```python
def advance(self, num_seasons: int) -> None:
    """Advance the world by a given number of seasonal ticks."""
    next_version = self._world.next_version_id()
    self._world.rasters.set_version(next_version)

    eco = EcologyTier(self._world)
    weather = WeatherSystem(self._world)

    for i in range(num_seasons):
        year = eco_clock.simulated_year
        season = eco_clock.season  # 0-3

        # 1. Generate weather for this season
        seasonal_weather = weather.generate(year, season)

        # 2. Tick ecology (reads weather)
        eco.tick(seasonal_weather)

        # 3. Lightweight erosion (driven by weather)
        self._seasonal_erosion(seasonal_weather)

        # 4. Thermal diffusion (tiny, every tick)
        self._thermal_diffusion()

        # 5. Recompute flow accumulation if needed
        if self._erosion_recompute_needed():
            self._recompute_flow()

        # 6. Advance clock
        eco_clock.season = (eco_clock.season + 1) % 4
        if eco_clock.season == 0:
            eco_clock.simulated_year += 1

    self._world.commit_version(trigger=f"advance {num_seasons} seasons")
```

### 5.2 Tier clock changes

The ecology tier clock needs a `season` field (0-3) in addition to `simulated_year` and `tick_number`. The climate-hydrology tier clock becomes less central — the weather system doesn't "tick" in the traditional sense; it evaluates cycles at any requested time point.

### 5.3 Geology tier

Unchanged. Still ticks every 100K years (effectively never during normal play). The very long weather cycle handles climate drift without needing geology to tick.

### 5.4 Climate-hydrology tier

The tier as a separate ticking entity is **deprecated for normal advancement**. Its responsibilities are redistributed:

- **Climate envelope** → absorbed into `WeatherSystem` (base climate from terrain, seasonal conditions from cycles)
- **Erosion** → `erode_seasonal()` called per tick from orchestrator
- **Flow accumulation** → recomputed periodically from orchestrator
- **Derived-state cache** → replaced by per-season `SeasonalWeather` object passed to ecology

The tier class may still exist for deep-history simulation (section 6) where batched operation is appropriate.

---

## 6. Deep History Strategy

### 6.1 Two-phase world creation

World creation proceeds in two phases:

**Phase A: Deep history (coarse ticks)**
- Simulate ~200M years of geology (unchanged)
- Simulate ~200K years of climate-hydrology using the **existing batched system** (large erosion passes, 1000-year ticks)
- Simulate ~10K years of coarse ecology (5-year ticks, simplified model — no seasonal distinction, just establishment + competition + speciation)
- Purpose: produce plausible terrain, drainage, species radiation, rough vegetation distribution

**Phase B: Recent history (seasonal ticks)**
- Switch to the seasonal tick system for the final ~1,000 years
- Weather cycles are active; ecology responds seasonally
- Purpose: bring the world into a seasonally-resolved state with realistic vegetation patterns, seed banks, drought stress memory, etc.

The boundary between phases (how many years of seasonal history to simulate) is configurable. Starting point: 1,000 years (4,000 seasonal ticks). This is enough to establish seasonal dynamics and weather cycle patterns without excessive creation time.

### 6.2 Deep-history-on-demand

The coarse-tick simulation remains available as a CLI command for manual use:

```
python -m bike_sim advance --years 100000 --mode coarse
```

This uses the old batched system (simplified ecology, batched erosion) to fast-forward through deep time. Useful for testing world longevity, recovering from weird states, or just seeing what happens over long timescales.

---

## 7. Ride Mapping and Config

### 7.1 Default mapping

```yaml
# config/advancement.yaml (or section in world config)
advancement:
  seasons_per_minute: 1.0    # 1 season per minute of riding
  max_seasons_per_ride: 120  # cap at 30 simulated years
```

A 30-minute ride advances 30 seasons (7.5 years). A 60-minute ride advances 60 seasons (15 years). Cap prevents extreme rides from fast-forwarding too far.

### 7.2 Derived numbers at default settings

| Riding pattern | Seasons advanced | Simulated years | What's visible |
|---------------|-----------------|-----------------|----------------|
| One 30-min ride | 30 | 7.5 | Subtle vegetation shifts, seasonal change |
| One week (7 rides) | 210 | 52.5 | Species range changes, drought effects |
| One month (20 rides) | 600 | 150 | Medium cycle effects, possible fire/recovery |
| One year (250 rides) | 7,500 | 1,875 | Long cycle effects, speciation, significant landscape change |

### 7.3 Tuning

The `seasons_per_minute` value is the primary tuning knob. Making it configurable as a simple float means frequent adjustment during development. Other knobs:

- `max_seasons_per_ride`: safety cap
- Phase B duration at world creation: how much seasonal history to pre-simulate

---

## 8. Species Genome Changes

### 8.1 Full trait list

**Functional (10 traits):**

| Trait | Range | Status |
|-------|-------|--------|
| `drought_tolerance` | [0, 1] | Existing, keep |
| `frost_tolerance` | [0, 1] | Existing, keep |
| `shade_tolerance` | [0, 1] | Existing, keep |
| `growth_rate` | [0, 1] | Existing, keep |
| `seed_mass` | [0, 1] | Existing, keep |
| `max_height` | [0, inf) | Existing, **make load-bearing** (canopy competition) |
| `lifespan` | [1, inf) | Existing, keep |
| `phenological_aggressiveness` | [0, 1] | **New** |
| `evergreenness` | [0, 1] | **New** |
| `mast_interval` | [1, 7] | **New** |

**Morphological (7 traits):**

| Trait | Range | Initialization coupling |
|-------|-------|------------------------|
| `growth_form` | 0-4 (enum) | From `max_height` + `lifespan` |
| `leaf_size` | [0, 1] | `shade_tolerance` (+), `drought_tolerance` (-) |
| `leaf_shape` | [0, 1] (needle-broad) | `evergreenness` (evergreen -> needle) |
| `flower_color` | [0, 1] (hue) | Independent (random) |
| `flower_size` | [0, 1] | `seed_mass` (-), `evergreenness` (-) |
| `bark_texture` | [0, 1] (smooth-rough) | `lifespan` (+) |
| `stem_woodiness` | [0, 1] (herb-woody) | `growth_rate` (-), `lifespan` (+) |

### 8.2 Ancestor templates

The 6 existing ancestors need updating to include new traits. Starting values:

| Ancestor | pheno_aggr | evergreen | mast_int | flower_size | Notes |
|----------|-----------|-----------|----------|-------------|-------|
| lowland_herb | 0.7 | 0.1 | 1 | 0.6 | Aggressive spring ephemeral, showy flowers |
| upland_grass | 0.4 | 0.3 | 1 | 0.1 | Moderate timing, inconspicuous flowers |
| valley_tree | 0.3 | 0.2 | 4 | 0.4 | Conservative, mast seeder |
| ridge_shrub | 0.2 | 0.7 | 2 | 0.3 | Conservative, semi-evergreen |
| pioneer_forb | 0.8 | 0.0 | 1 | 0.7 | Very aggressive, showy, annual strategy |
| alpine_cushion | 0.1 | 0.8 | 3 | 0.2 | Very conservative, mostly evergreen |

### 8.3 Speciation drift rates

During speciation, traits drift with normally-distributed perturbation:

- **Functional traits** (bounded): `Normal(0, 0.08)`, clipped to valid range (unchanged from current)
- **Morphological traits**: `Normal(0, 0.15)`, clipped to valid range (**higher variance** — visual divergence should be faster than functional divergence)
- **`growth_form`**: discrete; 10% chance of shifting one category during speciation
- **`mast_interval`**: integer; ±1 with 20% probability during speciation

---

## 9. Migration from Current System

### 9.1 What gets replaced

| Current component | Replacement |
|-------------------|-------------|
| `ClimateHydrologyTier.tick()` (batched) | `WeatherSystem.generate()` + `erode_seasonal()` per tick |
| `EcologyTier.tick()` (5-year) | `EcologyTier.tick(weather)` (seasonal) |
| `Orchestrator.advance(years)` | `Orchestrator.advance(num_seasons)` |
| Static derived-state cache | Per-season `SeasonalWeather` object |
| `ErosionParams` + 70K particles | `SeasonalErosionParams` + lightweight flow-proportional |
| 7-trait genome | 17-trait genome (10 functional + 7 morphological) |

### 9.2 What stays

- `World` object structure (seed, tier clocks, raster store, event store)
- `RasterStore` and `EventStore` (schema additions for new fields, no structural changes)
- Geology tier (unchanged)
- Versioning system (unchanged)
- Webview (will need layer updates later but not part of this spec)
- CLI structure (commands stay, argument semantics change)
- `erosion.py` module (kept for deep-history batched erosion; new `erode_seasonal` added alongside)
- D8 flow direction/accumulation code (moved to shared utility, used by both seasonal and batched paths)

### 9.3 Existing worlds

Existing worlds created under the old system will **not be compatible** with the new system. This is acceptable — we're on a new branch and worlds can be regenerated. No migration tooling needed (per existing decision in `decisions.md`).

---

## 10. Implementation Order

Work is structured so each step is independently testable and produces visible results through the webview.

### Step 1: Weather system (standalone, no ecology changes)

- Implement `WeatherSystem` class with cycle generation from seed
- Implement `SeasonalWeather` generation
- Unit tests: determinism from seed, cycle superposition math, spatial application
- **Verification**: plot weather time series for a few seeds; visually confirm cycle interference produces expected fat-tailed distribution

### Step 2: Genome expansion

- Add new functional traits (`phenological_aggressiveness`, `evergreenness`, `mast_interval`) to ancestor templates and species creation
- Add morphological traits with initialization coupling
- Update speciation drift to include new traits
- Update `EventStore` schema if needed (genome is JSON, so likely no schema change)
- Unit tests: genome creation, speciation drift, morphological coupling
- **Verification**: create a world, inspect species genomes, verify trait distributions make sense

### Step 3: Seasonal ecology model

- Rewrite `EcologyTier.tick()` to accept `SeasonalWeather` and branch by season
- Implement winter mortality, spring leaf-out/frost, summer growth/drought, fall seed/dispersal
- Add per-cell state buffers (cumulative drought stress, biomass age)
- Make `max_height` load-bearing (canopy shading)
- Recalibrate all rates for seasonal ticks (mortality, growth, dispersal, seed bank decay)
- Unit tests: seasonal invariants (no growth in winter for deciduous, frost kills aggressive species, drought stress accumulates)
- **Verification**: advance a world 100 years via CLI, inspect vegetation patterns in webview

### Step 4: Per-season erosion

- Implement `erode_seasonal()` in `erosion.py`
- Implement periodic flow accumulation recomputation
- Implement thermal diffusion
- Calibrate erosion rates so ~4000 ticks ≈ previous 1-tick batch result
- Unit tests: erosion proportional to storm intensity, sediment conservation
- **Verification**: advance 1000 years, compare terrain to old-system terrain

### Step 5: Orchestrator rewrite

- Rewrite `Orchestrator.advance()` to use seasonal loop
- Integrate weather generation, seasonal ecology, per-season erosion
- Implement two-phase world creation (deep history + seasonal recent history)
- Update CLI (`advance` takes seasons or years with `--seasonal` flag)
- Update ride mapping to use `seasons_per_minute` config
- Integration tests: full advance cycle, version creation, reproducibility from seed
- **Verification**: create a world end-to-end, advance via simulated ride, inspect in webview

### Step 6: Distinguished individual lifecycle

- Add aging (0.25 years per tick)
- Add death from age (stochastic around lifespan)
- Add post-mortem states (snag -> log -> mound -> removed)
- Add death from disturbance (fire, blowdown)
- Unit tests: individuals age, die, transition through post-mortem states
- **Verification**: advance a world 500 years, inspect individual lifespans and death events

### Step 7: Calibration and tuning

- Run worlds for 1000, 5000, 10000 simulated years
- Evaluate: does species diversity stabilize, oscillate, or collapse?
- Tune weather cycle amplitudes, ecological rates, erosion intensity
- Evaluate ride-to-ride change at different `seasons_per_minute` settings
- Document tuning results and any parameter changes

---

## 11. Testing Strategy

### Determinism / reproducibility

- Same seed + same advance = bit-identical state (critical, test early and often)
- Weather system is purely functional (time in, weather out) — easy to test

### Invariants

- Total density per cell <= carrying capacity
- All densities >= 0
- Seed bank >= 0
- Cumulative drought stress >= 0
- Sediment depth >= 0
- Mass conservation in erosion (sediment removed = sediment deposited, approximately)

### Seasonal correctness

- Deciduous species: no growth in winter
- Evergreen species: some growth in winter
- Frost damage only in winter/spring
- Fire only in summer (primarily)
- Seed production only in fall
- Aggressive species: higher spring growth, higher frost damage

### Emergent behavior (observational, not assertion-based)

- Multi-year drought produces visible forest thinning
- Wet year after drought + fire produces recruitment pulse
- Mast species produce cohort stands
- Species range shifts track long-term climate cycles
- Speciation still produces ~15 species from 6 ancestors over ~125 years

### Performance

- 4000-season advance (1000 years): must complete overnight (~6 hours acceptable)
- World creation (deep history + 1000 years seasonal): must complete overnight
- Per-tick speed will improve with optimization; not a blocker yet

---

## 12. Open Questions and Tuning

### Parameters that need empirical tuning

- Weather cycle amplitudes (how extreme should rare alignments get?)
- Seasonal erosion rate (how much erosion per mm of precipitation?)
- Drought stress accumulation/decay rates
- Biomass age effect on competition (how much advantage does establishment give?)
- Fire ignition rate per summer tick (currently calibrated for 5-year ticks)
- `seasons_per_minute` mapping (starting at 1.0, likely needs adjustment)
- Number of seasonal ticks for Phase B of world creation

### Unresolved design questions

- **Species immigration**: should new species occasionally arrive from "outside" the world to prevent diversity collapse? If so, at what rate and with what traits?
- **Nutrient cycling**: litter from senescence could feed soil quality, which feeds back into suitability. Worth the complexity? Defer unless ecology feels too static.
- **Snow**: should winter precipitation accumulate as snowpack that releases in spring? Creates nice seasonal moisture dynamics but adds state.
- **Flood events**: should extreme precipitation produce discrete flood events (scour riparian zones, deposit sediment downstream)? Or is continuous erosion sufficient?
- **World staleness detection**: automatic detection of equilibrium / diversity collapse. Defer until we have long-run test data.

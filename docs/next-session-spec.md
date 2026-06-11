# Next Session Spec: Fractal Climate + Climate-Responsive Biotic Pressure

Two changes for the next session. Change 1 replaces the weather system's sinusoidal cycle engine with fractal noise, producing aperiodic, realistic climate variability. Change 2 makes the Janzen-Connell biotic pressure respond to weather conditions.

---

## 1. Fractal noise climate system

### Problem

The current weather system (`weather.py`) generates climate variability from 8-9 overlapping sinusoidal cycles. This produces periodic, predictable oscillations. No matter how many cycles are overlaid, the result repeats exactly at the LCM of all periods. Every "rare event" recurs on a fixed schedule. Real climate data shows trends within trends with stochastic variation — fractal structure, not discrete frequency peaks.

### Solution

Replace `_generate_cycles` / `_evaluate_cycles` with a 1D fractal noise function (fractional Brownian motion). The noise is evaluated at time `t` with multiple octaves, producing a continuous power spectrum where low frequencies dominate (red noise) and each "cycle" is unique.

### Design

#### Noise function

Use 1D value noise with linear interpolation, seeded from the world seed. Two independent noise streams: one for temperature anomaly, one for precipitation anomaly.

```python
def _noise1d(seed: int, t: float) -> float:
    """Deterministic 1D noise at position t, seeded."""
    # Hash-based: floor(t) and ceil(t) each produce a deterministic
    # value via seed mixing, then lerp between them.
    i = int(np.floor(t))
    frac = t - i
    v0 = _hash_float(seed, i)   # deterministic float in [-1, 1]
    v1 = _hash_float(seed, i+1)
    # Smooth interpolation (smoothstep or cosine)
    return v0 + (v1 - v0) * _smoothstep(frac)

def _hash_float(seed: int, i: int) -> float:
    """Deterministic float in [-1, 1] from seed and integer position."""
    # Use a fast integer hash (e.g., murmurhash-style bit mixing)
    h = seed ^ (i * 0x9E3779B9)
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = (h >> 16) ^ h
    return (h & 0xFFFFFF) / 0x800000 - 1.0  # map to [-1, 1]
```

#### Fractal evaluation (fBm)

```python
def _evaluate_climate(self, time: float) -> tuple[float, float]:
    temp_anomaly = 0.0
    precip_anomaly = 0.0

    for octave in range(self._num_octaves):
        freq = self._base_freq * (self._lacunarity ** octave)
        amp = self._base_amp_temp * (self._persistence ** octave)
        temp_anomaly += _noise1d(self._seed_temp, time * freq) * amp

        amp_p = self._base_amp_precip * (self._persistence ** octave)
        precip_anomaly += _noise1d(self._seed_precip, time * freq) * amp_p

    precip_multiplier = exp(precip_anomaly)  # always positive
    return temp_anomaly, precip_multiplier
```

#### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_octaves` | 7 | Number of noise layers |
| `base_freq` | 1/800 | Lowest octave wavelength ~800 years |
| `lacunarity` | 2.0 | Each octave doubles in frequency |
| `persistence` | 0.55 | Each octave has 55% the amplitude of the previous |
| `base_amp_temp` | 3.0 | Lowest octave: ±3°C temperature swing |
| `base_amp_precip` | 0.15 | Lowest octave: ±15% precipitation swing (in log space) |

This produces:
- **Octave 0**: ~800yr trends, ±3°C — major climate epochs
- **Octave 1**: ~400yr, ±1.65°C — century-scale shifts
- **Octave 2**: ~200yr, ±0.91°C — multi-century oscillations
- **Octave 3**: ~100yr, ±0.50°C — ride-month-to-ride-month drift
- **Octave 4**: ~50yr, ±0.27°C — decade-scale variability
- **Octave 5**: ~25yr, ±0.15°C — fine variation
- **Octave 6**: ~12yr, ±0.08°C — year-to-year noise

Maximum possible temperature anomaly (all octaves aligned): ~±6.6°C. This is rare — typical anomaly is ~±2-3°C (RMS of the series).

Maximum precipitation multiplier (all octaves aligned): exp(±0.43) ≈ 0.65 to 1.54. Extreme cases: exp(±0.6) ≈ 0.55 to 1.82. Well within physical plausibility.

#### Seeds

Derived from world seed:
```python
rng = create_rng(world_seed, "weather", "fractal", 0)
self._seed_temp = int(rng.integers(0, 2**31))
self._seed_precip = int(rng.integers(0, 2**31))
```

Temperature and precipitation get independent noise streams. They're uncorrelated by default — sometimes warm and wet, sometimes warm and dry. This is more realistic than the current system's explicit `correlation` parameter.

#### What to remove

- `WeatherCycle` dataclass — no longer needed
- `_generate_cycles()` — replaced by seed derivation
- `_evaluate_cycles()` — replaced by `_evaluate_climate()`
- `get_cycles()` — replaced with `get_climate_params()` or similar

#### What stays unchanged

- `SeasonalWeather` dataclass
- `_SEASON_SCALES` — seasonal modulation still applies on top of anomalies
- `generate()` method — still the public API, still takes (year, season)
- `_compute_base_climate()` — terrain-based baseline unchanged
- `_compute_valley_depth()`, `_compute_frost()` — unchanged
- Spatial application of anomalies (valley amplification, mountain boost, rain shadow)

#### Migration

The `WeatherCycle` objects are currently stored in the World object. Replace with the fractal parameters (num_octaves, base_freq, lacunarity, persistence, base_amp_temp, base_amp_precip, seed_temp, seed_precip). Existing worlds that have stored cycles will need a migration path — regenerate from world seed.

#### Capping / safety

The current system caps anomalies at ±5°C and precip_multiplier at [0.3, 2.5]. The fractal system should use similar caps but they should rarely be hit — the natural falloff from persistence means extreme alignment is genuinely rare, not artificially clamped.

Keep caps as safety rails:
- Temperature: ±8°C (wider than before, since extremes should be possible but very rare)
- Precipitation multiplier: [0.25, 3.0] (wider, same reasoning)

#### Verification

After implementation, generate a 5000-year climate trace and plot it. It should show:
- Red noise power spectrum (more power at low frequencies)
- No visible periodicity
- "Trends within trends" — zoom into any 200-year window and it looks structurally similar to the full trace
- Rare extreme excursions (maybe 1-2 in 5000 years that approach the caps)

### Impact on existing tests

Tests that construct `WeatherSystem` may need updating if they reference `cycles` or `WeatherCycle`. The `make_weather()` helper in test_seasonal_ecology.py constructs `SeasonalWeather` directly and won't be affected.

---

## 2. Climate-responsive biotic pressure

### Problem

The Janzen-Connell biotic pressure currently uses fixed parameters — `growth_k`, `decay_rate`, and `mortality_strength` are constants. In reality, pathogen and herbivore pressure varies with climate conditions: warm wet periods favor disease, cold dry periods suppress it. This means pressure should oscillate not just with species density but with environmental conditions.

### Solution

Modulate the biotic pressure parameters based on the current season's weather.

### Design

In `_apply_biotic_pressure()`, compute a climate modifier from the weather:

```python
def _apply_biotic_pressure(self, species_list, densities, weather):
    # Climate modifies how fast pressure builds and decays.
    # Warm + wet = pathogen-friendly = faster pressure buildup.
    # Cold + dry = pathogens suppressed = faster decay.
    mean_temp = float(weather.temperature.mean())
    mean_precip = float(weather.precipitation.mean())

    # Normalize to a 0-1 "pathogen favorability" score.
    # Baseline: ~8°C mean temp, ~1000mm precip.
    temp_factor = np.clip((mean_temp - 2.0) / 16.0, 0, 1)  # 2°C=0, 18°C=1
    precip_factor = np.clip(mean_precip / 2000.0, 0, 1)     # 0mm=0, 2000mm=1
    pathogen_favorability = temp_factor * precip_factor       # 0 to 1

    # Modulate parameters
    effective_growth_k = growth_k * (0.5 + pathogen_favorability)     # 50%-150% of base
    effective_decay = decay_rate + (1 - decay_rate) * 0.5 * (1 - pathogen_favorability)
    # When pathogen_favorability is low, decay is faster (pressure drops)
    # When high, decay stays at base rate (pressure persists)
```

### Changes to tick loop

Pass `weather` to `_apply_biotic_pressure()`:

```python
# In tick():
if tick_num >= 8:
    self._apply_biotic_pressure(species_list, densities, weather)
```

The method signature changes from `(self, species_list, densities)` to `(self, species_list, densities, weather)`.

### Expected behavior

- **Warm wet period**: pressure builds faster on dominant species → crashes happen sooner → more turnover
- **Cold dry period**: pressure decays → dominant species get a reprieve → monocultures can temporarily expand
- **Transition from cold→warm**: species that expanded during the cold window suddenly face pressure they haven't accumulated immunity to → potential crash
- **Transition from warm→cold**: pressure drops, but species are already depleted → slow recovery

This creates asymmetric dynamics around climate transitions — exactly the kind of "perturbation followed by recovery into a new equilibrium" that we want.

### Impact on tests

Tests that call `_apply_biotic_pressure` directly (if any) will need the weather argument. The tick-level tests use `make_weather()` which already provides temperature and precipitation.

---

## Implementation order

1. Implement the 1D noise function and fractal evaluation
2. Replace cycle generation/evaluation in WeatherSystem
3. Update WeatherSystem.__init__ to store fractal params instead of cycles
4. Verify with a climate trace plot
5. Add weather parameter to biotic pressure
6. Implement climate modulation of pressure parameters
7. Run tests, fix failures
8. Generate calibration world and compare dynamics

## Not in scope

- Climate-responsive speciation thresholds (deferred — want to see if fractal climate + responsive pressure is sufficient)
- Lineage age effects on pressure (interesting but secondary)
- Changes to the speciation mechanism itself

"""Diagnostic script: compare suitability vs density across climate snapshots.

Checks whether species distributions track shifting suitability as
precipitation cycles through highs and lows.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bike_sim.world import World
from bike_sim.weather import WeatherSystem

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORLD_PATH = Path(__file__).resolve().parent.parent / "worlds" / "calibration_v12"
OUTPUT_DIR = WORLD_PATH / "ride_output"

# Species to analyze
SPECIES = [
    "anc_06_valley_thicket",
    "anc_05_ridge_scrub",
    "anc_01_upland_conifer",
]

# Load diagnostics to pick contrasting precip versions
with open(WORLD_PATH / "diagnostics.jsonl") as f:
    diags = [json.loads(line) for line in f]

# Build version -> precip map
version_precip = {}
version_temp = {}
for d in diags:
    v = int(d["year"] / 100)
    version_precip[v] = d["climate"]["mean_precip"]
    version_temp[v] = d["climate"]["mean_temp"]

# Sort by precip to find contrasting pairs
sorted_by_precip = sorted(version_precip.items(), key=lambda x: x[1])
print("All versions sorted by precip:")
for v, p in sorted_by_precip:
    print(f"  v={v:2d}  year={v*100:5d}  precip={p:7.1f}mm  temp={version_temp[v]:.1f}C")

# Pick 4 versions: 2 driest, 2 wettest (spread across time)
driest = sorted_by_precip[:5]   # candidates
wettest = sorted_by_precip[-5:] # candidates

# Pick spread: one early-dry, one late-dry, one early-wet, one late-wet
driest_sorted_by_time = sorted(driest, key=lambda x: x[0])
wettest_sorted_by_time = sorted(wettest, key=lambda x: x[0])

VERSIONS = [
    driest_sorted_by_time[0][0],   # earliest dry
    driest_sorted_by_time[-1][0],  # latest dry
    wettest_sorted_by_time[0][0],  # earliest wet
    wettest_sorted_by_time[-1][0], # latest wet
]
VERSIONS = sorted(VERSIONS)

print(f"\nSelected versions: {VERSIONS}")
for v in VERSIONS:
    print(f"  v={v}  year={v*100}  precip={version_precip[v]:.1f}mm  temp={version_temp[v]:.1f}C")

# ---------------------------------------------------------------------------
# Open world and set up weather system
# ---------------------------------------------------------------------------

world = World.open(WORLD_PATH)
store = world.rasters

# Read geology heightmap (version-invariant, written at creation)
heightmap = store.read_layer("geology", "heightmap", version=1)

# Load spatial bias fields
ch_layers = store.list_layers("climate_hydrology")
moisture_bias = (
    store.read_layer("climate_hydrology", "moisture_bias", version=1)
    if "moisture_bias" in ch_layers else None
)
continentality = (
    store.read_layer("climate_hydrology", "continentality", version=1)
    if "continentality" in ch_layers else None
)

weather_sys = WeatherSystem(
    world.seed, heightmap,
    moisture_bias=moisture_bias,
    continentality=continentality,
)

# Load species genomes
genomes = {}
for sid in SPECIES:
    sp = world.events.get_species(sid)
    genomes[sid] = sp["genome"]
    print(f"\n{sid} genome:")
    for k in ["drought_tolerance", "frost_tolerance", "shade_tolerance",
              "growth_rate", "seed_mass", "max_height"]:
        print(f"  {k}: {genomes[sid].get(k, 'N/A')}")


# ---------------------------------------------------------------------------
# Suitability computation (extracted from EcologyTier)
# ---------------------------------------------------------------------------

def _gaussian_match(field, optimum, sigma=0.3):
    return np.exp(-((field - optimum) ** 2) / (2 * sigma ** 2))


def compute_suitability(genome, weather):
    """Replicate EcologyTier._compute_suitability_from_weather without canopy."""
    n = weather.temperature.shape[0]
    suit = np.ones((n, n), dtype=np.float64)

    # Moisture suitability
    p_min = float(weather.precipitation.min())
    p_max = float(weather.precipitation.max())
    p_span = p_max - p_min
    if p_span > 200.0:
        precip_norm = np.clip((weather.precipitation - p_min) / p_span, 0, 1)
    else:
        precip_norm = np.clip(weather.precipitation / 3000.0, 0, 1)
    drought_stress = 1.0 - precip_norm
    suit *= _gaussian_match(drought_stress, genome["drought_tolerance"], sigma=0.25)

    # Temperature suitability
    t_min = float(weather.temperature.min())
    t_max = float(weather.temperature.max())
    t_span = t_max - t_min
    if t_span > 5.0:
        temp_norm = np.clip((weather.temperature - t_min) / t_span, 0, 1)
    else:
        temp_norm = np.clip((weather.temperature - (-10.0)) / 35.0, 0, 1)
    warmth_preference = 1.0 - genome["frost_tolerance"]
    suit *= _gaussian_match(temp_norm, warmth_preference, sigma=0.25)

    return suit


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------

results = {}  # species_id -> list of dicts per version

for sid in SPECIES:
    results[sid] = []
    genome = genomes[sid]

    for v in VERSIONS:
        year = v * 100.0
        tick = v * 400  # 400 ticks per 100 years

        # Generate weather for summer (season=2) at that year —
        # summer is the primary growing season
        weather = weather_sys.generate(year, season=2)

        # Compute suitability
        suit = compute_suitability(genome, weather)

        # Read density
        density_layer = f"species_{sid}_density"
        try:
            density = store.read_layer("ecology", density_layer, version=v)
        except KeyError:
            print(f"  WARNING: density layer not found for {sid} at version {v}")
            density = np.zeros_like(suit)

        # Compute metrics
        occupied = density > 0.01
        suitable = suit > 0.15

        occupied_count = int(occupied.sum())
        suitable_count = int(suitable.sum())

        mean_suit_in_suitable = float(suit[suitable].mean()) if suitable.any() else 0.0
        mean_density_in_occupied = float(density[occupied].mean()) if occupied.any() else 0.0

        # Overlap: % of suitable cells that are occupied
        if suitable_count > 0:
            overlap = float((suitable & occupied).sum()) / suitable_count * 100
        else:
            overlap = 0.0

        results[sid].append({
            "version": v,
            "year": year,
            "precip": version_precip[v],
            "temp": version_temp[v],
            "suit": suit,
            "density": density,
            "occupied_count": occupied_count,
            "suitable_count": suitable_count,
            "mean_suit": mean_suit_in_suitable,
            "mean_density": mean_density_in_occupied,
            "overlap": overlap,
        })


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 120)
print(f"{'Species':<28} {'Version':>7} {'Year':>6} {'Precip':>9} {'Suitable':>10} {'Occupied':>10} "
      f"{'MeanSuit':>9} {'MeanDens':>9} {'Overlap%':>9}")
print("-" * 120)

for sid in SPECIES:
    for r in results[sid]:
        print(f"{sid:<28} {r['version']:>7} {r['year']:>6.0f} {r['precip']:>8.1f}mm "
              f"{r['suitable_count']:>10} {r['occupied_count']:>10} "
              f"{r['mean_suit']:>9.4f} {r['mean_density']:>9.4f} {r['overlap']:>8.1f}%")
    print()

print("=" * 120)


# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------

for sid in SPECIES:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{sid}  (drought_tol={genomes[sid]['drought_tolerance']:.2f}, "
                 f"frost_tol={genomes[sid]['frost_tolerance']:.2f})", fontsize=14)

    # Find common colorscales
    all_suit = np.concatenate([r["suit"].ravel() for r in results[sid]])
    all_dens = np.concatenate([r["density"].ravel() for r in results[sid]])
    suit_vmax = np.percentile(all_suit, 99)
    dens_vmax = max(np.percentile(all_dens, 99), 0.01)

    for col, r in enumerate(results[sid]):
        v = r["version"]
        year = r["year"]
        precip = r["precip"]

        # Top row: suitability
        ax = axes[0, col]
        im = ax.imshow(r["suit"], vmin=0, vmax=max(suit_vmax, 0.01), cmap="YlGn", origin="upper")
        ax.set_title(f"Year {year:.0f}\nprecip={precip:.0f}mm\ntemp={r['temp']:.1f}C", fontsize=10)
        if col == 0:
            ax.set_ylabel("Suitability", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: density
        ax = axes[1, col]
        im = ax.imshow(r["density"], vmin=0, vmax=dens_vmax, cmap="YlOrRd", origin="upper")
        ax.set_title(f"suit>{0.15}: {r['suitable_count']}\nocc>{0.01}: {r['occupied_count']}\noverlap: {r['overlap']:.1f}%",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    outpath = OUTPUT_DIR / f"diagnostic_suitability_{sid}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}")

# ---------------------------------------------------------------------------
# Additional diagnostic: check if suitability MAPS actually differ
# ---------------------------------------------------------------------------

print("\n\nSuitability spatial correlation between versions (per species):")
print("-" * 80)
for sid in SPECIES:
    rs = results[sid]
    for i in range(len(rs)):
        for j in range(i+1, len(rs)):
            corr = np.corrcoef(rs[i]["suit"].ravel(), rs[j]["suit"].ravel())[0, 1]
            print(f"  {sid}: v{rs[i]['version']} vs v{rs[j]['version']}  r={corr:.6f}")

print("\n\nPrecipitation field stats at each version (summer):")
print("-" * 80)
for v in VERSIONS:
    weather = weather_sys.generate(v * 100.0, season=2)
    p = weather.precipitation
    t = weather.temperature
    print(f"  v={v:2d}: precip min={p.min():.0f} max={p.max():.0f} mean={p.mean():.0f} span={p.max()-p.min():.0f}  |  "
          f"temp min={t.min():.1f} max={t.max():.1f} mean={t.mean():.1f} span={t.max()-t.min():.1f}")

world.close()
print("\nDone.")

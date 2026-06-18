"""Ecology testing framework — equilibrium validation and perturbation tests.

Usage:
    # Run to equilibrium with static weather
    python scripts/test_ecology.py equilibrium worlds/eq_test --seed 42

    # Apply perturbation and run to new equilibrium
    python scripts/test_ecology.py perturb worlds/eq_test --temperature +5 --years 500
    python scripts/test_ecology.py perturb worlds/eq_test --fire 500,500 --radius 100 --years 200
    python scripts/test_ecology.py perturb worlds/eq_test --precipitation -500 --years 500
    python scripts/test_ecology.py perturb worlds/eq_test --remove-species anc_06_valley_thicket --years 500

    # Report on all experiments
    python scripts/test_ecology.py report worlds/eq_test

Equilibrium test: creates a world, runs with static weather until species
populations stabilize (max per-species density change < 2% for 2 consecutive
100yr chunks). Records strip samples (N-S and E-W through center).

Perturbation test: copies the equilibrium snapshot, applies a perturbation,
runs until new equilibrium or max years. Records time-to-equilibrium and
strip samples showing how distributions shifted.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from bike_sim.world import World
from bike_sim.orchestrator import Orchestrator
from bike_sim.weather import WeatherSystem
from bike_sim.tiers.ecology import EcologyTier, _ANCESTOR_TEMPLATES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKS_PER_YEAR = 4
STABILITY_THRESHOLD = 0.02   # 2% max density change per epoch
STABILITY_EPOCHS = 2         # consecutive stable epochs needed
EPOCH_YEARS = 100
EPOCH_TICKS = EPOCH_YEARS * TICKS_PER_YEAR
MAX_EQUILIBRIUM_YEARS = 2000
MIN_PERTURBATION_YEARS = 50

GRID_SIZE = 1000
CELL_SIZE = 50.0


# ---------------------------------------------------------------------------
# Strip sampling
# ---------------------------------------------------------------------------

def sample_strips(world: World) -> dict:
    """Sample species density along N-S and E-W strips through world center.

    Returns dict with:
    - ns_strip: {species_id: [density values along N-S center column]}
    - ew_strip: {species_id: [density values along E-W center row]}
    - positions_m: [position in meters along the strip]
    """
    store = world.rasters
    center = GRID_SIZE // 2
    import re
    pattern = re.compile(r"^species_(.+)_density$")

    ns_data: dict[str, list[float]] = {}
    ew_data: dict[str, list[float]] = {}

    for layer_name in store.list_layers("ecology"):
        m = pattern.match(layer_name)
        if not m:
            continue
        sid = m.group(1)
        density = store.read_layer("ecology", layer_name)

        # N-S strip: center column, all rows
        ns_data[sid] = density[:, center].tolist()
        # E-W strip: center row, all columns
        ew_data[sid] = density[center, :].tolist()

    positions = [i * CELL_SIZE + CELL_SIZE / 2 for i in range(GRID_SIZE)]

    return {
        "ns_strip": ns_data,
        "ew_strip": ew_data,
        "positions_m": positions,
    }


def compute_species_summary(world: World) -> list[dict]:
    """Compute per-species summary statistics."""
    store = world.rasters
    current_year = world.tier_clocks["ecology"].simulated_year
    species_list = world.events.list_species(alive_at_year=current_year)
    import re
    pattern = re.compile(r"^species_(.+)_density$")

    summaries = []
    for sp in species_list:
        sid = sp["species_id"]
        layer = f"species_{sid}_density"
        if layer not in store.list_layers("ecology"):
            continue
        density = store.read_layer("ecology", layer)
        total = float(density.sum())
        occupied = int((density > 0.001).sum())

        # Centroid
        if occupied > 0:
            rows, cols = np.where(density > 0.001)
            centroid_r = float(np.average(rows, weights=density[rows, cols]))
            centroid_c = float(np.average(cols, weights=density[rows, cols]))
        else:
            centroid_r = centroid_c = 0.0

        # Analytical breakeven
        genome = world.events.get_species(sid)["genome"]
        growth_rate = genome["growth_rate"]
        lifespan = genome["lifespan"]
        breakeven = 1.0 / (growth_rate * lifespan * TICKS_PER_YEAR)

        summaries.append({
            "species_id": sid,
            "total_density": round(total, 1),
            "occupied_cells": occupied,
            "centroid_r": round(centroid_r, 1),
            "centroid_c": round(centroid_c, 1),
            "breakeven_suit": round(breakeven, 4),
            "growth_rate": growth_rate,
            "lifespan": lifespan,
        })

    return sorted(summaries, key=lambda s: -s["total_density"])


# ---------------------------------------------------------------------------
# Equilibrium detection
# ---------------------------------------------------------------------------

def check_stability(prev_densities: dict[str, float], curr_densities: dict[str, float]) -> tuple[bool, float]:
    """Check if all species densities changed less than STABILITY_THRESHOLD.

    Returns (is_stable, max_change_pct).
    """
    if not prev_densities:
        return False, 1.0

    max_change = 0.0
    for sid, curr in curr_densities.items():
        prev = prev_densities.get(sid, 0.0)
        if prev > 1.0:  # only check species with meaningful density
            change = abs(curr - prev) / prev
            max_change = max(max_change, change)

    return max_change < STABILITY_THRESHOLD, max_change


# ---------------------------------------------------------------------------
# Plot strips
# ---------------------------------------------------------------------------

def plot_strips(
    strips: dict,
    output_path: str | Path,
    title: str = "Species Distribution Strips",
    compare_strips: dict | None = None,
    compare_label: str = "After",
) -> None:
    """Plot N-S and E-W strip samples. Optionally overlay a comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    positions_km = [p / 1000.0 for p in strips["positions_m"]]

    # Collect all species
    all_species = set(strips["ns_strip"].keys())
    if compare_strips:
        all_species |= set(compare_strips["ns_strip"].keys())
    sorted_species = sorted(all_species)
    n_sp = len(sorted_species)
    colors = {
        sid: hsv_to_rgb((i / max(n_sp, 1), 0.7, 0.85))
        for i, sid in enumerate(sorted_species)
    }

    n_panels = 4 if compare_strips else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)

    def plot_strip_panel(ax, strip_data, label, direction):
        for sid in sorted_species:
            if sid in strip_data:
                vals = strip_data[sid]
                ax.fill_between(positions_km[:len(vals)], vals, alpha=0.25, color=colors[sid])
                ax.plot(positions_km[:len(vals)], vals, linewidth=0.7, color=colors[sid], label=sid)
        ax.set_ylabel(f"Density\n{direction} {label}")

    plot_strip_panel(axes[0], strips["ns_strip"], "Before" if compare_strips else "", "N-S")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=5, ncol=min(5, n_sp))

    plot_strip_panel(axes[1], strips["ew_strip"], "Before" if compare_strips else "", "E-W")

    if compare_strips:
        plot_strip_panel(axes[2], compare_strips["ns_strip"], compare_label, "N-S")
        plot_strip_panel(axes[3], compare_strips["ew_strip"], compare_label, "E-W")

    axes[-1].set_xlabel("Position (km)")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history: list[dict], output_path: str | Path) -> None:
    """Plot total density per species over time to show convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    if not history:
        return

    # Collect all species across all epochs
    all_sids = set()
    for entry in history:
        all_sids.update(entry["densities"].keys())
    sorted_sids = sorted(all_sids)
    n_sp = len(sorted_sids)
    colors = {
        sid: hsv_to_rgb((i / max(n_sp, 1), 0.7, 0.85))
        for i, sid in enumerate(sorted_sids)
    }

    years = [e["year"] for e in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in sorted_sids:
        vals = [e["densities"].get(sid, 0.0) for e in history]
        label = sid.replace("anc_", "").replace("_", " ")
        ax.plot(years, vals, linewidth=1.2, color=colors[sid], label=label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Density")
    ax.set_title("Species Density Convergence")
    ax.legend(loc="upper right", fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_equilibrium(args):
    """Create a world and run to equilibrium with static weather."""
    world_dir = Path(args.world_dir)

    # Create fresh world
    print(f"Creating world at {world_dir} (seed={args.seed})...")
    from bike_sim.__main__ import _cmd_create
    create_args = argparse.Namespace(world_dir=str(world_dir), seed=args.seed)
    _cmd_create(create_args)

    world = World.open(world_dir)
    orch = Orchestrator(world)

    # Generate static weather (one-time, reused every tick)
    heightmap = world.rasters.read_layer("geology", "heightmap")

    # Check for spatial climate fields
    ch_layers = world.rasters.list_layers("climate_hydrology")
    moisture_bias = None
    continentality = None
    if "moisture_bias" in ch_layers:
        moisture_bias = world.rasters.read_layer("climate_hydrology", "moisture_bias")
    if "continentality" in ch_layers:
        continentality = world.rasters.read_layer("climate_hydrology", "continentality")

    weather_sys = WeatherSystem(
        world.seed, heightmap,
        moisture_bias=moisture_bias,
        continentality=continentality,
    )
    # Generate summer weather as our static baseline (representative growing conditions)
    static_weather = weather_sys.generate(year=25.0, season=2)

    print(f"Running to equilibrium (static weather, max {MAX_EQUILIBRIUM_YEARS}yr)...")
    print(f"  Stability: <{STABILITY_THRESHOLD*100:.0f}% change for {STABILITY_EPOCHS} consecutive {EPOCH_YEARS}yr epochs")

    eco = EcologyTier(world)
    # Set version so raster writes work in versioned mode
    next_version = world.current_version + 1
    world.rasters.set_version(next_version)

    prev_densities: dict[str, float] = {}
    stable_count = 0
    history: list[dict] = []

    start_time = time.time()

    for epoch in range(MAX_EQUILIBRIUM_YEARS // EPOCH_YEARS):
        epoch_start = time.time()

        for _ in range(EPOCH_TICKS):
            # Use static weather but vary season for fire/blowdown triggers
            tick = world.tier_clocks["ecology"].tick_number
            season = tick % 4
            sw = weather_sys.generate(year=tick * 0.25, season=season)
            # Override temperature and precipitation with static values
            sw.temperature = static_weather.temperature.copy()
            sw.precipitation = static_weather.precipitation.copy()
            eco.tick(sw)

        # Snapshot version for this epoch
        next_version += 1
        world.rasters.set_version(next_version)
        world.save(world_dir / "world.json")

        # Measure
        current_year = world.tier_clocks["ecology"].simulated_year
        species_summary = compute_species_summary(world)
        curr_densities = {s["species_id"]: s["total_density"] for s in species_summary}

        is_stable, max_change = check_stability(prev_densities, curr_densities)
        elapsed = time.time() - epoch_start

        history.append({
            "year": round(current_year, 1),
            "densities": curr_densities,
            "max_change": round(max_change, 4),
            "stable": is_stable,
        })

        alive = len([s for s in species_summary if s["total_density"] > 1.0])
        total_biomass = sum(s["total_density"] for s in species_summary)

        print(f"  Year {current_year:6.0f} | {alive} species | "
              f"biomass {total_biomass:8.0f} | "
              f"max Δ {max_change*100:5.1f}% | "
              f"{'STABLE' if is_stable else 'settling'} | "
              f"{elapsed:.0f}s")

        if is_stable:
            stable_count += 1
            if stable_count >= STABILITY_EPOCHS:
                print(f"\n  EQUILIBRIUM reached at year {current_year:.0f}")
                break
        else:
            stable_count = 0

        prev_densities = curr_densities

    else:
        print(f"\n  WARNING: Did not reach equilibrium in {MAX_EQUILIBRIUM_YEARS}yr")

    total_time = time.time() - start_time
    eq_year = world.tier_clocks["ecology"].simulated_year

    # Save equilibrium state
    world.save(world_dir / "world.json")

    # Sample strips
    strips = sample_strips(world)
    species_summary = compute_species_summary(world)

    # Save results
    results_dir = world_dir / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eq_result = {
        "type": "equilibrium",
        "seed": args.seed,
        "year": eq_year,
        "runtime_seconds": round(total_time, 1),
        "species": species_summary,
        "history": history,
    }
    (results_dir / "equilibrium.json").write_text(json.dumps(eq_result, indent=2))
    json.dump(strips, open(results_dir / "equilibrium_strips.json", "w"), indent=2)

    # Plot
    plot_strips(strips, results_dir / "equilibrium_strips.png",
                title=f"Equilibrium Strips (year {eq_year:.0f})")
    plot_convergence(history, results_dir / "convergence.png")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  EQUILIBRIUM SUMMARY (year {eq_year:.0f}, {total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"  {'Species':<35s} {'Density':>8s} {'Cells':>7s} "
          f"{'Centroid':>12s} {'Breakeven':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*12} {'-'*10}")
    for sp in species_summary:
        name = sp["species_id"].replace("anc_", "")[:33]
        print(f"  {name:<35s} {sp['total_density']:>8.0f} {sp['occupied_cells']:>7d} "
              f"  ({sp['centroid_r']:>4.0f},{sp['centroid_c']:>4.0f}) "
              f"  {sp['breakeven_suit']:>8.4f}")

    print(f"\n  Results: {results_dir}/")
    world.close()


def cmd_perturb(args):
    """Apply a perturbation to the equilibrium world and run to new equilibrium."""
    world_dir = Path(args.world_dir)
    results_dir = world_dir / "test_results"

    # Load equilibrium strips for comparison
    eq_strips_path = results_dir / "equilibrium_strips.json"
    if not eq_strips_path.exists():
        print("ERROR: No equilibrium data. Run 'equilibrium' first.")
        return
    eq_strips = json.loads(eq_strips_path.read_text())
    eq_result = json.loads((results_dir / "equilibrium.json").read_text())
    eq_summary = {s["species_id"]: s for s in eq_result["species"]}

    world = World.open(world_dir)

    # Build perturbation description
    perturb_name = ""
    perturb_desc = {}

    if args.temperature is not None:
        perturb_name = f"temp{args.temperature:+.0f}"
        perturb_desc["temperature_offset"] = args.temperature
        print(f"Perturbation: temperature {args.temperature:+.1f}°C")

    if args.precipitation is not None:
        perturb_name = f"precip{args.precipitation:+.0f}"
        perturb_desc["precipitation_offset"] = args.precipitation
        print(f"Perturbation: precipitation {args.precipitation:+.0f}mm")

    if args.fire:
        parts = args.fire.split(",")
        fire_r, fire_c = int(parts[0]), int(parts[1])
        radius = args.radius or 50
        perturb_name = f"fire_{fire_r}_{fire_c}_r{radius}"
        perturb_desc["fire"] = {"row": fire_r, "col": fire_c, "radius": radius}
        print(f"Perturbation: fire at ({fire_r},{fire_c}) radius {radius}")

    if args.remove_species:
        perturb_name = f"remove_{args.remove_species}"
        perturb_desc["remove_species"] = args.remove_species
        print(f"Perturbation: remove {args.remove_species}")

    if not perturb_name:
        print("ERROR: No perturbation specified.")
        return

    # Apply perturbation
    heightmap = world.rasters.read_layer("geology", "heightmap")
    ch_layers = world.rasters.list_layers("climate_hydrology")
    moisture_bias = None
    continentality = None
    if "moisture_bias" in ch_layers:
        moisture_bias = world.rasters.read_layer("climate_hydrology", "moisture_bias")
    if "continentality" in ch_layers:
        continentality = world.rasters.read_layer("climate_hydrology", "continentality")

    weather_sys = WeatherSystem(
        world.seed, heightmap,
        moisture_bias=moisture_bias,
        continentality=continentality,
    )
    static_weather = weather_sys.generate(year=25.0, season=2)

    # Apply climate perturbations to the static weather
    if args.temperature is not None:
        static_weather.temperature = static_weather.temperature + args.temperature
    if args.precipitation is not None:
        static_weather.precipitation = np.clip(
            static_weather.precipitation + args.precipitation, 0, None
        )

    # Apply fire perturbation
    if args.fire:
        eco = EcologyTier(world)
        from bike_sim.rng import create_rng
        rng = create_rng(world.seed, "test", "fire_perturb", 0)

        fire_r, fire_c = perturb_desc["fire"]["row"], perturb_desc["fire"]["col"]
        radius = perturb_desc["fire"]["radius"]

        # Burn a circular area
        y_coords, x_coords = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        dist = np.sqrt((y_coords - fire_r)**2 + (x_coords - fire_c)**2)
        burned = dist <= radius

        current_year = world.tier_clocks["ecology"].simulated_year
        species_list = world.events.list_species(alive_at_year=current_year)
        for sp in species_list:
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in world.rasters.list_layers("ecology"):
                density = world.rasters.read_layer("ecology", layer).copy()
                density[burned] *= 0.05  # 95% kill in burned area
                tick = world.tier_clocks["ecology"].tick_number
                world.rasters.write_layer("ecology", layer, density, tick)

        print(f"  Burned {int(burned.sum())} cells")

    # Remove species
    if args.remove_species:
        sid = args.remove_species
        layer = f"species_{sid}_density"
        if layer in world.rasters.list_layers("ecology"):
            tick = world.tier_clocks["ecology"].tick_number
            world.rasters.write_layer(
                "ecology", layer,
                np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
                tick,
            )
            current_year = world.tier_clocks["ecology"].simulated_year
            world.events.mark_species_extinct(sid, current_year)
            print(f"  Removed {sid}")

    # Run to new equilibrium
    eco = EcologyTier(world)
    next_version = world.current_version + 1
    world.rasters.set_version(next_version)
    years = args.years or 500
    n_epochs = years // EPOCH_YEARS

    print(f"\nRunning {years}yr post-perturbation...")
    prev_densities: dict[str, float] = {}
    stable_count = 0
    history: list[dict] = []
    eq_reached = False
    eq_year = None

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        for _ in range(EPOCH_TICKS):
            tick = world.tier_clocks["ecology"].tick_number
            season = tick % 4
            sw = weather_sys.generate(year=tick * 0.25, season=season)
            sw.temperature = static_weather.temperature.copy()
            sw.precipitation = static_weather.precipitation.copy()
            eco.tick(sw)

        current_year = world.tier_clocks["ecology"].simulated_year
        species_summary = compute_species_summary(world)
        curr_densities = {s["species_id"]: s["total_density"] for s in species_summary}

        is_stable, max_change = check_stability(prev_densities, curr_densities)
        elapsed = time.time() - epoch_start

        epoch_years = (epoch + 1) * EPOCH_YEARS
        history.append({
            "year": round(current_year, 1),
            "epoch_years_post_perturbation": epoch_years,
            "densities": curr_densities,
            "max_change": round(max_change, 4),
            "stable": is_stable,
        })

        alive = len([s for s in species_summary if s["total_density"] > 1.0])
        total_biomass = sum(s["total_density"] for s in species_summary)

        print(f"  +{epoch_years:4d}yr | {alive} species | "
              f"biomass {total_biomass:8.0f} | "
              f"max Δ {max_change*100:5.1f}% | "
              f"{'STABLE' if is_stable else 'settling'} | "
              f"{elapsed:.0f}s")

        if epoch_years >= MIN_PERTURBATION_YEARS and is_stable:
            stable_count += 1
            if stable_count >= STABILITY_EPOCHS and not eq_reached:
                eq_reached = True
                eq_year = epoch_years
                print(f"\n  NEW EQUILIBRIUM at +{eq_year}yr")
        else:
            stable_count = 0

        prev_densities = curr_densities

    total_time = time.time() - start_time

    # Save state
    world.save(world_dir / "world.json")

    # Sample strips
    post_strips = sample_strips(world)
    post_summary = compute_species_summary(world)

    # Compute shifts
    print(f"\n{'='*70}")
    print(f"  PERTURBATION RESULTS: {perturb_name}")
    if eq_reached:
        print(f"  New equilibrium: +{eq_year}yr")
    else:
        print(f"  Did not reach equilibrium in {years}yr")
    print(f"{'='*70}")

    print(f"  {'Species':<30s} {'Before':>8s} {'After':>8s} {'Δ%':>7s} "
          f"{'Cells Δ':>8s} {'Centroid Shift':>14s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*14}")

    for sp in post_summary:
        sid = sp["species_id"]
        name = sid.replace("anc_", "")[:28]
        before = eq_summary.get(sid, {})
        before_dens = before.get("total_density", 0)
        after_dens = sp["total_density"]
        pct = ((after_dens - before_dens) / before_dens * 100) if before_dens > 1 else 0
        cells_before = before.get("occupied_cells", 0)
        cells_delta = sp["occupied_cells"] - cells_before

        cr_before = before.get("centroid_r", 0)
        cc_before = before.get("centroid_c", 0)
        shift_km = np.sqrt((sp["centroid_r"] - cr_before)**2 +
                          (sp["centroid_c"] - cc_before)**2) * CELL_SIZE / 1000

        print(f"  {name:<30s} {before_dens:>8.0f} {after_dens:>8.0f} {pct:>+6.1f}% "
              f"{cells_delta:>+8d} {shift_km:>11.1f}km")

    # Save results
    perturb_result = {
        "type": "perturbation",
        "name": perturb_name,
        "perturbation": perturb_desc,
        "years_run": years,
        "equilibrium_reached": eq_reached,
        "time_to_equilibrium_years": eq_year,
        "runtime_seconds": round(total_time, 1),
        "species_before": eq_result["species"],
        "species_after": post_summary,
        "history": history,
    }

    perturb_dir = results_dir / perturb_name
    perturb_dir.mkdir(parents=True, exist_ok=True)
    (perturb_dir / "result.json").write_text(json.dumps(perturb_result, indent=2))
    json.dump(post_strips, open(perturb_dir / "strips.json", "w"), indent=2)

    # Plot comparison
    plot_strips(
        eq_strips, perturb_dir / "comparison.png",
        title=f"Perturbation: {perturb_name}",
        compare_strips=post_strips,
        compare_label=f"After (+{years}yr)",
    )
    plot_convergence(history, perturb_dir / "convergence.png")

    print(f"\n  Results: {perturb_dir}/")
    world.close()


def cmd_report(args):
    """Report on all experiments in a world directory."""
    world_dir = Path(args.world_dir)
    results_dir = world_dir / "test_results"

    if not results_dir.exists():
        print("No test results found.")
        return

    # Equilibrium
    eq_path = results_dir / "equilibrium.json"
    if eq_path.exists():
        eq = json.loads(eq_path.read_text())
        print(f"EQUILIBRIUM (seed {eq['seed']}, year {eq['year']:.0f}, {eq['runtime_seconds']:.0f}s)")
        print(f"  Species: {len(eq['species'])}")
        for sp in eq["species"][:5]:
            name = sp["species_id"].replace("anc_", "")
            print(f"    {name}: {sp['total_density']:.0f} density, {sp['occupied_cells']} cells")
        if len(eq["species"]) > 5:
            print(f"    ... and {len(eq['species']) - 5} more")

    # Perturbations
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        result_path = subdir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        eq_str = (f"reached at +{result['time_to_equilibrium_years']}yr"
                  if result["equilibrium_reached"] else "not reached")
        print(f"\nPERTURBATION: {result['name']}")
        print(f"  Equilibrium: {eq_str}")
        print(f"  Runtime: {result['runtime_seconds']:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ecology testing framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # equilibrium
    p_eq = subparsers.add_parser("equilibrium", help="Run to equilibrium with static weather")
    p_eq.add_argument("world_dir", help="Path for the test world")
    p_eq.add_argument("--seed", type=int, default=42, help="World seed")

    # perturb
    p_pt = subparsers.add_parser("perturb", help="Apply perturbation and run")
    p_pt.add_argument("world_dir", help="Path to equilibrium world")
    p_pt.add_argument("--temperature", type=float, default=None, help="Temperature offset (°C)")
    p_pt.add_argument("--precipitation", type=float, default=None, help="Precipitation offset (mm)")
    p_pt.add_argument("--fire", type=str, default=None, help="Fire at row,col")
    p_pt.add_argument("--radius", type=int, default=50, help="Fire radius in cells")
    p_pt.add_argument("--remove-species", type=str, default=None, help="Species to remove")
    p_pt.add_argument("--years", type=int, default=500, help="Years to run")

    # report
    p_rp = subparsers.add_parser("report", help="Report on all experiments")
    p_rp.add_argument("world_dir", help="Path to test world")

    args = parser.parse_args()

    if args.command == "equilibrium":
        cmd_equilibrium(args)
    elif args.command == "perturb":
        cmd_perturb(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()

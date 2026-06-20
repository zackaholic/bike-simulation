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

    # Run a declarative scenario (never mutates the baseline)
    python scripts/test_ecology.py scenario worlds/eq_test scenarios/fire_large.yaml
    python scripts/test_ecology.py batch worlds/eq_test scenarios/

Scenarios are declarative YAML specs (sustained weather mods + one-time shocks).
Each scenario runs on a fresh copy of the baseline world; results land under
``results/<runid>/<scenario_name>/``.  See the ``Scenario`` dataclass and the
shock registry below for the schema.

Equilibrium test: creates a world, runs with static weather until species
populations stabilize (max per-species density change < 2% for 2 consecutive
100yr chunks). Records strip samples (N-S and E-W through center).

Perturbation test: copies the equilibrium snapshot, applies a perturbation,
runs until new equilibrium or max years. Records time-to-equilibrium and
strip samples showing how distributions shifted.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import yaml

from bike_sim.world import World
from bike_sim.orchestrator import Orchestrator
from bike_sim.weather import WeatherSystem
from bike_sim.rng import create_rng
from bike_sim.tiers.ecology import (
    EcologyTier,
    _ANCESTOR_TEMPLATES,
    COMPETITION_BASELINE,
    PRECIP_REF_MIN,
    PRECIP_REF_MAX,
    TEMP_REF_MIN,
    TEMP_REF_MAX,
)

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

# Extinction and speciation are deferred; we test the fixed 14-species substrate.
# The refugium floor keeps marginal species alive at trace density so they can
# rebound under a favorable perturbation (the test we actually want to run).
REFUGIUM_FLOOR = 1.0


def _configure_eco(eco):
    """Configure an EcologyTier for testing: no extinction/speciation/individuals, refugium on."""
    eco.enable_extinction = False
    eco.enable_speciation = False
    eco.enable_individuals = False
    eco.refugium_floor = REFUGIUM_FLOOR
    return eco


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

# Species below this fraction of total biomass are ignored by the stability
# check — a tiny species swinging 200% on negligible absolute density is noise,
# not a sign the world is still settling.
STABILITY_MIN_BIOMASS_FRACTION = 0.001  # 0.1% of total biomass


def check_stability(
    prev_densities: dict[str, float], curr_densities: dict[str, float]
) -> tuple[bool, float, str | None]:
    """Check if all meaningful species' densities changed less than the threshold.

    Only species whose current density exceeds STABILITY_MIN_BIOMASS_FRACTION of
    total biomass are considered, so negligible species reshuffling doesn't mask
    aggregate convergence.

    Returns (is_stable, max_change_pct, biggest_mover_id).
    """
    if not prev_densities:
        return False, 1.0, None

    total_biomass = sum(curr_densities.values())
    if total_biomass <= 0:
        return False, 1.0, None
    min_density = total_biomass * STABILITY_MIN_BIOMASS_FRACTION

    max_change = 0.0
    biggest_mover: str | None = None
    for sid, curr in curr_densities.items():
        prev = prev_densities.get(sid, 0.0)
        # Only meaningful species (above the biomass-fraction floor) count.
        if curr < min_density and prev < min_density:
            continue
        if prev > 0.0:
            change = abs(curr - prev) / prev
            if change > max_change:
                max_change = change
                biggest_mover = sid

    return max_change < STABILITY_THRESHOLD, max_change, biggest_mover


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
# Reusable per-epoch run loop
# ---------------------------------------------------------------------------

def run_epochs(
    world,
    eco,
    weather_for_tick,
    n_epochs,
    world_dir,
    label_prefix="+",
    trigger_prefix="run",
):
    """Run ``n_epochs`` ecology epochs, checking for equilibrium each epoch.

    ``weather_for_tick(tick)`` returns the SeasonalWeather to apply at a given
    ecology tick number. This is the shared engine for both ``cmd_perturb`` and
    the declarative scenario runner — the only thing that varies between callers
    is how weather is constructed (frozen seasons vs. a live climate cycle).

    Returns ``(history, eq_reached, eq_year, total_time)`` where ``eq_year`` is
    the post-start year at which equilibrium was first reached (or None).
    """
    prev_densities: dict[str, float] = {}
    stable_count = 0
    history: list[dict] = []
    eq_reached = False
    eq_year = None

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Fresh version per epoch; set_version + commit_version paired so the
        # manifest stays in sync with raster versions (see equilibrium note).
        world.rasters.set_version(world.current_version + 1)

        for _ in range(EPOCH_TICKS):
            tick = world.tier_clocks["ecology"].tick_number
            eco.tick(weather_for_tick(tick))

        world.commit_version(trigger=f"{trigger_prefix} epoch {epoch}")
        world.save(world_dir / "world.json")

        current_year = world.tier_clocks["ecology"].simulated_year
        species_summary = compute_species_summary(world)
        curr_densities = {s["species_id"]: s["total_density"] for s in species_summary}

        is_stable, max_change, mover = check_stability(prev_densities, curr_densities)
        elapsed = time.time() - epoch_start

        epoch_years = (epoch + 1) * EPOCH_YEARS
        history.append({
            "year": round(current_year, 1),
            "epoch_years_post_perturbation": epoch_years,
            "densities": curr_densities,
            "max_change": round(max_change, 4),
            "stable": is_stable,
            "biggest_mover": mover,
        })

        alive = len([s for s in species_summary if s["total_density"] > 1.0])
        total_biomass = sum(s["total_density"] for s in species_summary)
        mover_str = mover.replace("anc_", "")[:18] if mover else "-"

        print(f"  {label_prefix}{epoch_years:4d}yr | {alive} species | "
              f"biomass {total_biomass:10.0f} | "
              f"max Δ {max_change*100:5.1f}% ({mover_str}) | "
              f"{'STABLE' if is_stable else 'settling'} | "
              f"{elapsed:.0f}s")

        if epoch_years >= MIN_PERTURBATION_YEARS and is_stable:
            stable_count += 1
            if stable_count >= STABILITY_EPOCHS and not eq_reached:
                eq_reached = True
                eq_year = epoch_years
                print(f"\n  NEW EQUILIBRIUM at {label_prefix}{eq_year}yr")
        else:
            stable_count = 0

        prev_densities = curr_densities

    total_time = time.time() - start_time
    return history, eq_reached, eq_year, total_time


# ---------------------------------------------------------------------------
# Scenario schema + declarative loader
# ---------------------------------------------------------------------------

VALID_WEATHER_MODES = ("frozen", "cycling")


@dataclass
class Scenario:
    """A declarative perturbation scenario loaded from YAML.

    - sustained: list of {field, op, value} weather modifications applied to
      EVERY tick's weather (e.g. a permanent +5°C shift).
    - shocks: list of {type, ...params} one-time disturbances applied to the
      density rasters at t=0 (dispatched through the shock registry).
    """

    name: str
    description: str = ""
    weather_mode: str = "frozen"
    sustained: list = field(default_factory=list)
    shocks: list = field(default_factory=list)
    run_years: int = 500

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "weather_mode": self.weather_mode,
            "sustained": self.sustained,
            "shocks": self.shocks,
            "run_years": self.run_years,
        }


def load_scenario(path) -> Scenario:
    """Parse and validate a scenario YAML file into a Scenario."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario {path}: top-level YAML must be a mapping")
    if not raw.get("name"):
        raise ValueError(f"Scenario {path}: missing required field 'name'")

    mode = raw.get("weather_mode", "frozen")
    if mode not in VALID_WEATHER_MODES:
        raise ValueError(
            f"Scenario {path}: unknown weather_mode {mode!r} "
            f"(expected one of {VALID_WEATHER_MODES})"
        )

    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        weather_mode=mode,
        sustained=raw.get("sustained", []) or [],
        shocks=raw.get("shocks", []) or [],
        run_years=int(raw.get("run_years", 500)),
    )


# ---------------------------------------------------------------------------
# Sustained weather modification
# ---------------------------------------------------------------------------

def apply_sustained_mods(sw, mods):
    """Apply sustained weather modifications to a SeasonalWeather in place.

    Each mod is {field: 'temperature'|'precipitation', op: 'add'|'multiply'|'set',
    value: number}. Precipitation is clipped to >= 0. Returns ``sw``.
    """
    for mod in mods:
        field_name = mod["field"]
        op = mod["op"]
        value = mod["value"]

        cur = getattr(sw, field_name)
        if op == "add":
            new = cur + value
        elif op == "multiply":
            new = cur * value
        elif op == "set":
            new = np.full_like(cur, value, dtype=np.float64)
        else:
            raise ValueError(f"Unknown sustained op {op!r}")

        if field_name == "precipitation":
            new = np.clip(new, 0, None)
        setattr(sw, field_name, new)
    return sw


def build_weather_for_tick(world, scenario):
    """Build a ``weather_for_tick(tick)`` callable per the scenario's mode.

    - frozen: 4 seasonal snapshots at year 25.0 (matching the equilibrium run),
      sustained mods baked in once; tick -> static_seasons[tick % 4].
    - cycling: regenerate weather live each tick from the real climate cycle
      (slow drift included), applying sustained mods to a fresh copy each call.
    """
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

    if scenario.weather_mode == "frozen":
        static_seasons = [weather_sys.generate(year=25.0, season=s) for s in range(4)]
        for sw in static_seasons:
            apply_sustained_mods(sw, scenario.sustained)
        return lambda tick: static_seasons[tick % 4]

    # cycling: regenerate live each tick (slow climate drift is preserved), then
    # apply sustained mods to that fresh copy so the modifications ride on top of
    # the real climate signal.
    def weather_for_tick(tick):
        sw = weather_sys.generate(year=tick * 0.25, season=tick % 4)
        return apply_sustained_mods(sw, scenario.sustained)

    return weather_for_tick


# ---------------------------------------------------------------------------
# Shock registry
# ---------------------------------------------------------------------------

# Maps a shock 'type' string to a handler fn(world, densities, rng, params) that
# mutates the densities dict {species_id: ndarray} in place. Add a new shock by
# writing a new @register_shock("name") function — no other wiring needed.
SHOCK_REGISTRY = {}


def register_shock(name):
    """Decorator: register a shock handler under ``name``."""
    def deco(fn):
        SHOCK_REGISTRY[name] = fn
        return fn
    return deco


def apply_shocks(world, densities, shocks, rng):
    """Apply each shock (in order) to the densities dict, in place."""
    for shock in shocks:
        stype = shock.get("type")
        if stype not in SHOCK_REGISTRY:
            raise ValueError(
                f"Unknown shock type {stype!r} "
                f"(registered: {sorted(SHOCK_REGISTRY)})"
            )
        SHOCK_REGISTRY[stype](world, densities, rng, shock)


@register_shock("fire")
def _shock_fire(world, densities, rng, params):
    """Circular burn: multiply all species density by (1-kill) inside a disk."""
    fire_r, fire_c = params["location"]
    radius = params["radius"]
    kill = params["kill"]

    y_coords, x_coords = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    dist = np.sqrt((y_coords - fire_r) ** 2 + (x_coords - fire_c) ** 2)
    burned = dist <= radius

    for sid in densities:
        densities[sid][burned] *= (1.0 - kill)


@register_shock("flood")
def _shock_flood(world, densities, rng, params):
    """Drown low ground: kill all species below an elevation percentile."""
    elevation_percentile = params["elevation_percentile"]
    kill = params["kill"]

    heightmap = world.rasters.read_layer("geology", "heightmap")
    threshold = np.percentile(heightmap, elevation_percentile)
    flooded = heightmap < threshold

    for sid in densities:
        densities[sid][flooded] *= (1.0 - kill)


@register_shock("remove_species")
def _shock_remove_species(world, densities, rng, params):
    """Zero out a single species' density field entirely."""
    sid = params["species"]
    if sid in densities:
        densities[sid][:] = 0.0


# Comparison ops for trait filters (pestilence and friends).
_FILTER_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


@register_shock("pestilence")
def _shock_pestilence(world, densities, rng, params):
    """Trait-targeted die-off: kill species whose genome matches a filter.

    filter = {trait, op, value}; e.g. {trait: max_height, op: '>', value: 10}
    targets tall species (a canopy disease). Ops: >, <, >=, <=.
    """
    filt = params["filter"]
    kill = params["kill"]
    trait = filt["trait"]
    op = filt["op"]
    value = filt["value"]
    if op not in _FILTER_OPS:
        raise ValueError(f"Unknown filter op {op!r} (expected one of {sorted(_FILTER_OPS)})")
    cmp = _FILTER_OPS[op]

    for sid in densities:
        genome = world.events.get_species(sid)["genome"]
        if trait in genome and cmp(genome[trait], value):
            densities[sid] *= (1.0 - kill)


def write_densities(world, densities):
    """Write a densities dict back to a fresh raster version (paired set/commit).

    Uses the same set_version + commit_version pattern the run loop uses so the
    manifest and raster versions advance together (the versioning bug guard).
    """
    world.rasters.set_version(world.current_version + 1)
    tick = world.tier_clocks["ecology"].tick_number
    for sid, arr in densities.items():
        world.rasters.write_layer("ecology", f"species_{sid}_density", arr, tick)
    world.commit_version(trigger="scenario shocks (t=0)")


def load_all_densities(world):
    """Load every alive species' density field into a {species_id: ndarray} dict."""
    current_year = world.tier_clocks["ecology"].simulated_year
    species_list = world.events.list_species(alive_at_year=current_year)
    eco_layers = world.rasters.list_layers("ecology")
    densities = {}
    for sp in species_list:
        sid = sp["species_id"]
        layer = f"species_{sid}_density"
        if layer in eco_layers:
            densities[sid] = world.rasters.read_layer("ecology", layer).copy()
    return densities


# ---------------------------------------------------------------------------
# Baseline copy-on-test + staleness hashing
# ---------------------------------------------------------------------------

def copy_baseline(baseline_dir, scratch_dir):
    """Copy the baseline world to a scratch dir (removing scratch first).

    The baseline is NEVER mutated — each scenario runs on a fresh copy.
    """
    baseline_dir = Path(baseline_dir)
    scratch_dir = Path(scratch_dir)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    shutil.copytree(baseline_dir, scratch_dir)
    return scratch_dir


def compute_config_hash():
    """Short hash of the key ecology config that defines the baseline.

    If any of these change, a previously computed baseline equilibrium no longer
    reflects the current model and should be regenerated.
    """
    payload = "|".join([
        f"COMPETITION_BASELINE={COMPETITION_BASELINE}",
        f"PRECIP_REF_MIN={PRECIP_REF_MIN}",
        f"PRECIP_REF_MAX={PRECIP_REF_MAX}",
        f"TEMP_REF_MIN={TEMP_REF_MIN}",
        f"TEMP_REF_MAX={TEMP_REF_MAX}",
        f"ANCESTORS={_ANCESTOR_TEMPLATES!r}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def check_baseline_staleness(baseline_dir):
    """Compare the stored baseline config hash to the current one.

    Returns (is_stale, current_hash, stored_hash). Prints a clear warning when
    stale, but does not abort — the caller still runs (on possibly-stale data).
    """
    current_hash = compute_config_hash()
    hash_path = Path(baseline_dir) / "config_hash.txt"
    stored_hash = hash_path.read_text().strip() if hash_path.exists() else None

    is_stale = stored_hash is not None and stored_hash != current_hash
    if stored_hash is None:
        print(f"  WARNING: baseline has no config_hash.txt — cannot verify freshness "
              f"(current {current_hash}).")
    elif is_stale:
        print(f"  WARNING: baseline is STALE. Stored config hash {stored_hash} != "
              f"current {current_hash}. Regenerate the equilibrium baseline.")
    return is_stale, current_hash, stored_hash


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
    # Four fixed seasonal snapshots at a reference year. We freeze the SLOW
    # multi-year climate drift (so equilibrium is detectable) but keep the FAST
    # seasonal cycle (so each species experiences the frost/drought extremes its
    # niche is built around). Using a single season instead biases the
    # equilibrium toward that season's specialists.
    static_seasons = [weather_sys.generate(year=25.0, season=s) for s in range(4)]

    print(f"Running to equilibrium (frozen-drift seasonal cycle, max {MAX_EQUILIBRIUM_YEARS}yr)...")
    print(f"  Stability: <{STABILITY_THRESHOLD*100:.0f}% change for {STABILITY_EPOCHS} consecutive {EPOCH_YEARS}yr epochs")
    print(f"  Extinction/speciation OFF, refugium floor {REFUGIUM_FLOOR}")

    eco = _configure_eco(EcologyTier(world))

    prev_densities: dict[str, float] = {}
    stable_count = 0
    history: list[dict] = []

    start_time = time.time()

    for epoch in range(MAX_EQUILIBRIUM_YEARS // EPOCH_YEARS):
        epoch_start = time.time()

        # Direct raster writes to a fresh version for this epoch. set_version +
        # commit_version must be paired so world.current_version (the manifest)
        # advances in lockstep with the raster version groups — otherwise reads
        # (which return the latest raster version) and writes diverge.
        world.rasters.set_version(world.current_version + 1)

        for _ in range(EPOCH_TICKS):
            # Cycle the 4 fixed seasonal snapshots — same weather each year,
            # realistic within-year seasonal swing.
            tick = world.tier_clocks["ecology"].tick_number
            eco.tick(static_seasons[tick % 4])

        # Snapshot version for this epoch
        world.commit_version(trigger=f"equilibrium epoch {epoch}")
        world.save(world_dir / "world.json")

        # Measure
        current_year = world.tier_clocks["ecology"].simulated_year
        species_summary = compute_species_summary(world)
        curr_densities = {s["species_id"]: s["total_density"] for s in species_summary}

        is_stable, max_change, mover = check_stability(prev_densities, curr_densities)
        elapsed = time.time() - epoch_start

        history.append({
            "year": round(current_year, 1),
            "densities": curr_densities,
            "max_change": round(max_change, 4),
            "stable": is_stable,
            "biggest_mover": mover,
        })

        alive = len([s for s in species_summary if s["total_density"] > 1.0])
        total_biomass = sum(s["total_density"] for s in species_summary)
        mover_str = mover.replace("anc_", "")[:18] if mover else "-"

        print(f"  Year {current_year:6.0f} | {alive} species | "
              f"biomass {total_biomass:10.0f} | "
              f"max Δ {max_change*100:5.1f}% ({mover_str}) | "
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

    # Stamp the baseline with the config hash so scenario runs can detect when
    # the ecology model has drifted out from under a stale equilibrium.
    (world_dir / "config_hash.txt").write_text(compute_config_hash() + "\n")

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
    # Four fixed seasonal snapshots (same frozen-drift seasonal cycle the
    # equilibrium used), so the only change is the perturbation itself.
    static_seasons = [weather_sys.generate(year=25.0, season=s) for s in range(4)]

    # Apply climate perturbations to every season's weather
    if args.temperature is not None:
        for sw in static_seasons:
            sw.temperature = sw.temperature + args.temperature
    if args.precipitation is not None:
        for sw in static_seasons:
            sw.precipitation = np.clip(sw.precipitation + args.precipitation, 0, None)

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
    eco = _configure_eco(EcologyTier(world))
    years = args.years or 500
    n_epochs = years // EPOCH_YEARS

    print(f"\nRunning {years}yr post-perturbation...")
    history, eq_reached, eq_year, total_time = run_epochs(
        world, eco,
        weather_for_tick=lambda tick: static_seasons[tick % 4],
        n_epochs=n_epochs,
        world_dir=world_dir,
        label_prefix="+",
        trigger_prefix=f"perturb {perturb_name}",
    )

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
# Scenario / batch commands
# ---------------------------------------------------------------------------

def _compute_species_deltas(before_summary, after_summary):
    """Build per-species before/after deltas (density %, cells, centroid shift km).

    Returns a list of dicts sorted by descending density delta (winners first).
    ``before_summary`` is keyed by species_id; ``after_summary`` is a list.
    """
    deltas = []
    for sp in after_summary:
        sid = sp["species_id"]
        before = before_summary.get(sid, {})
        before_dens = before.get("total_density", 0)
        after_dens = sp["total_density"]
        pct = ((after_dens - before_dens) / before_dens * 100) if before_dens > 1 else 0.0
        cells_delta = sp["occupied_cells"] - before.get("occupied_cells", 0)
        shift_km = np.sqrt(
            (sp["centroid_r"] - before.get("centroid_r", 0)) ** 2
            + (sp["centroid_c"] - before.get("centroid_c", 0)) ** 2
        ) * CELL_SIZE / 1000
        deltas.append({
            "species_id": sid,
            "density_before": before_dens,
            "density_after": after_dens,
            "density_pct": round(pct, 1),
            "cells_delta": cells_delta,
            "centroid_shift_km": round(float(shift_km), 2),
        })
    return sorted(deltas, key=lambda d: -(d["density_after"] - d["density_before"]))


def run_scenario(scenario, baseline_dir, results_root, keep=False):
    """Run one scenario on a fresh copy of the baseline; write results.

    Returns the result dict (also written to result.json). The baseline world is
    never mutated; a scratch copy is made, run, and deleted unless ``keep``.
    """
    baseline_dir = Path(baseline_dir)
    results_root = Path(results_root)

    # Baseline freshness check (warns but does not abort).
    is_stale, current_hash, stored_hash = check_baseline_staleness(baseline_dir)

    # Load the baseline equilibrium strips + summary for before/after comparison.
    eq_strips_path = baseline_dir / "test_results" / "equilibrium_strips.json"
    eq_result_path = baseline_dir / "test_results" / "equilibrium.json"
    if not eq_strips_path.exists() or not eq_result_path.exists():
        raise FileNotFoundError(
            f"Baseline {baseline_dir} has no equilibrium data — run 'equilibrium' first."
        )
    before_strips = json.loads(eq_strips_path.read_text())
    eq_result = json.loads(eq_result_path.read_text())
    before_summary = {s["species_id"]: s for s in eq_result["species"]}

    # Results land in results/<runid>/<scenario_name>/.
    out_dir = results_root / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy baseline -> scratch (baseline is never mutated).
    scratch_dir = out_dir / "_scratch_world"
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario.name}  ({scenario.weather_mode} weather, "
          f"{scenario.run_years}yr)")
    print(f"  {scenario.description}")
    print(f"{'='*70}")
    print(f"  Copying baseline -> {scratch_dir}")
    copy_baseline(baseline_dir, scratch_dir)

    world = World.open(scratch_dir)

    # Build weather (frozen seasons or live cycle) with sustained mods baked in.
    weather_for_tick = build_weather_for_tick(world, scenario)

    # Apply one-time shocks to the density rasters at t=0.
    if scenario.shocks:
        rng = create_rng(world.seed, "test", "scenario_shocks", 0)
        densities = load_all_densities(world)
        apply_shocks(world, densities, scenario.shocks, rng)
        write_densities(world, densities)
        print(f"  Applied {len(scenario.shocks)} shock(s) at t=0")

    # Run.
    eco = _configure_eco(EcologyTier(world))
    n_epochs = scenario.run_years // EPOCH_YEARS
    print(f"\nRunning {scenario.run_years}yr...")
    history, eq_reached, eq_year, total_time = run_epochs(
        world, eco,
        weather_for_tick=weather_for_tick,
        n_epochs=n_epochs,
        world_dir=scratch_dir,
        label_prefix="+",
        trigger_prefix=f"scenario {scenario.name}",
    )

    world.save(scratch_dir / "world.json")

    # Sample after-state.
    after_strips = sample_strips(world)
    after_summary = compute_species_summary(world)
    deltas = _compute_species_deltas(before_summary, after_summary)

    # Print before/after table (same shape as cmd_perturb).
    print(f"\n  {'Species':<30s} {'Before':>8s} {'After':>8s} {'Δ%':>7s} "
          f"{'Cells Δ':>8s} {'Centroid Shift':>14s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*14}")
    for d in sorted(after_summary, key=lambda s: -s["total_density"]):
        sid = d["species_id"]
        delta = next((x for x in deltas if x["species_id"] == sid), {})
        name = sid.replace("anc_", "")[:28]
        print(f"  {name:<30s} {delta.get('density_before', 0):>8.0f} "
              f"{delta.get('density_after', 0):>8.0f} {delta.get('density_pct', 0):>+6.1f}% "
              f"{delta.get('cells_delta', 0):>+8d} {delta.get('centroid_shift_km', 0):>11.1f}km")

    world.close()

    # Write results.
    result = {
        "type": "scenario",
        "scenario": scenario.to_dict(),
        "config_hash": current_hash,
        "baseline_stale": is_stale,
        "baseline_config_hash": stored_hash,
        "run_years": scenario.run_years,
        "equilibrium_reached": eq_reached,
        "time_to_equilibrium_years": eq_year,
        "species_before": eq_result["species"],
        "species_after": after_summary,
        "species_deltas": deltas,
        "runtime_seconds": round(total_time, 1),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    json.dump(after_strips, open(out_dir / "strips.json", "w"), indent=2)
    (out_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario.to_dict(), sort_keys=False))

    plot_strips(
        before_strips, out_dir / "comparison.png",
        title=f"Scenario: {scenario.name}",
        compare_strips=after_strips,
        compare_label=f"After (+{scenario.run_years}yr)",
    )
    plot_convergence(history, out_dir / "convergence.png")

    print(f"\n  Results: {out_dir}/")

    # Clean up scratch unless asked to keep it.
    if not keep:
        shutil.rmtree(scratch_dir)
    else:
        print(f"  Scratch world kept at {scratch_dir}")

    return result


def cmd_scenario(args):
    """Run a single declarative scenario file against a baseline world."""
    scenario = load_scenario(args.scenario_file)
    runid = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root) / runid
    run_scenario(scenario, args.baseline_dir, results_root, keep=args.keep)


def _top_movers(deltas, n=3):
    """Return (winners, losers) — top/bottom n species by absolute density delta."""
    ranked = sorted(deltas, key=lambda d: (d["density_after"] - d["density_before"]))
    losers = ranked[:n]
    winners = list(reversed(ranked[-n:]))
    return winners, losers


def _fmt_movers(movers):
    """Format a list of mover deltas for a markdown cell."""
    return "; ".join(
        f"{m['species_id'].replace('anc_', '')} ({m['density_pct']:+.0f}%)"
        for m in movers
    ) or "—"


def cmd_batch(args):
    """Run every *.yaml scenario in a directory, then write an aggregate index.md."""
    scenarios_dir = Path(args.scenarios_dir)
    scenario_files = sorted(scenarios_dir.glob("*.yaml"))
    if not scenario_files:
        print(f"No *.yaml scenarios found in {scenarios_dir}")
        return

    runid = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root) / runid
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Batch: {len(scenario_files)} scenario(s) -> {results_root}/")

    rows = []
    for sf in scenario_files:
        scenario = load_scenario(sf)
        result = run_scenario(scenario, args.baseline_dir, results_root, keep=args.keep)
        winners, losers = _top_movers(result["species_deltas"])
        eq = (f"+{result['time_to_equilibrium_years']}yr"
              if result["equilibrium_reached"] else "no")
        rows.append({
            "name": scenario.name,
            "description": scenario.description,
            "eq_reached": "yes" if result["equilibrium_reached"] else "no",
            "time_to_eq": eq,
            "winners": _fmt_movers(winners),
            "losers": _fmt_movers(losers),
        })

    # Aggregate markdown index.
    lines = [
        f"# Scenario batch {runid}",
        "",
        f"Baseline: `{args.baseline_dir}`",
        "",
        "| Scenario | Description | Equilibrium | Time to Eq | Top winners | Top losers |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        link = f"[{r['name']}](./{r['name']}/)"
        lines.append(
            f"| {link} | {r['description']} | {r['eq_reached']} | {r['time_to_eq']} "
            f"| {r['winners']} | {r['losers']} |"
        )
    (results_root / "index.md").write_text("\n".join(lines) + "\n")
    print(f"\nBatch complete. Index: {results_root / 'index.md'}")


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

    # scenario
    p_sc = subparsers.add_parser("scenario", help="Run a declarative scenario YAML")
    p_sc.add_argument("baseline_dir", help="Path to equilibrium baseline world")
    p_sc.add_argument("scenario_file", help="Path to a scenario .yaml file")
    p_sc.add_argument("--results-root", default="results", help="Root dir for results")
    p_sc.add_argument("--keep", action="store_true", help="Keep the scratch world copy")

    # batch
    p_ba = subparsers.add_parser("batch", help="Run every *.yaml scenario in a dir")
    p_ba.add_argument("baseline_dir", help="Path to equilibrium baseline world")
    p_ba.add_argument("scenarios_dir", help="Directory of scenario .yaml files")
    p_ba.add_argument("--results-root", default="results", help="Root dir for results")
    p_ba.add_argument("--keep", action="store_true", help="Keep the scratch world copies")

    args = parser.parse_args()

    if args.command == "equilibrium":
        cmd_equilibrium(args)
    elif args.command == "perturb":
        cmd_perturb(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "scenario":
        cmd_scenario(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()

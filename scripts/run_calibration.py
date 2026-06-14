"""Run a calibration world in chunks with epoch diagnostics.

Usage:
    uv run python scripts/run_calibration.py worlds/calibration_v9 --years 1000
    uv run python scripts/run_calibration.py worlds/calibration_v9 --years 2000 --threshold 120

Writes a diagnostics.jsonl file alongside stdout with per-epoch analysis
of climate-ecology coupling, suitability response, density changes, and events.
"""

import argparse
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
from bike_sim.tiers.ecology import EcologyTier


def compute_epoch_diagnostics(world, prev_state, tick_start, tick_end):
    """Compute diagnostic metrics for one epoch (chunk of simulation time).

    Returns a dict with climate, suitability, density, and event analysis.
    """
    events = world.events
    store = world.rasters

    current_year = world.tier_clocks["ecology"].simulated_year

    # --- Climate summary from tick_weather ---
    weather_rows = events.get_tick_weather(tick_start=tick_start, tick_end=tick_end)
    if weather_rows:
        temps = [r["mean_temp"] for r in weather_rows]
        precips = [r["mean_precip"] for r in weather_rows]
        droughts = [r["mean_drought"] for r in weather_rows]
        climate = {
            "mean_temp": sum(temps) / len(temps),
            "temp_range": [min(temps), max(temps)],
            "mean_precip": sum(precips) / len(precips),
            "precip_range": [min(precips), max(precips)],
            "mean_drought_stress": sum(droughts) / len(droughts),
            "max_drought_stress": max(droughts),
        }
    else:
        climate = {"mean_temp": 0, "mean_precip": 0, "mean_drought_stress": 0}

    # --- Per-species density and suitability ---
    species_list = events.list_species()
    alive = [s for s in species_list if s.get("alive", True)]
    dead = [s for s in species_list if not s.get("alive", True)]
    ecology_layers = store.list_layers("ecology")

    # Generate weather at current time for suitability computation
    heightmap = store.read_layer("geology", "heightmap")
    weather_sys = WeatherSystem(world.seed, heightmap)
    season = world.tier_clocks["ecology"].tick_number % 4
    weather = weather_sys.generate(current_year, season)

    eco = EcologyTier(world)

    species_diagnostics = []
    for sp in alive:
        sid = sp["species_id"]
        genome = events.get_species(sid)["genome"]
        layer = f"species_{sid}_density"
        if layer not in ecology_layers:
            continue

        density = store.read_layer("ecology", layer)
        total_density = float(density.sum())
        occupied = int((density > 0.1).sum())
        max_density = float(density.max())

        # Compute current suitability
        suit = eco._compute_suitability_from_weather(genome, weather)
        mean_suit = float(suit.mean())
        suit_at_occupied = float(suit[density > 0.1].mean()) if occupied > 0 else 0.0

        # Range centroid (density-weighted)
        if total_density > 0:
            rows, cols = np.mgrid[0:density.shape[0], 0:density.shape[1]]
            centroid_r = float((rows * density).sum() / total_density)
            centroid_c = float((cols * density).sum() / total_density)
        else:
            centroid_r, centroid_c = 0.0, 0.0

        # Change from previous epoch
        prev = prev_state.get(sid, {})
        density_change = total_density - prev.get("total_density", total_density)
        density_change_pct = (
            density_change / prev["total_density"] * 100
            if prev.get("total_density", 0) > 1 else 0
        )
        cells_change = occupied - prev.get("occupied_cells", occupied)
        suit_change = mean_suit - prev.get("mean_suitability", mean_suit)
        centroid_shift = (
            np.sqrt(
                (centroid_r - prev.get("centroid_r", centroid_r)) ** 2
                + (centroid_c - prev.get("centroid_c", centroid_c)) ** 2
            )
            * 50.0 / 1000.0  # convert cells to km
        )

        # Ancestor lineage
        root = sid.split("_d")[0]

        sp_diag = {
            "species_id": sid,
            "ancestor": root,
            "total_density": round(total_density, 1),
            "occupied_cells": occupied,
            "max_density": round(max_density, 3),
            "mean_suitability": round(mean_suit, 4),
            "suit_at_range": round(suit_at_occupied, 4),
            "centroid": [round(centroid_r, 1), round(centroid_c, 1)],
            "density_change": round(density_change, 1),
            "density_change_pct": round(density_change_pct, 1),
            "cells_change": cells_change,
            "suit_change": round(suit_change, 4),
            "centroid_shift_km": round(centroid_shift, 2),
        }
        species_diagnostics.append(sp_diag)

    # --- Biotic pressure ---
    pressures = events.get_biotic_pressures()
    pressure_vals = list(pressures.values()) if pressures else [0.0]

    # --- Events this epoch (speciation, extinction, reabsorption) ---
    epoch_events = []
    for sp in species_list:
        appeared = sp.get("appeared_year", 0)
        prev_year = current_year - (tick_end - tick_start) * 0.25
        if sp.get("parent_id") and prev_year <= appeared <= current_year:
            epoch_events.append({
                "type": "speciation",
                "species_id": sp["species_id"],
                "parent_id": sp["parent_id"],
                "year": appeared,
            })
    for sp in dead:
        extinct_yr = sp.get("extinct_year")
        if extinct_yr and prev_year <= extinct_yr <= current_year:
            epoch_events.append({
                "type": "extinction",
                "species_id": sp["species_id"],
                "year": extinct_yr,
            })

    # --- Build current state for next epoch's comparison ---
    current_state = {}
    for sp_diag in species_diagnostics:
        sid = sp_diag["species_id"]
        current_state[sid] = {
            "total_density": sp_diag["total_density"],
            "occupied_cells": sp_diag["occupied_cells"],
            "mean_suitability": sp_diag["mean_suitability"],
            "centroid_r": sp_diag["centroid"][0],
            "centroid_c": sp_diag["centroid"][1],
        }

    # --- Aggregate metrics ---
    total_biomass = sum(s["total_density"] for s in species_diagnostics)
    density_movers = sorted(
        species_diagnostics,
        key=lambda s: abs(s["density_change_pct"]),
        reverse=True,
    )

    return {
        "year": round(current_year, 1),
        "tick_range": [tick_start, tick_end],
        "climate": climate,
        "species_count": {"alive": len(alive), "extinct": len(dead), "total": len(species_list)},
        "total_biomass": round(total_biomass, 1),
        "pressure": {
            "mean": round(sum(pressure_vals) / len(pressure_vals), 4),
            "max": round(max(pressure_vals), 4),
        },
        "species": species_diagnostics,
        "events": epoch_events,
        "top_movers": [
            {"id": s["species_id"], "change_pct": s["density_change_pct"]}
            for s in density_movers[:3]
            if abs(s["density_change_pct"]) > 0.5
        ],
    }, current_state


def print_epoch_summary(diag):
    """Print a human-readable summary of epoch diagnostics."""
    d = diag
    print(f"\n{'=' * 70}")
    print(f"  YEAR {d['year']:.0f}  |  {d['species_count']['alive']} alive, "
          f"{d['species_count']['extinct']} extinct  |  "
          f"biomass: {d['total_biomass']:.0f}  |  "
          f"pressure: {d['pressure']['mean']:.4f}/{d['pressure']['max']:.4f}")
    print(f"{'=' * 70}")

    c = d["climate"]
    print(f"  Climate: temp={c['mean_temp']:.1f}°C "
          f"[{c['temp_range'][0]:.1f}–{c['temp_range'][1]:.1f}], "
          f"precip={c['mean_precip']:.0f}mm, "
          f"drought={c['mean_drought_stress']:.4f} (max {c['max_drought_stress']:.4f})")

    # Species table
    print(f"\n  {'Species':<40s} {'Density':>8s} {'Δ%':>7s} {'Cells':>6s} {'ΔCells':>7s} "
          f"{'Suit':>6s} {'ΔSuit':>7s} {'Shift':>6s}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 7} "
          f"{'-' * 6} {'-' * 7} {'-' * 6}")
    for sp in sorted(d["species"], key=lambda s: -s["total_density"]):
        name = sp["species_id"]
        if len(name) > 38:
            name = name[:35] + "..."
        print(f"  {name:<40s} {sp['total_density']:>8.0f} {sp['density_change_pct']:>+6.1f}% "
              f"{sp['occupied_cells']:>6d} {sp['cells_change']:>+7d} "
              f"{sp['mean_suitability']:>6.3f} {sp['suit_change']:>+7.4f} "
              f"{sp['centroid_shift_km']:>5.1f}km")

    # Events
    if d["events"]:
        print(f"\n  Events:")
        for evt in d["events"]:
            if evt["type"] == "speciation":
                print(f"    ★ Speciation: {evt['species_id']} from {evt['parent_id']} (year {evt['year']:.0f})")
            elif evt["type"] == "extinction":
                print(f"    ✗ Extinction: {evt['species_id']} (year {evt['year']:.0f})")

    # Top movers
    if d["top_movers"]:
        print(f"\n  Biggest movers: " + ", ".join(
            f"{m['id']} ({m['change_pct']:+.1f}%)" for m in d["top_movers"]
        ))

    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Run calibration with epoch diagnostics")
    parser.add_argument("world_path", type=Path, help="Path to world directory")
    parser.add_argument("--years", type=int, default=1000, help="Years to advance")
    parser.add_argument("--chunk-years", type=int, default=100, help="Years per monitoring chunk")
    parser.add_argument("--threshold", type=int, default=120, help="Species count runaway threshold")
    args = parser.parse_args()

    world = World.open(args.world_path)
    orch = Orchestrator(world)

    ticks_per_chunk = args.chunk_years * 4  # 4 ticks per year
    num_chunks = args.years // args.chunk_years

    diag_path = args.world_path / "diagnostics.jsonl"
    diag_file = open(diag_path, "a")

    start_time = time.time()
    prev_state = {}

    tick_before = world.tier_clocks["ecology"].tick_number

    for chunk in range(num_chunks):
        chunk_start = time.time()
        tick_start = world.tier_clocks["ecology"].tick_number
        orch.advance_seasons(ticks_per_chunk)
        tick_end = world.tier_clocks["ecology"].tick_number

        elapsed = time.time() - chunk_start
        total_elapsed = time.time() - start_time

        diag, prev_state = compute_epoch_diagnostics(
            world, prev_state, tick_start, tick_end
        )
        diag["runtime"] = {"chunk_seconds": round(elapsed, 1), "total_seconds": round(total_elapsed, 1)}

        print_epoch_summary(diag)
        print(f"  Runtime: {elapsed:.0f}s chunk, {total_elapsed:.0f}s total")

        # Write to JSONL
        diag_file.write(json.dumps(diag) + "\n")
        diag_file.flush()

        # Save manifest so snapshots are visible in webview even if killed
        world.save(args.world_path / "world.json")

        if diag["species_count"]["alive"] > args.threshold:
            print(f"\n  *** RUNAWAY: {diag['species_count']['alive']} > {args.threshold} ***")
            break

    print(f"\n{'=' * 70}")
    print(f"  DONE: {time.time() - start_time:.0f}s total")
    print(f"  Diagnostics written to: {diag_path}")
    print(f"{'=' * 70}")

    diag_file.close()
    world.save(args.world_path / "world.json")
    world.close()


if __name__ == "__main__":
    main()

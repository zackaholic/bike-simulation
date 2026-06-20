#!/usr/bin/env python
"""Diagnose species range contraction in calibration world.

Tests two hypotheses:
  H2: Cell gain/loss asymmetry — do losses consistently outpace gains?
  H3: Mortality in suitable territory — are species dying where they should survive?

Reads versioned snapshots (1-30) from calibration_v12, reconstructs suitability
from weather, and produces diagnostic plots + summary tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project source
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bike_sim.world import World
from bike_sim.weather import WeatherSystem

# ── Configuration ────────────────────────────────────────────────────────

WORLD_DIR = Path(__file__).resolve().parent.parent / "worlds" / "calibration_v12"
OUT_DIR = WORLD_DIR / "ride_output"
OUT_DIR.mkdir(exist_ok=True)

SPECIES = [
    "anc_06_valley_thicket",
    "anc_05_ridge_scrub",
    "anc_01_upland_conifer",
]
DENSITY_THRESHOLD = 0.01
SUITABILITY_THRESHOLD = 0.15
VERSIONS = list(range(1, 31))  # snapshots 1-30

# ── Helpers ──────────────────────────────────────────────────────────────


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_suitability(genome, weather_temp, weather_precip):
    """Reconstruct suitability from weather, matching ecology.py logic."""
    # Precipitation -> drought stress
    p_min, p_max = float(weather_precip.min()), float(weather_precip.max())
    p_span = p_max - p_min
    if p_span > 200.0:
        precip_norm = np.clip((weather_precip - p_min) / p_span, 0, 1)
    else:
        precip_norm = np.clip(weather_precip / 3000.0, 0, 1)
    drought_stress = 1.0 - precip_norm

    # Temperature -> warmth
    t_min, t_max = float(weather_temp.min()), float(weather_temp.max())
    t_span = t_max - t_min
    if t_span > 5.0:
        temp_norm = np.clip((weather_temp - t_min) / t_span, 0, 1)
    else:
        temp_norm = np.clip((weather_temp - (-10.0)) / 35.0, 0, 1)

    warmth_preference = 1.0 - genome["frost_tolerance"]

    suit = (
        gaussian(drought_stress, genome["drought_tolerance"], sigma=0.25)
        * gaussian(temp_norm, warmth_preference, sigma=0.25)
    )
    return suit


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    w = World.open(str(WORLD_DIR))

    # Load static rasters (geology doesn't change across versions)
    w.rasters.set_version(1)
    heightmap = w.rasters.read_layer("geology", "heightmap")
    moisture_bias = w.rasters.read_layer("climate_hydrology", "moisture_bias")
    continentality = w.rasters.read_layer("climate_hydrology", "continentality")
    grid_size = heightmap.shape[0]

    # Build weather system
    ws = WeatherSystem(
        world_seed=w.seed,
        geology_heightmap=heightmap,
        grid_size=grid_size,
        cell_size=50.0,
        moisture_bias=moisture_bias,
        continentality=continentality,
    )

    # Load genomes
    genomes = {}
    for sp_id in SPECIES:
        genomes[sp_id] = w.events.get_species(sp_id)["genome"]

    # ── Collect version metadata ─────────────────────────────────────
    version_info = {}
    for v in VERSIONS:
        vi = w.get_version(v)
        tick = vi["tier_clocks"]["ecology"]["tick_number"]
        year = vi["tier_clocks"]["ecology"]["simulated_year"]
        version_info[v] = {"tick": tick, "year": year}

    # ── Pre-load all density rasters ─────────────────────────────────
    # {species: {version: ndarray}}
    densities = {sp: {} for sp in SPECIES}
    for v in VERSIONS:
        for sp_id in SPECIES:
            layer_name = f"species_{sp_id}_density"
            densities[sp_id][v] = w.rasters.read_layer("ecology", layer_name, version=v)

    # Quick sanity check
    for sp_id in SPECIES:
        d1 = densities[sp_id][1]
        d30 = densities[sp_id][30]
        changed = np.sum(d1 != d30)
        print(f"  {sp_id}: v1 sum={d1.sum():.0f}, v30 sum={d30.sum():.0f}, changed cells={changed}")

    # ── TEST 1: Cell gain/loss asymmetry (H2) ────────────────────────
    print("=" * 72)
    print("TEST 1: Cell gain/loss asymmetry (H2)")
    print("=" * 72)

    # {species: list of (year, gained, lost, ratio)}
    flux_data = {sp: [] for sp in SPECIES}

    for sp_id in SPECIES:
        print(f"\n  Species: {sp_id}")
        print(f"  {'Epoch':<12} {'Year':<8} {'Gained':>8} {'Lost':>8} {'G/L Ratio':>10} {'Net':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
        total_gained = 0
        total_lost = 0

        for i in range(len(VERSIONS) - 1):
            v_from = VERSIONS[i]
            v_to = VERSIONS[i + 1]
            year_to = version_info[v_to]["year"]

            d_from = densities[sp_id][v_from]
            d_to = densities[sp_id][v_to]

            present_from = d_from > DENSITY_THRESHOLD
            present_to = d_to > DENSITY_THRESHOLD

            lost = np.sum(present_from & ~present_to)
            gained = np.sum(~present_from & present_to)
            ratio = gained / lost if lost > 0 else float("inf")

            total_gained += gained
            total_lost += lost
            flux_data[sp_id].append((year_to, int(gained), int(lost), ratio))

            print(f"  {v_from:>2} -> {v_to:<5} {year_to:<8.0f} {gained:>8} {lost:>8} {ratio:>10.3f} {gained-lost:>+8}")

        overall_ratio = total_gained / total_lost if total_lost > 0 else float("inf")
        print(f"  {'TOTAL':<12} {'':8} {total_gained:>8} {total_lost:>8} {overall_ratio:>10.3f} {total_gained-total_lost:>+8}")

    # ── TEST 2: Mortality in suitable territory (H3) ─────────────────
    print("\n" + "=" * 72)
    print("TEST 2: Mortality in suitable territory (H3)")
    print("=" * 72)

    # {species: list of (year, frac_suitable_lost, n_lost, median_suit)}
    mort_data = {sp: [] for sp in SPECIES}
    # Collect all lost-cell suitabilities for histogram
    all_lost_suit = {sp: [] for sp in SPECIES}

    for sp_id in SPECIES:
        genome = genomes[sp_id]
        print(f"\n  Species: {sp_id}")
        print(f"    drought_tolerance={genome['drought_tolerance']:.3f}, "
              f"warmth_pref={1 - genome['frost_tolerance']:.3f}")
        print(f"  {'Epoch':<12} {'Year':<8} {'Lost':>8} {'Suitable':>10} {'Frac':>8} {'MedSuit':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

        for i in range(len(VERSIONS) - 1):
            v_from = VERSIONS[i]
            v_to = VERSIONS[i + 1]
            year_to = version_info[v_to]["year"]

            d_from = densities[sp_id][v_from]
            d_to = densities[sp_id][v_to]

            present_from = d_from > DENSITY_THRESHOLD
            present_to = d_to > DENSITY_THRESHOLD
            lost_mask = present_from & ~present_to
            n_lost = int(np.sum(lost_mask))

            if n_lost == 0:
                mort_data[sp_id].append((year_to, 0.0, 0, 0.0))
                print(f"  {v_from:>2} -> {v_to:<5} {year_to:<8.0f} {0:>8} {0:>10} {'N/A':>8} {'N/A':>8}")
                continue

            # Generate weather at the version where loss occurs (v_to's year, summer)
            weather = ws.generate(year_to, season=2)
            suit = compute_suitability(genome, weather.temperature, weather.precipitation)

            lost_suit = suit[lost_mask]
            all_lost_suit[sp_id].extend(lost_suit.tolist())
            n_suitable = int(np.sum(lost_suit > SUITABILITY_THRESHOLD))
            frac = n_suitable / n_lost
            median_s = float(np.median(lost_suit))

            mort_data[sp_id].append((year_to, frac, n_lost, median_s))
            print(f"  {v_from:>2} -> {v_to:<5} {year_to:<8.0f} {n_lost:>8} {n_suitable:>10} {frac:>8.3f} {median_s:>8.3f}")

        # Summary for this species
        all_s = all_lost_suit[sp_id]
        if all_s:
            arr = np.array(all_s)
            frac_all = float(np.mean(arr > SUITABILITY_THRESHOLD))
            print(f"  OVERALL: {len(arr)} lost cells, {frac_all:.1%} had suitability > {SUITABILITY_THRESHOLD}")
            print(f"           median suitability at death: {np.median(arr):.3f}")
            print(f"           mean suitability at death:   {np.mean(arr):.3f}")

    # ── PLOT 1: Cell flux timeseries ─────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Cell Gain/Loss per Epoch (H2: Asymmetry Test)", fontsize=14)

    for ax, sp_id in zip(axes, SPECIES):
        data = flux_data[sp_id]
        years = [d[0] for d in data]
        gained = [d[1] for d in data]
        lost = [-d[2] for d in data]  # negative for visual contrast

        ax.bar(years, gained, width=70, color="green", alpha=0.7, label="Gained")
        ax.bar(years, lost, width=70, color="red", alpha=0.7, label="Lost")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Cell count")
        ax.set_title(sp_id)
        ax.legend(loc="lower left", fontsize=9)

        # Annotate G/L ratio
        for yr, g, l, r in data:
            if l > 0 and r < 10:
                ax.annotate(f"{r:.2f}", (yr, g), fontsize=6,
                            ha="center", va="bottom", color="darkgreen")

    axes[-1].set_xlabel("Simulated Year")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "diagnostic_cell_flux.png", dpi=150)
    print(f"\n  Saved: {OUT_DIR / 'diagnostic_cell_flux.png'}")

    # ── PLOT 2: Suitability at lost cells ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Suitability Distribution at Lost Cells (H3: Mortality in Suitable Territory)", fontsize=13)

    for ax, sp_id in zip(axes, SPECIES):
        suit_vals = all_lost_suit[sp_id]
        if not suit_vals:
            ax.set_title(f"{sp_id}\n(no losses)")
            continue
        arr = np.array(suit_vals)
        ax.hist(arr, bins=50, range=(0, 1), color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(SUITABILITY_THRESHOLD, color="red", linestyle="--",
                   label=f"Threshold ({SUITABILITY_THRESHOLD})")
        frac_above = float(np.mean(arr > SUITABILITY_THRESHOLD))
        ax.set_title(f"{sp_id}\n{frac_above:.0%} lost in suitable habitat")
        ax.set_xlabel("Suitability at loss")
        ax.set_ylabel("Cell count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "diagnostic_mortality_suitability.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'diagnostic_mortality_suitability.png'}")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for sp_id in SPECIES:
        data = flux_data[sp_id]
        total_g = sum(d[1] for d in data)
        total_l = sum(d[2] for d in data)
        ratio = total_g / total_l if total_l > 0 else float("inf")
        suit_vals = all_lost_suit[sp_id]
        if suit_vals:
            arr = np.array(suit_vals)
            frac_suit = float(np.mean(arr > SUITABILITY_THRESHOLD))
            med_suit = float(np.median(arr))
        else:
            frac_suit = 0.0
            med_suit = 0.0

        print(f"\n  {sp_id}:")
        print(f"    H2 — Gain/Loss ratio: {ratio:.3f}  (total gained: {total_g}, lost: {total_l}, net: {total_g - total_l:+d})")
        print(f"    H3 — {frac_suit:.1%} of lost cells had suitability > {SUITABILITY_THRESHOLD} (median suitability at death: {med_suit:.3f})")

        # Trend: is ratio getting worse over time?
        ratios = [d[3] for d in data if d[2] > 0]  # only epochs with losses
        if len(ratios) >= 5:
            first_half = np.mean(ratios[:len(ratios)//2])
            second_half = np.mean(ratios[len(ratios)//2:])
            trend = "worsening" if second_half < first_half else "improving"
            print(f"    Trend: G/L ratio {trend} ({first_half:.3f} early -> {second_half:.3f} late)")

    w.close()


if __name__ == "__main__":
    main()

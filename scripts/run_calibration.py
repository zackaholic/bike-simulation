"""Run a calibration world in chunks with monitoring.

Usage:
    uv run python scripts/run_calibration.py worlds/calibration_v7 --years 1000
    uv run python scripts/run_calibration.py worlds/calibration_v7 --years 2000 --threshold 120
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from bike_sim.world import World
from bike_sim.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run calibration with monitoring")
    parser.add_argument("world_path", type=Path, help="Path to world directory")
    parser.add_argument("--years", type=int, default=1000, help="Years to advance")
    parser.add_argument("--chunk-years", type=int, default=100, help="Years per monitoring chunk")
    parser.add_argument("--threshold", type=int, default=120, help="Species count runaway threshold")
    args = parser.parse_args()

    world = World.open(args.world_path)
    orch = Orchestrator(world)

    ticks_per_chunk = args.chunk_years * 4  # 4 ticks per year
    num_chunks = args.years // args.chunk_years

    start_time = time.time()

    for chunk in range(num_chunks):
        chunk_start = time.time()
        orch.advance_seasons(ticks_per_chunk)

        current_year = world.tier_clocks["ecology"].simulated_year
        species = world.events.list_species()
        alive = [s for s in species if s.get("alive", True)]
        dead = [s for s in species if not s.get("alive", True)]

        ancestor_counts = {}
        for s in alive:
            root = s["species_id"].split("_d")[0]
            ancestor_counts[root] = ancestor_counts.get(root, 0) + 1

        pressures = world.events.get_biotic_pressures()
        pressure_vals = list(pressures.values()) if pressures else [0.0]

        elapsed = time.time() - chunk_start
        total_elapsed = time.time() - start_time

        print(f"\n=== Year {current_year:.0f} ({elapsed:.0f}s chunk, {total_elapsed:.0f}s total) ===")
        print(f"  Species: {len(alive)} alive, {len(dead)} extinct, {len(species)} total")
        print(f"  By ancestor: {ancestor_counts}")
        print(f"  Pressure: mean={sum(pressure_vals)/len(pressure_vals):.4f}, max={max(pressure_vals):.4f}")
        sys.stdout.flush()

        # Save manifest so snapshots are visible in webview even if killed
        world.save(args.world_path / "world.json")

        if len(alive) > args.threshold:
            print(f"\n  *** RUNAWAY: {len(alive)} > {args.threshold} ***")
            break

    print(f"\n=== DONE: {time.time() - start_time:.0f}s total ===")
    world.save(args.world_path / "world.json")
    world.close()


if __name__ == "__main__":
    main()

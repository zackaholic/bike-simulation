"""CLI entry point for the bike-sim simulation.

Run as:
    python -m bike_sim create <world_dir> [--seed 42]
    python -m bike_sim advance <world_dir> <years>
    python -m bike_sim ride <world_dir> <duration_minutes>
    python -m bike_sim status <world_dir>
    python -m bike_sim fire <world_dir> <x> <y>
    python -m bike_sim visualize <world_dir> [--output-dir output/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bike_sim.orchestrator import Orchestrator
from bike_sim.world import World


def _cmd_create(args: argparse.Namespace) -> None:
    """Create a new world and run initial geology + climate passes."""
    world_dir = Path(args.world_dir)
    seed = args.seed

    print(f"Creating world at {world_dir} (seed={seed})...")
    world = World.create(world_dir, seed=seed)
    orch = Orchestrator(world)
    orch.create_world()

    world.save(world_dir / "world.json")
    info = orch.status()
    print("World created.")
    print(f"  Seed: {info['seed']}")
    print(f"  Geology ticks: {world.tier_clocks['geology'].tick_number}")
    print(f"  Climate ticks: {world.tier_clocks['climate_hydrology'].tick_number}")
    print(f"  Ecology ticks: {world.tier_clocks['ecology'].tick_number}")
    print(f"  Version: {world.current_version}")
    world.close()


def _cmd_advance(args: argparse.Namespace) -> None:
    """Open an existing world and advance by N years."""
    world_dir = Path(args.world_dir)
    years = args.years

    print(f"Opening world at {world_dir}...")
    world = World.open(world_dir)
    orch = Orchestrator(world)

    print(f"Advancing {years} years...")
    result = orch.advance(years)

    world.save(world_dir / "world.json")
    print(f"Advanced {result['years_advanced']:.1f} years.")
    print(f"  Ecology ticks: {result['ecology_ticks']}")
    print(f"  Climate ticks: {result['climate_hydrology_ticks']}")
    print(f"  Geology ticks: {result['geology_ticks']}")
    print(f"  Version: {world.current_version}")
    world.close()


def _cmd_ride(args: argparse.Namespace) -> None:
    """Open an existing world and advance by ride duration (minutes -> sim years)."""
    world_dir = Path(args.world_dir)
    minutes = args.duration_minutes

    print(f"Opening world at {world_dir}...")
    world = World.open(world_dir)
    orch = Orchestrator(world)

    print(f"Riding for {minutes} minutes...")
    result = orch.advance_ride(minutes)

    world.save(world_dir / "world.json")
    print(f"Ride complete. Advanced {result['years_advanced']:.1f} sim-years.")
    print(f"  Ecology ticks: {result['ecology_ticks']}")
    print(f"  Climate ticks: {result['climate_hydrology_ticks']}")
    print(f"  Geology ticks: {result['geology_ticks']}")
    print(f"  Version: {world.current_version}")
    world.close()


def _cmd_status(args: argparse.Namespace) -> None:
    """Open an existing world and print its status."""
    world_dir = Path(args.world_dir)

    world = World.open(world_dir)
    orch = Orchestrator(world)
    info = orch.status()

    print(f"World: {world_dir}")
    print(f"  Seed: {info['seed']}")
    print(f"  Simulated year: {world.simulated_year:.1f}")
    for tier_name, clock in world.tier_clocks.items():
        print(f"  {tier_name}: tick {clock.tick_number}, year {clock.simulated_year:.1f}")
    print(f"  Species count: {info['species_count']}")
    print(f"  Individual count: {info['individual_count']}")
    print(f"  Versions: {info['total_versions']}")
    world.close()


def _cmd_fire(args: argparse.Namespace) -> None:
    """Open an existing world and introduce a fire at (x, y)."""
    world_dir = Path(args.world_dir)
    x, y = args.x, args.y

    print(f"Opening world at {world_dir}...")
    world = World.open(world_dir)
    orch = Orchestrator(world)

    print(f"Introducing fire at ({x}, {y})...")
    orch.introduce_fire(x, y)

    world.save(world_dir / "world.json")
    print("Fire introduced.")
    world.close()


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Open an existing world and render all layers to PNGs."""
    world_dir = Path(args.world_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid matplotlib overhead for non-visualize commands.
    from bike_sim.extract.debug2d.visualizer import render_individuals, render_layer

    print(f"Opening world at {world_dir}...")
    world = World.open(world_dir)

    tiers = world.rasters.list_tiers()

    # Colormap choices for known layers.
    layer_cmaps = {
        "heightmap": "terrain",
        "eroded_heightmap": "terrain",
        "temperature": "coolwarm",
    }

    for tier in tiers:
        layers = world.rasters.list_layers(tier)
        for layer_name in layers:
            cmap = layer_cmaps.get(layer_name, "Greens" if "density" in layer_name else "viridis")
            out = output_dir / f"{tier}_{layer_name}.png"
            print(f"  Rendering {tier}/{layer_name} -> {out}")
            render_layer(world, tier, layer_name, out, cmap=cmap)

    # Individuals overlay on heightmap if available.
    has_heightmap = "heightmap" in world.rasters.list_layers("geology")
    if has_heightmap:
        out = output_dir / "individuals.png"
        print(f"  Rendering individuals overlay -> {out}")
        render_individuals(world, out)

    world.close()
    print(f"Done. Output in {output_dir}/")


def _cmd_ride_experience(args: argparse.Namespace) -> None:
    """Generate a ride path and experience profile."""
    from bike_sim.extract.ride_experience import run_ride_experience

    output_dir = getattr(args, "output_dir", None)
    run_ride_experience(args.world_dir, output_dir=output_dir)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m bike_sim",
        description="CLI for the bike-sim world simulation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- create --
    p_create = subparsers.add_parser("create", help="Create a new world.")
    p_create.add_argument("world_dir", help="Path for the new world directory.")
    p_create.add_argument("--seed", type=int, default=42, help="World seed (default: 42).")

    # -- advance --
    p_advance = subparsers.add_parser("advance", help="Advance a world by N years.")
    p_advance.add_argument("world_dir", help="Path to an existing world directory.")
    p_advance.add_argument("years", type=float, help="Number of years to advance.")

    # -- ride --
    p_ride = subparsers.add_parser("ride", help="Advance a world by ride duration.")
    p_ride.add_argument("world_dir", help="Path to an existing world directory.")
    p_ride.add_argument("duration_minutes", type=float, help="Ride duration in minutes.")

    # -- status --
    p_status = subparsers.add_parser("status", help="Print world status.")
    p_status.add_argument("world_dir", help="Path to an existing world directory.")

    # -- fire --
    p_fire = subparsers.add_parser("fire", help="Introduce a fire at (x, y).")
    p_fire.add_argument("world_dir", help="Path to an existing world directory.")
    p_fire.add_argument("x", type=float, help="X coordinate (meters).")
    p_fire.add_argument("y", type=float, help="Y coordinate (meters).")

    # -- visualize --
    p_viz = subparsers.add_parser("visualize", help="Render all layers to PNGs.")
    p_viz.add_argument("world_dir", help="Path to an existing world directory.")
    p_viz.add_argument(
        "--output-dir",
        default="output/",
        help="Directory to save PNGs (default: output/).",
    )

    # -- ride-experience --
    p_rexp = subparsers.add_parser(
        "ride-experience", help="Generate ride path and experience profile.",
    )
    p_rexp.add_argument("world_dir", help="Path to an existing world directory.")
    p_rexp.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <world_dir>/ride_output/).",
    )

    args = parser.parse_args(argv)

    dispatch = {
        "create": _cmd_create,
        "advance": _cmd_advance,
        "ride": _cmd_ride,
        "status": _cmd_status,
        "fire": _cmd_fire,
        "visualize": _cmd_visualize,
        "ride-experience": _cmd_ride_experience,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

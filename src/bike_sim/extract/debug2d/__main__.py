"""CLI entry point for the debug 2D visualizer.

Run as:
    python -m bike_sim.extract.debug2d demo [--seed 42] [--output-dir ./output/demo]
    python -m bike_sim.extract.debug2d render <world_dir> [--output-dir ./output] \
        [--tier geology] [--layer heightmap]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from bike_sim.extract.debug2d.synthetic import create_synthetic_world
from bike_sim.extract.debug2d.visualizer import (
    render_composite,
    render_individuals,
    render_layer,
)
from bike_sim.world import World

# Colormaps chosen per tier/layer for readability
_LAYER_CMAPS: dict[str, str] = {
    "heightmap": "terrain",
    "bedrock_type": "tab10",
    "soil_moisture_summer": "YlGnBu",
}

_DEFAULT_CMAP = "viridis"


def _cmap_for(layer_name: str) -> str:
    """Pick a sensible colormap for a layer name."""
    if "density" in layer_name:
        return "Greens"
    return _LAYER_CMAPS.get(layer_name, _DEFAULT_CMAP)


def _list_tiers(world: World) -> list[str]:
    """List all tier names in a world's raster store."""
    return list(world.rasters._root.keys())


def _render_all_layers(world: World, output_dir: Path) -> None:
    """Render every raster layer in the world, plus composites and individuals."""
    tiers = _list_tiers(world)
    for tier in tiers:
        layers = world.rasters.list_layers(tier)
        for layer_name in layers:
            out = output_dir / f"{tier}_{layer_name}.png"
            print(f"  Rendering {tier}/{layer_name} -> {out}")
            render_layer(world, tier, layer_name, out, cmap=_cmap_for(layer_name))

    # Composite: heightmap + moisture if both exist
    has_heightmap = "heightmap" in world.rasters.list_layers("geology")
    has_moisture = "soil_moisture_summer" in world.rasters.list_layers("climate_hydrology")
    if has_heightmap and has_moisture:
        out = output_dir / "composite_heightmap_moisture.png"
        print(f"  Rendering composite (heightmap + moisture) -> {out}")
        render_composite(
            world,
            layers=[
                {"tier": "geology", "layer_name": "heightmap", "cmap": "terrain", "alpha": 1.0},
                {
                    "tier": "climate_hydrology",
                    "layer_name": "soil_moisture_summer",
                    "cmap": "Blues",
                    "alpha": 0.4,
                },
            ],
            output_path=out,
            title="Composite: heightmap + soil moisture",
        )

    # Individuals overlay
    if has_heightmap:
        out = output_dir / "individuals.png"
        print(f"  Rendering individuals overlay -> {out}")
        render_individuals(world, out)


def _cmd_demo(args: argparse.Namespace) -> None:
    """Execute the 'demo' subcommand."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic world (seed={args.seed})...")
    with tempfile.TemporaryDirectory() as tmp:
        world = create_synthetic_world(Path(tmp), seed=args.seed)
        print(f"Rendering layers to {output_dir}/")
        _render_all_layers(world, output_dir)
        world.close()

    print("Done.")


def _cmd_render(args: argparse.Namespace) -> None:
    """Execute the 'render' subcommand."""
    world_dir = Path(args.world_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening world at {world_dir}")
    world = World.open(world_dir)

    try:
        if args.tier and args.layer:
            # Single layer
            out = output_dir / f"{args.tier}_{args.layer}.png"
            print(f"  Rendering {args.tier}/{args.layer} -> {out}")
            render_layer(world, args.tier, args.layer, out, cmap=_cmap_for(args.layer))
        elif args.tier:
            # All layers in one tier
            layers = world.rasters.list_layers(args.tier)
            if not layers:
                print(f"  No layers found in tier '{args.tier}'")
                return
            for layer_name in layers:
                out = output_dir / f"{args.tier}_{layer_name}.png"
                print(f"  Rendering {args.tier}/{layer_name} -> {out}")
                render_layer(world, args.tier, layer_name, out, cmap=_cmap_for(layer_name))
        else:
            # Everything
            _render_all_layers(world, output_dir)
    finally:
        world.close()

    print("Done.")


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m bike_sim.extract.debug2d",
        description="Debug 2D visualizer for bike-sim worlds.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- demo --
    demo_parser = subparsers.add_parser(
        "demo",
        help="Generate a synthetic world and render all layers as PNGs.",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="World seed for reproducible generation (default: 42).",
    )
    demo_parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/demo",
        help="Directory to save PNGs (default: ./output/demo).",
    )

    # -- render --
    render_parser = subparsers.add_parser(
        "render",
        help="Render layers from an existing world directory.",
    )
    render_parser.add_argument(
        "world_dir",
        type=str,
        help="Path to an existing world directory.",
    )
    render_parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save PNGs (default: ./output).",
    )
    render_parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Render only layers from this tier.",
    )
    render_parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Render only this specific layer (requires --tier).",
    )

    args = parser.parse_args()

    if args.command == "demo":
        _cmd_demo(args)
    elif args.command == "render":
        if args.layer and not args.tier:
            render_parser.error("--layer requires --tier")
        _cmd_render(args)


if __name__ == "__main__":
    main()

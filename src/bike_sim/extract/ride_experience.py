"""Ride experience tool — path generation, perpendicular sampling, and experience graphs.

Generates a meandering bike path through the world that avoids steep grades
and tight turns, then samples species density along perpendicular cross-sections
to produce a 1D "experience slice" of the 2D world.

The path is stored as a raster layer for webview display, and the experience
data can be plotted as a multi-series graph showing what the rider sees at
each point along the route.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bike_sim.world import World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE: float = 50.0  # meters per cell
GRID_SIZE: int = 1000
WORLD_EXTENT: float = CELL_SIZE * GRID_SIZE  # 50,000m

# Path generation
MAX_GRADE_PCT: float = 8.0       # reject edges steeper than 8%
GRADE_COST_WEIGHT: float = 5.0   # cost multiplier for grade
TURN_COST_WEIGHT: float = 2.0    # cost multiplier for heading change
MIN_TURN_RADIUS_CELLS: int = 3   # ~150m minimum turn radius

# Sampling
SAMPLE_INTERVAL_M: float = 50.0   # sample every 50m of path distance
SAMPLE_BAR_HALF_M: float = 50.0   # ±50m perpendicular to heading
SAMPLE_BAR_POINTS: int = 5        # points across the bar (evenly spaced)

# Waypoint generation
NUM_WAYPOINTS: int = 12           # spread across the map for coverage
WAYPOINT_MARGIN: int = 50         # cells from edge


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------

def _compute_grade_grid(heightmap: np.ndarray) -> np.ndarray:
    """Compute grade (%) between adjacent cells in all 8 directions.

    Returns array of shape (rows, cols) with the maximum absolute grade
    to any neighbor, useful for cost estimation.
    """
    dy, dx = np.gradient(heightmap, CELL_SIZE)
    return np.sqrt(dx**2 + dy**2) * 100.0  # percent grade


def _edge_cost(
    heightmap: np.ndarray,
    r1: int, c1: int,
    r2: int, c2: int,
    prev_dir: tuple[int, int] | None = None,
) -> float | None:
    """Cost of moving from cell (r1,c1) to (r2,c2).

    Returns None if the edge is impassable (grade > MAX_GRADE_PCT).
    """
    dr, dc = r2 - r1, c2 - c1
    dist = np.sqrt(dr**2 + dc**2) * CELL_SIZE
    dh = abs(float(heightmap[r2, c2] - heightmap[r1, c1]))
    grade = (dh / dist) * 100.0 if dist > 0 else 0.0

    if grade > MAX_GRADE_PCT:
        return None

    # Base cost is distance
    cost = dist

    # Grade penalty (quadratic — gentle slopes are cheap, steep ones expensive)
    cost += GRADE_COST_WEIGHT * (grade / MAX_GRADE_PCT) ** 2 * dist

    # Turn penalty — discourage sharp direction changes
    if prev_dir is not None:
        cur_dir = (dr, dc)
        # Dot product of normalized direction vectors
        len_prev = np.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
        len_cur = np.sqrt(cur_dir[0]**2 + cur_dir[1]**2)
        if len_prev > 0 and len_cur > 0:
            dot = (prev_dir[0] * cur_dir[0] + prev_dir[1] * cur_dir[1]) / (len_prev * len_cur)
            dot = max(-1.0, min(1.0, dot))
            # 1 - dot: 0 for straight, 1 for 90°, 2 for reversal
            cost += TURN_COST_WEIGHT * (1.0 - dot) * dist

    return cost


def _astar_segment(
    heightmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_steps: int = 500_000,
) -> list[tuple[int, int]] | None:
    """A* pathfinding from start to goal on the heightmap grid.

    Returns list of (row, col) cells, or None if no path found.
    """
    rows, cols = heightmap.shape

    # Heuristic: Euclidean distance in meters
    def h(r: int, c: int) -> float:
        return np.sqrt((r - goal[0])**2 + (c - goal[1])**2) * CELL_SIZE

    # Priority queue: (f_cost, g_cost, row, col, prev_direction)
    open_set: list[tuple[float, float, int, int, tuple[int, int] | None]] = []
    heapq.heappush(open_set, (h(start[0], start[1]), 0.0, start[0], start[1], None))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    # 8-connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    steps = 0
    while open_set and steps < max_steps:
        steps += 1
        f, g, r, c, prev_dir = heapq.heappop(open_set)

        if (r, c) == goal:
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return list(reversed(path))

        if g > g_score.get((r, c), float("inf")):
            continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                ec = _edge_cost(heightmap, r, c, nr, nc, prev_dir)
                if ec is None:
                    continue
                ng = g + ec
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc, (dr, dc)))

    return None  # No path found


def _generate_waypoints(
    heightmap: np.ndarray,
    rng: np.random.Generator,
    num_points: int = NUM_WAYPOINTS,
) -> list[tuple[int, int]]:
    """Generate well-spread waypoints avoiding steep terrain.

    Uses a Poisson-disk-like approach: generate candidates, pick those
    that maximize minimum distance to already-chosen points while
    avoiding extreme grades.
    """
    grade = _compute_grade_grid(heightmap)
    rows, cols = heightmap.shape
    margin = WAYPOINT_MARGIN

    # Candidate pool: cells with grade < 5% in the interior
    candidates_r, candidates_c = np.where(
        (grade[margin:-margin, margin:-margin] < 5.0)
    )
    candidates_r += margin
    candidates_c += margin

    if len(candidates_r) < num_points:
        # Fallback: relax grade constraint
        candidates_r, candidates_c = np.where(
            (grade[margin:-margin, margin:-margin] < MAX_GRADE_PCT)
        )
        candidates_r += margin
        candidates_c += margin

    if len(candidates_r) < num_points:
        # Extreme fallback: use random interior points
        candidates_r = rng.integers(margin, rows - margin, size=num_points * 10)
        candidates_c = rng.integers(margin, cols - margin, size=num_points * 10)

    waypoints: list[tuple[int, int]] = []

    # First point: near center
    center_r, center_c = rows // 2, cols // 2
    dists = (candidates_r - center_r)**2 + (candidates_c - center_c)**2
    idx = int(np.argmin(dists))
    waypoints.append((int(candidates_r[idx]), int(candidates_c[idx])))

    # Greedy farthest-point sampling
    for _ in range(num_points - 1):
        min_dists = np.full(len(candidates_r), float("inf"))
        for wr, wc in waypoints:
            d = (candidates_r - wr)**2 + (candidates_c - wc)**2
            min_dists = np.minimum(min_dists, d)
        idx = int(np.argmax(min_dists))
        waypoints.append((int(candidates_r[idx]), int(candidates_c[idx])))

    return waypoints


def _order_waypoints_tsp(
    waypoints: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Order waypoints using nearest-neighbor TSP heuristic.

    Starts from waypoint[0] (near center) and greedily visits nearest unvisited.
    """
    if len(waypoints) <= 2:
        return waypoints

    ordered = [waypoints[0]]
    remaining = set(range(1, len(waypoints)))

    while remaining:
        last = ordered[-1]
        best_idx = min(remaining, key=lambda i: (
            (waypoints[i][0] - last[0])**2 + (waypoints[i][1] - last[1])**2
        ))
        ordered.append(waypoints[best_idx])
        remaining.remove(best_idx)

    return ordered


def generate_ride_path(
    heightmap: np.ndarray,
    world_seed: int,
) -> list[tuple[int, int]]:
    """Generate a full ride path through the world.

    1. Generate well-spread waypoints on gentle terrain.
    2. Order them via nearest-neighbor TSP.
    3. Connect consecutive waypoints via A* on a grade cost surface.

    Returns list of (row, col) cells comprising the path.
    """
    from bike_sim.rng import create_rng
    rng = create_rng(world_seed, "ride", "path", 0)

    waypoints = _generate_waypoints(heightmap, rng)
    ordered = _order_waypoints_tsp(waypoints)

    full_path: list[tuple[int, int]] = []
    for i in range(len(ordered) - 1):
        segment = _astar_segment(heightmap, ordered[i], ordered[i + 1])
        if segment is None:
            # Skip impassable segments
            continue
        if full_path and segment[0] == full_path[-1]:
            segment = segment[1:]
        full_path.extend(segment)

    # Close the loop back to start
    if len(ordered) > 1 and full_path:
        closing = _astar_segment(heightmap, ordered[-1], ordered[0])
        if closing is not None:
            if closing[0] == full_path[-1]:
                closing = closing[1:]
            full_path.extend(closing)

    return full_path


# ---------------------------------------------------------------------------
# Path to world coordinates + distance
# ---------------------------------------------------------------------------

def _path_to_world_coords(
    path: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cell path to world coordinates and cumulative distances.

    Returns (xy_array[N,2], headings[N], cum_dist[N]) where:
    - xy_array has world-space (x, y) in meters (col*50+25, row*50+25)
    - headings[i] is the heading angle in radians at point i
    - cum_dist[i] is cumulative distance from start in meters
    """
    coords = np.array([
        (c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2)
        for r, c in path
    ], dtype=np.float64)

    # Segment distances
    diffs = np.diff(coords, axis=0)
    seg_dists = np.sqrt((diffs**2).sum(axis=1))
    cum_dist = np.zeros(len(coords))
    cum_dist[1:] = np.cumsum(seg_dists)

    # Headings (angle from positive x-axis, radians)
    headings = np.zeros(len(coords))
    if len(diffs) > 0:
        headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
        headings[-1] = headings[-2]  # extend last heading

    return coords, headings, cum_dist


# ---------------------------------------------------------------------------
# Perpendicular sampling
# ---------------------------------------------------------------------------

def sample_ride_experience(
    world: "World",
    path: list[tuple[int, int]],
    sample_interval: float = SAMPLE_INTERVAL_M,
    bar_half: float = SAMPLE_BAR_HALF_M,
    bar_points: int = SAMPLE_BAR_POINTS,
    version: int | None = None,
) -> dict:
    """Sample species density and terrain along perpendicular cross-sections.

    For every `sample_interval` meters of path distance, places a bar
    perpendicular to the heading extending ±bar_half meters, samples
    bar_points along it, and averages.

    Returns a dict with:
    - distances: [N] sample distances along path (meters)
    - elevation: [N] elevation at path center
    - grade: [N] grade (%) at path center
    - species: {species_id: [N] mean density across bar}
    - ground_cover_type: [N] dominant cover type at center
    - ground_cover_vigor: [N] vigor at center
    """
    import re

    store = world.rasters
    coords, headings, cum_dist = _path_to_world_coords(path)
    total_dist = cum_dist[-1] if len(cum_dist) > 0 else 0.0

    # Determine sample points along the path
    sample_dists = np.arange(0, total_dist, sample_interval)
    n_samples = len(sample_dists)

    if n_samples == 0:
        return {"distances": [], "elevation": [], "grade": [], "species": {},
                "ground_cover_type": [], "ground_cover_vigor": []}

    # Interpolate path positions and headings at sample distances
    sample_x = np.interp(sample_dists, cum_dist, coords[:, 0])
    sample_y = np.interp(sample_dists, cum_dist, coords[:, 1])
    sample_heading = np.interp(sample_dists, cum_dist, headings)

    # Load raster layers (version-aware)
    heightmap = store.read_layer("geology", "heightmap", version=version)

    # Find all species density layers
    species_pattern = re.compile(r"^species_(.+)_density$")
    species_layers: dict[str, np.ndarray] = {}
    for layer_name in store.list_layers("ecology", version=version):
        m = species_pattern.match(layer_name)
        if m:
            species_layers[m.group(1)] = store.read_layer("ecology", layer_name, version=version)

    # Try to load ground cover
    eco_layers = store.list_layers("ecology", version=version)
    has_ground_cover = "ground_cover_type" in eco_layers
    if has_ground_cover:
        gc_type = store.read_layer("ecology", "ground_cover_type", version=version)
        gc_vigor = store.read_layer("ecology", "ground_cover_vigor", version=version)

    # Bar offsets perpendicular to heading
    bar_offsets = np.linspace(-bar_half, bar_half, bar_points)

    # Sample at each point
    elevations = np.zeros(n_samples)
    grades = np.zeros(n_samples)
    gc_types = np.zeros(n_samples, dtype=np.int32)
    gc_vigors = np.zeros(n_samples)
    species_data: dict[str, np.ndarray] = {
        sid: np.zeros(n_samples) for sid in species_layers
    }

    for i in range(n_samples):
        cx, cy = sample_x[i], sample_y[i]
        heading = sample_heading[i]

        # Perpendicular direction (rotate heading by 90°)
        perp_dx = -np.sin(heading)
        perp_dy = np.cos(heading)

        # Sample points along the bar
        bar_densities: dict[str, list[float]] = {sid: [] for sid in species_layers}

        for offset in bar_offsets:
            bx = cx + perp_dx * offset
            by = cy + perp_dy * offset

            # Clamp to world bounds
            bx = max(0, min(WORLD_EXTENT - 1, bx))
            by = max(0, min(WORLD_EXTENT - 1, by))

            col = min(int(bx / CELL_SIZE), GRID_SIZE - 1)
            row = min(int(by / CELL_SIZE), GRID_SIZE - 1)

            for sid, density_arr in species_layers.items():
                bar_densities[sid].append(float(density_arr[row, col]))

        # Average across bar
        for sid in species_layers:
            species_data[sid][i] = float(np.mean(bar_densities[sid]))

        # Center point samples
        center_col = min(int(cx / CELL_SIZE), GRID_SIZE - 1)
        center_row = min(int(cy / CELL_SIZE), GRID_SIZE - 1)
        elevations[i] = float(heightmap[center_row, center_col])

        # Grade from heightmap gradient
        if i > 0:
            dh = abs(elevations[i] - elevations[i - 1])
            dd = sample_dists[i] - sample_dists[i - 1]
            grades[i] = (dh / dd * 100.0) if dd > 0 else 0.0

        if has_ground_cover:
            gc_types[i] = int(gc_type[center_row, center_col])
            gc_vigors[i] = float(gc_vigor[center_row, center_col])

    # Filter to species with any density along the route
    active_species = {
        sid: densities.tolist()
        for sid, densities in species_data.items()
        if densities.max() > 0.01
    }

    return {
        "distances": sample_dists.tolist(),
        "elevation": elevations.tolist(),
        "grade": grades.tolist(),
        "species": active_species,
        "ground_cover_type": gc_types.tolist(),
        "ground_cover_vigor": gc_vigors.tolist(),
        "total_distance_km": total_dist / 1000.0,
        "num_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Rasterize path for webview display
# ---------------------------------------------------------------------------

def rasterize_path(
    path: list[tuple[int, int]],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Create a raster layer marking path cells.

    Value is the cumulative distance along the path (in meters) at each cell,
    or 0 for cells not on the path. Useful for coloring the path by progress.
    """
    raster = np.zeros((grid_size, grid_size), dtype=np.float32)
    prev_r, prev_c = None, None
    cum_dist = 0.0

    for r, c in path:
        if prev_r is not None:
            dr = r - prev_r
            dc = c - prev_c
            cum_dist += np.sqrt(dr**2 + dc**2) * CELL_SIZE
        raster[r, c] = cum_dist + 1.0  # +1 so start cell isn't zero
        prev_r, prev_c = r, c

    return raster


# ---------------------------------------------------------------------------
# Plot experience graph
# ---------------------------------------------------------------------------

def plot_experience(
    experience: dict,
    output_path: str | Path,
    title: str = "Ride Experience Profile",
) -> None:
    """Plot the ride experience as a multi-panel figure.

    Panels:
    1. Elevation profile + grade
    2. Species density along ride (stacked area)
    3. Ground cover type + vigor
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    distances_km = [d / 1000.0 for d in experience["distances"]]
    n = len(distances_km)
    if n == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Panel 1: Elevation + Grade
    ax1 = axes[0]
    ax1.fill_between(distances_km, experience["elevation"], alpha=0.3, color="brown")
    ax1.plot(distances_km, experience["elevation"], color="brown", linewidth=0.8)
    ax1.set_ylabel("Elevation (m)", color="brown")
    ax1.tick_params(axis="y", labelcolor="brown")

    ax1b = ax1.twinx()
    ax1b.plot(distances_km, experience["grade"], color="red", alpha=0.5, linewidth=0.5)
    ax1b.set_ylabel("Grade (%)", color="red")
    ax1b.tick_params(axis="y", labelcolor="red")
    ax1b.set_ylim(0, MAX_GRADE_PCT * 1.5)
    ax1.set_title(title)

    # Panel 2: Species density
    ax2 = axes[1]
    species = experience["species"]
    if species:
        # Sort by total density (most abundant on bottom)
        sorted_species = sorted(species.keys(), key=lambda s: sum(species[s]), reverse=True)

        # Generate distinct colors
        n_sp = len(sorted_species)
        colors = [hsv_to_rgb((i / n_sp, 0.7, 0.85)) for i in range(n_sp)]

        # Plot as overlapping filled areas (not stacked — densities are independent)
        for idx, sid in enumerate(sorted_species):
            densities = species[sid]
            ax2.fill_between(distances_km, densities, alpha=0.3, color=colors[idx])
            ax2.plot(distances_km, densities, linewidth=0.8, color=colors[idx], label=sid)

        ax2.legend(loc="upper right", fontsize=6, ncol=min(4, n_sp))
    ax2.set_ylabel("Species Density")

    # Panel 3: Ground cover
    ax3 = axes[2]
    cover_names = ["bare_rock", "lichen", "dry_grass", "patchy", "meadow",
                   "lush_grass", "fern_moss", "alpine"]
    cover_colors = ["#8B7355", "#A0A060", "#C8B560", "#90B060",
                    "#60C040", "#20A020", "#208060", "#B0B0D0"]

    gc_types = experience["ground_cover_type"]
    gc_vigors = experience["ground_cover_vigor"]

    # Color-coded scatter by ground cover type
    for cover_idx in range(8):
        mask = [j for j in range(n) if gc_types[j] == cover_idx]
        if mask:
            ax3.scatter(
                [distances_km[j] for j in mask],
                [gc_vigors[j] for j in mask],
                c=cover_colors[cover_idx],
                label=cover_names[cover_idx],
                s=2, alpha=0.6,
            )
    ax3.set_ylabel("Ground Cover Vigor")
    ax3.set_xlabel("Distance Along Ride (km)")
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right", fontsize=6, ncol=4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save/load path
# ---------------------------------------------------------------------------

def save_path(path: list[tuple[int, int]], filepath: str | Path) -> None:
    """Save path to JSON file."""
    Path(filepath).write_text(json.dumps(path))


def load_path(filepath: str | Path) -> list[tuple[int, int]]:
    """Load path from JSON file."""
    data = json.loads(Path(filepath).read_text())
    return [(r, c) for r, c in data]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_ride_experience(
    world_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    """Generate ride path, sample experience, produce outputs.

    Returns the experience dict and saves:
    - ride_path.json — the raw path
    - ride_path raster layer in the world
    - ride_experience.json — full sample data
    - ride_experience.png — the experience graph
    """
    from bike_sim.world import World

    world_dir = Path(world_dir)
    if output_dir is None:
        output_dir = world_dir / "ride_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World.open(world_dir)
    heightmap = world.rasters.read_layer("geology", "heightmap")

    print(f"Generating ride path (world seed {world.seed})...")
    path = generate_ride_path(heightmap, world.seed)
    print(f"  Path: {len(path)} cells")

    if not path:
        print("  ERROR: No path generated!")
        return {}

    # Compute total distance
    _, _, cum_dist = _path_to_world_coords(path)
    total_km = cum_dist[-1] / 1000.0 if len(cum_dist) > 0 else 0.0
    print(f"  Total distance: {total_km:.1f} km")

    # Save path
    save_path(path, output_dir / "ride_path.json")

    # Write path as raster layer
    path_raster = rasterize_path(path)
    world.rasters.write_layer("ride", "path_distance", path_raster, 0)
    print("  Path raster written to ride/path_distance layer")

    # Sample experience
    print("Sampling ride experience...")
    experience = sample_ride_experience(world, path)
    print(f"  {experience['num_samples']} samples over {experience['total_distance_km']:.1f} km")
    print(f"  {len(experience['species'])} species visible along route")

    # Save experience data
    exp_path = output_dir / "ride_experience.json"
    Path(exp_path).write_text(json.dumps(experience, indent=2))

    # Plot
    plot_path = output_dir / "ride_experience.png"
    plot_experience(experience, plot_path)
    print(f"  Experience graph: {plot_path}")

    # Summary
    print("\nRide Experience Summary:")
    print(f"  Total distance: {experience['total_distance_km']:.1f} km")
    print(f"  Elevation range: {min(experience['elevation']):.0f}m - {max(experience['elevation']):.0f}m")
    print(f"  Max grade: {max(experience['grade']):.1f}%")
    if experience["species"]:
        # Sort species by peak density along route
        ranked = sorted(
            experience["species"].items(),
            key=lambda kv: max(kv[1]),
            reverse=True,
        )
        print(f"  Top species along ride:")
        for sid, densities in ranked[:8]:
            print(f"    {sid}: peak {max(densities):.1f}, mean {np.mean(densities):.1f}")

    return experience


# ---------------------------------------------------------------------------
# Multi-snapshot comparison
# ---------------------------------------------------------------------------

def plot_snapshot_comparison(
    snapshots: dict[str, dict],
    output_path: str | Path,
    title: str = "Ride Experience Across Time",
) -> None:
    """Plot species density along the ride for multiple snapshots.

    Each snapshot gets its own sub-panel showing the species density profile,
    with shared x-axis (ride distance) for easy comparison.

    Args:
        snapshots: {label: experience_dict} ordered by time
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    labels = list(snapshots.keys())
    n_panels = len(labels)
    if n_panels == 0:
        return

    # Collect all species across all snapshots for consistent coloring
    all_species: set[str] = set()
    for exp in snapshots.values():
        all_species.update(exp.get("species", {}).keys())
    sorted_species = sorted(all_species)
    n_sp = len(sorted_species)
    colors = {
        sid: hsv_to_rgb((i / max(n_sp, 1), 0.7, 0.85))
        for i, sid in enumerate(sorted_species)
    }

    fig, axes = plt.subplots(
        n_panels + 1, 1,
        figsize=(16, 3 * (n_panels + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [1] + [1] * n_panels},
    )

    # Top panel: elevation (same for all snapshots)
    first_exp = next(iter(snapshots.values()))
    distances_km = [d / 1000.0 for d in first_exp["distances"]]

    ax0 = axes[0]
    ax0.fill_between(distances_km, first_exp["elevation"], alpha=0.3, color="brown")
    ax0.plot(distances_km, first_exp["elevation"], color="brown", linewidth=0.8)
    ax0.set_ylabel("Elevation (m)")
    ax0.set_title(title)

    # Find global y-max for consistent species density scale
    y_max = 0.01
    for exp in snapshots.values():
        for densities in exp.get("species", {}).values():
            if densities:
                y_max = max(y_max, max(densities))
    y_max *= 1.1

    # Species panels
    for i, label in enumerate(labels):
        ax = axes[i + 1]
        exp = snapshots[label]
        species = exp.get("species", {})

        # Sort by total density for this snapshot
        present = sorted(
            [(sid, species[sid]) for sid in sorted_species if sid in species],
            key=lambda kv: sum(kv[1]),
            reverse=True,
        )

        for sid, densities in present:
            ax.fill_between(distances_km[:len(densities)], densities, alpha=0.25, color=colors[sid])
            ax.plot(distances_km[:len(densities)], densities, linewidth=0.7, color=colors[sid], label=sid)

        ax.set_ylabel(f"Density\n{label}")
        ax.set_ylim(0, y_max)

        # Only show legend on first species panel
        if i == 0 and present:
            ax.legend(loc="upper right", fontsize=5, ncol=min(5, len(present)))

    axes[-1].set_xlabel("Distance Along Ride (km)")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_snapshot_comparison(
    world_dir: str | Path,
    output_dir: str | Path | None = None,
    snapshot_versions: list[int] | None = None,
) -> None:
    """Sample the ride experience at multiple snapshots and produce a comparison.

    If snapshot_versions is None, samples every available snapshot.
    Reuses existing ride path from ride_output/ride_path.json.
    """
    from bike_sim.world import World

    world_dir = Path(world_dir)
    if output_dir is None:
        output_dir = world_dir / "ride_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing path
    path_file = output_dir / "ride_path.json"
    if not path_file.exists():
        print("No ride path found. Run ride-experience first.")
        return
    path = load_path(path_file)

    world = World.open(world_dir)

    # Determine which snapshots to sample
    if snapshot_versions is None:
        versions_info = world.list_versions()
        snapshot_versions = []
        for entry in versions_info:
            vid = entry["version_id"]
            eco = entry.get("tier_clocks", {}).get("ecology", {})
            year = eco.get("simulated_year", 0) if isinstance(eco, dict) else 0
            if year > 0:  # skip version 0 (no ecology yet)
                snapshot_versions.append((vid, year))
    else:
        versions_info = world.list_versions()
        year_map = {}
        for entry in versions_info:
            eco = entry.get("tier_clocks", {}).get("ecology", {})
            year_map[entry["version_id"]] = eco.get("simulated_year", 0) if isinstance(eco, dict) else 0
        snapshot_versions = [(v, year_map.get(v, 0)) for v in snapshot_versions]

    print(f"Sampling {len(snapshot_versions)} snapshots along the ride path...")
    snapshots: dict[str, dict] = {}

    for vid, year in snapshot_versions:
        label = f"Year {year:.0f}"
        print(f"  Sampling {label} (version {vid})...")
        exp = sample_ride_experience(world, path, version=vid)
        snapshots[label] = exp
        n_visible = len(exp.get("species", {}))
        print(f"    {n_visible} species visible")

    # Plot comparison
    comp_path = output_dir / "ride_comparison.png"
    plot_snapshot_comparison(snapshots, comp_path)
    print(f"\nComparison graph: {comp_path}")

    # Save data
    comp_data_path = output_dir / "ride_comparison.json"
    Path(comp_data_path).write_text(json.dumps(
        {label: {"species": exp["species"], "num_species": len(exp["species"])}
         for label, exp in snapshots.items()},
        indent=2,
    ))
    print(f"Comparison data: {comp_data_path}")

    world.close()

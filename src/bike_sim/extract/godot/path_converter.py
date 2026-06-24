"""Convert A* cell path to Bezier control points for the Godot renderer.

Pipeline:
    1. Cell coordinates (row, col) → 3D Godot world coordinates
    2. Douglas-Peucker simplification on XZ plane
    3. Catmull-Rom tangent fitting → Bezier control points

Also provides bilinear heightmap sampling at arbitrary world positions.
"""

from __future__ import annotations

import numpy as np

CELL_SIZE: float = 50.0
GRID_SIZE: int = 1000


def sample_heightmap_bilinear(
    heightmap: np.ndarray,
    godot_x: float,
    godot_z: float,
    cell_size: float = CELL_SIZE,
) -> float:
    """Sample the heightmap at an arbitrary Godot world position using bilinear interpolation.

    Coordinate mapping:
        godot_x → sim col direction (0 to 50000m)
        godot_z → sim -row direction (0 to -50000m)
    """
    rows, cols = heightmap.shape

    # Convert Godot coords to continuous cell indices (cell centers at 0.5, 1.5, ...)
    col_f = godot_x / cell_size - 0.5
    row_f = -godot_z / cell_size - 0.5

    # Clamp to valid range
    col_f = np.clip(col_f, 0.0, cols - 1.001)
    row_f = np.clip(row_f, 0.0, rows - 1.001)

    c0 = int(np.floor(col_f))
    r0 = int(np.floor(row_f))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)

    tc = col_f - c0
    tr = row_f - r0

    h00 = float(heightmap[r0, c0])
    h01 = float(heightmap[r0, c1])
    h10 = float(heightmap[r1, c0])
    h11 = float(heightmap[r1, c1])

    return (
        h00 * (1 - tr) * (1 - tc)
        + h01 * (1 - tr) * tc
        + h10 * tr * (1 - tc)
        + h11 * tr * tc
    )


def cell_path_to_world_3d(
    path: list[tuple[int, int]],
    heightmap: np.ndarray,
    cell_size: float = CELL_SIZE,
) -> np.ndarray:
    """Convert (row, col) cell path to Godot 3D coordinates.

    Returns array of shape (N, 3) with [x, y, z] per point.

    Coordinate mapping:
        col → Godot X (col * cell_size + cell_size/2)
        heightmap[row, col] → Godot Y
        row → Godot -Z (-(row * cell_size + cell_size/2))
    """
    points = np.zeros((len(path), 3), dtype=np.float64)
    half = cell_size / 2.0
    for i, (r, c) in enumerate(path):
        points[i, 0] = c * cell_size + half  # X
        points[i, 1] = float(heightmap[r, c])  # Y (elevation)
        points[i, 2] = -(r * cell_size + half)  # Z (negative = forward)
    return points


def _douglas_peucker_xz(
    points: np.ndarray, tolerance: float
) -> list[int]:
    """Douglas-Peucker simplification on the XZ plane.

    Returns indices of retained points.
    """
    if len(points) <= 2:
        return list(range(len(points)))

    # Find the point farthest from the line between first and last (in XZ)
    start = points[0, [0, 2]]
    end = points[-1, [0, 2]]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        # All points at same XZ — keep first and last
        return [0, len(points) - 1]

    line_dir = line_vec / line_len

    # Perpendicular distance from each point to the line
    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(points) - 1):
        pt = points[i, [0, 2]]
        v = pt - start
        proj = np.dot(v, line_dir)
        perp = v - proj * line_dir
        dist = np.linalg.norm(perp)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > tolerance:
        left = _douglas_peucker_xz(points[: max_idx + 1], tolerance)
        right = _douglas_peucker_xz(points[max_idx:], tolerance)
        # Combine, avoiding duplicate at max_idx
        return left + [idx + max_idx for idx in right[1:]]
    else:
        return [0, len(points) - 1]


def simplify_path(
    points_3d: np.ndarray,
    tolerance: float = 50.0,
) -> np.ndarray:
    """Simplify a 3D path using Douglas-Peucker on the XZ plane.

    Y values are preserved from the original points at retained indices.

    Args:
        points_3d: array of shape (N, 3).
        tolerance: maximum perpendicular deviation in meters.

    Returns:
        Simplified array of shape (M, 3) where M <= N.
    """
    indices = _douglas_peucker_xz(points_3d, tolerance)
    return points_3d[indices]


def fit_bezier_control_points(
    points: np.ndarray,
    handle_fraction: float = 0.25,
) -> list[dict]:
    """Fit Catmull-Rom tangents to produce Bezier control points.

    Args:
        points: array of shape (N, 3) — simplified 3D path.
        handle_fraction: fraction of segment length for handle magnitude.

    Returns:
        List of dicts with position, handle_in, handle_out (Godot convention).
    """
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points for Bezier fitting")

    control_points: list[dict] = []

    for i in range(n):
        pos = points[i].tolist()

        if i == 0:
            # First point: handle_out toward next point
            seg = points[1] - points[0]
            handle_out = (seg * handle_fraction).tolist()
            handle_in = [0.0, 0.0, 0.0]
        elif i == n - 1:
            # Last point: handle_in from previous point
            seg = points[-1] - points[-2]
            handle_in = (-seg * handle_fraction).tolist()
            handle_out = [0.0, 0.0, 0.0]
        else:
            # Interior: Catmull-Rom tangent from neighbors
            tangent = points[i + 1] - points[i - 1]
            seg_len = np.linalg.norm(points[i + 1] - points[i])
            prev_len = np.linalg.norm(points[i] - points[i - 1])
            avg_len = (seg_len + prev_len) * 0.5

            tan_len = np.linalg.norm(tangent)
            if tan_len > 1e-10:
                tangent = tangent / tan_len * avg_len * handle_fraction
            else:
                tangent = np.zeros(3)

            handle_in = (-tangent).tolist()
            handle_out = tangent.tolist()

        control_points.append(
            {"position": pos, "handle_in": handle_in, "handle_out": handle_out}
        )

    return control_points


def path_to_bezier(
    path: list[tuple[int, int]],
    heightmap: np.ndarray,
    cell_size: float = CELL_SIZE,
    simplify_tolerance: float = 100.0,
    handle_fraction: float = 0.25,
    subsample_step: int = 3,
) -> list[dict]:
    """Full pipeline: cell path → Bezier control points.

    Args:
        path: list of (row, col) cells from A* pathfinder.
        heightmap: elevation array.
        cell_size: meters per cell.
        simplify_tolerance: Douglas-Peucker tolerance in meters.
        handle_fraction: Bezier handle length as fraction of segment.
        subsample_step: take every Nth cell before simplification.

    Returns:
        List of Bezier control point dicts for BakedCurve.
    """
    # Convert to 3D world coords
    points_3d = cell_path_to_world_3d(path, heightmap, cell_size)

    # Subsample to reduce point count before simplification
    if subsample_step > 1 and len(points_3d) > subsample_step * 2:
        indices = list(range(0, len(points_3d), subsample_step))
        if indices[-1] != len(points_3d) - 1:
            indices.append(len(points_3d) - 1)
        points_3d = points_3d[indices]

    # Simplify
    simplified = simplify_path(points_3d, tolerance=simplify_tolerance)

    # Ensure minimum point count
    if len(simplified) < 2:
        simplified = points_3d[:2] if len(points_3d) >= 2 else points_3d

    # Fit Bezier
    return fit_bezier_control_points(simplified, handle_fraction=handle_fraction)

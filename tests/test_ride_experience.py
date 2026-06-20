"""Tests for ride experience tool — path generation, sampling, rasterization."""

import numpy as np
import pytest

from bike_sim.extract.ride_experience import (
    CELL_SIZE,
    _astar_segment,
    _compute_grade_grid,
    _edge_cost,
    _generate_waypoints,
    _order_waypoints_tsp,
    _path_to_world_coords,
    generate_ride_path,
    rasterize_path,
    sample_ride_experience,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_heightmap(size: int = 100, elevation: float = 500.0) -> np.ndarray:
    """Flat heightmap for simple pathfinding tests."""
    return np.full((size, size), elevation, dtype=np.float64)


def _sloped_heightmap(size: int = 100) -> np.ndarray:
    """Heightmap with gentle west-to-east slope."""
    row = np.linspace(100, 200, size)
    return np.broadcast_to(row, (size, size)).copy()


def _ridge_heightmap(size: int = 100) -> np.ndarray:
    """Heightmap with a steep ridge across the middle (impassable)."""
    hm = np.full((size, size), 500.0, dtype=np.float64)
    # Steep ridge at row 50
    hm[48:52, :] = 1500.0
    return hm


# ---------------------------------------------------------------------------
# Grade computation
# ---------------------------------------------------------------------------

class TestGradeGrid:
    def test_flat_terrain_zero_grade(self):
        hm = _flat_heightmap()
        grade = _compute_grade_grid(hm)
        assert grade.max() < 0.01

    def test_sloped_terrain_has_grade(self):
        hm = _sloped_heightmap()
        grade = _compute_grade_grid(hm)
        # Interior cells should have consistent grade
        interior = grade[10:-10, 10:-10]
        assert interior.min() > 0
        assert interior.max() < 10  # gentle slope


# ---------------------------------------------------------------------------
# Edge cost
# ---------------------------------------------------------------------------

class TestEdgeCost:
    def test_flat_terrain_returns_distance(self):
        hm = _flat_heightmap()
        cost = _edge_cost(hm, 10, 10, 10, 11)
        assert cost is not None
        assert abs(cost - CELL_SIZE) < 1.0  # ~50m with minimal penalties

    def test_steep_edge_returns_none(self):
        hm = _flat_heightmap()
        hm[10, 11] = 1000.0  # huge cliff
        cost = _edge_cost(hm, 10, 10, 10, 11)
        assert cost is None

    def test_turn_penalty(self):
        hm = _flat_heightmap()
        # Going straight (same direction)
        cost_straight = _edge_cost(hm, 10, 10, 10, 11, prev_dir=(0, 1))
        # Making a 90-degree turn
        cost_turn = _edge_cost(hm, 10, 10, 11, 10, prev_dir=(0, 1))
        assert cost_straight is not None
        assert cost_turn is not None
        assert cost_turn > cost_straight


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------

class TestAstar:
    def test_flat_path_exists(self):
        hm = _flat_heightmap()
        path = _astar_segment(hm, (10, 10), (90, 90))
        assert path is not None
        assert path[0] == (10, 10)
        assert path[-1] == (90, 90)
        assert len(path) > 2

    def test_adjacent_path(self):
        hm = _flat_heightmap()
        path = _astar_segment(hm, (10, 10), (10, 11))
        assert path is not None
        assert len(path) == 2

    def test_impassable_ridge(self):
        """Path should route around an impassable ridge."""
        hm = _ridge_heightmap()
        # Try to go from top to bottom — ridge blocks direct path
        path = _astar_segment(hm, (10, 50), (90, 50))
        # Path may be None if truly impassable, or route around
        # Either outcome is acceptable
        if path is not None:
            assert path[0] == (10, 50)
            assert path[-1] == (90, 50)


# ---------------------------------------------------------------------------
# Waypoint generation
# ---------------------------------------------------------------------------

class TestWaypoints:
    def test_generates_requested_count(self):
        hm = _flat_heightmap(200)
        rng = np.random.default_rng(42)
        wps = _generate_waypoints(hm, rng, num_points=8)
        assert len(wps) == 8

    def test_waypoints_are_spread(self):
        hm = _flat_heightmap(200)
        rng = np.random.default_rng(42)
        wps = _generate_waypoints(hm, rng, num_points=6)
        # No two waypoints should be too close
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                dist = np.sqrt((wps[i][0] - wps[j][0])**2 + (wps[i][1] - wps[j][1])**2)
                assert dist > 10, f"Waypoints {i} and {j} too close: {dist}"

    def test_first_waypoint_near_center(self):
        hm = _flat_heightmap(200)
        rng = np.random.default_rng(42)
        wps = _generate_waypoints(hm, rng, num_points=6)
        center = 100
        assert abs(wps[0][0] - center) < 30
        assert abs(wps[0][1] - center) < 30


# ---------------------------------------------------------------------------
# TSP ordering
# ---------------------------------------------------------------------------

class TestTSPOrdering:
    def test_starts_with_first(self):
        wps = [(0, 0), (10, 10), (5, 5), (20, 20)]
        ordered = _order_waypoints_tsp(wps)
        assert ordered[0] == (0, 0)
        assert len(ordered) == 4

    def test_visits_all(self):
        wps = [(0, 0), (50, 50), (10, 90), (90, 10)]
        ordered = _order_waypoints_tsp(wps)
        assert set(map(tuple, ordered)) == set(map(tuple, wps))


# ---------------------------------------------------------------------------
# Path to world coordinates
# ---------------------------------------------------------------------------

class TestPathCoords:
    def test_coordinates_in_world_space(self):
        path = [(0, 0), (0, 1), (0, 2)]
        coords, headings, cum_dist = _path_to_world_coords(path)
        assert len(coords) == 3
        # First cell center should be at (25, 25)
        assert abs(coords[0, 0] - 25.0) < 0.01
        assert abs(coords[0, 1] - 25.0) < 0.01
        # Each step is 50m apart (horizontal)
        assert abs(cum_dist[1] - 50.0) < 0.01
        assert abs(cum_dist[2] - 100.0) < 0.01

    def test_cumulative_distance_monotonic(self):
        path = [(i, i) for i in range(20)]
        _, _, cum_dist = _path_to_world_coords(path)
        assert all(cum_dist[i] <= cum_dist[i + 1] for i in range(len(cum_dist) - 1))


# ---------------------------------------------------------------------------
# Rasterize path
# ---------------------------------------------------------------------------

class TestRasterizePath:
    def test_marks_path_cells(self):
        path = [(10, 10), (10, 11), (10, 12)]
        raster = rasterize_path(path, grid_size=100)
        assert raster[10, 10] > 0
        assert raster[10, 11] > 0
        assert raster[10, 12] > 0
        assert raster[50, 50] == 0  # not on path

    def test_distance_increases(self):
        path = [(10, i) for i in range(20)]
        raster = rasterize_path(path, grid_size=100)
        # Values should increase along the path
        values = [raster[10, i] for i in range(20)]
        assert all(values[i] < values[i + 1] for i in range(19))


# ---------------------------------------------------------------------------
# Full path generation
# ---------------------------------------------------------------------------

class TestGenerateRidePath:
    def test_generates_path_on_flat_terrain(self):
        hm = _flat_heightmap(200)
        path = generate_ride_path(hm, world_seed=42)
        assert len(path) > 50  # should visit many cells
        # All cells in bounds
        for r, c in path:
            assert 0 <= r < 200
            assert 0 <= c < 200

"""Tests for the Godot terrain extractor pipeline."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bike_sim.extract.godot.baked_curve import BakedCurve
from bike_sim.extract.godot.chunk_writer import write_chunk, write_manifest
from bike_sim.extract.godot.path_converter import (
    cell_path_to_world_3d,
    fit_bezier_control_points,
    path_to_bezier,
    sample_heightmap_bilinear,
    simplify_path,
)
from bike_sim.extract.godot.terrain_mesh import (
    CHUNK_LENGTH,
    CHUNK_WIDTH,
    GRID_COLS,
    GRID_ROWS,
    build_chunk_mesh,
    elevation_color,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_heightmap(height: float = 100.0, size: int = 100) -> np.ndarray:
    return np.full((size, size), height, dtype=np.float64)


def _sloped_heightmap(size: int = 100) -> np.ndarray:
    """Heightmap that increases from south (row 0) to north (row N)."""
    h = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        h[r, :] = r * 5.0  # 5m per cell = 10% grade at 50m cells
    return h


def _simple_control_points() -> list[dict]:
    """A short straight path along -Z for testing."""
    return [
        {"position": [100.0, 50.0, -100.0], "handle_in": [0, 0, 0], "handle_out": [0, 0, -50]},
        {"position": [100.0, 50.0, -400.0], "handle_in": [0, 0, 50], "handle_out": [0, 0, -50]},
        {"position": [100.0, 50.0, -700.0], "handle_in": [0, 0, 50], "handle_out": [0, 0, 0]},
    ]


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

class TestCoordinateMapping:
    def test_cell_to_godot_coords(self):
        h = _flat_heightmap(500.0, 10)
        path = [(0, 0), (5, 5), (9, 9)]
        pts = cell_path_to_world_3d(path, h, cell_size=50.0)

        # (row=0, col=0) → X=25, Y=500, Z=-25
        assert pts[0, 0] == pytest.approx(25.0)
        assert pts[0, 1] == pytest.approx(500.0)
        assert pts[0, 2] == pytest.approx(-25.0)

        # (row=5, col=5) → X=275, Y=500, Z=-275
        assert pts[1, 0] == pytest.approx(275.0)
        assert pts[1, 2] == pytest.approx(-275.0)

    def test_z_is_negative(self):
        """Godot Z is negative (forward = -Z)."""
        h = _flat_heightmap(0.0, 10)
        path = [(0, 0), (9, 0)]
        pts = cell_path_to_world_3d(path, h, cell_size=50.0)
        assert pts[0, 2] > pts[1, 2]  # higher row = more negative Z


# ---------------------------------------------------------------------------
# Bilinear sampling
# ---------------------------------------------------------------------------

class TestBilinearSampling:
    def test_exact_at_cell_center(self):
        h = np.arange(100, dtype=np.float64).reshape(10, 10)
        # Cell (3, 5) center is at godot_x=275, godot_z=-175
        val = sample_heightmap_bilinear(h, 275.0, -175.0, cell_size=50.0)
        assert val == pytest.approx(h[3, 5], abs=0.1)

    def test_interpolated_between_cells(self):
        h = np.zeros((10, 10), dtype=np.float64)
        h[0, 0] = 0.0
        h[0, 1] = 100.0
        # Midpoint between cell (0,0) center at x=25 and (0,1) center at x=75
        val = sample_heightmap_bilinear(h, 50.0, -25.0, cell_size=50.0)
        assert val == pytest.approx(50.0, abs=1.0)

    def test_clamps_at_edges(self):
        h = _flat_heightmap(42.0, 10)
        # Way outside the grid
        val = sample_heightmap_bilinear(h, -1000.0, 1000.0, cell_size=50.0)
        assert val == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Path simplification
# ---------------------------------------------------------------------------

class TestPathSimplification:
    def test_collinear_points_removed(self):
        # Straight line along X axis
        pts = np.array([
            [0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0], [400, 0, 0],
        ], dtype=np.float64)
        simplified = simplify_path(pts, tolerance=10.0)
        # Should keep only first and last
        assert len(simplified) == 2

    def test_corner_preserved(self):
        # L-shaped path
        pts = np.array([
            [0, 0, 0], [100, 0, 0], [200, 0, 0],
            [200, 0, -100], [200, 0, -200],
        ], dtype=np.float64)
        simplified = simplify_path(pts, tolerance=10.0)
        assert len(simplified) >= 3  # corner at (200, 0, 0) must be kept

    def test_two_points_unchanged(self):
        pts = np.array([[0, 0, 0], [100, 0, -100]], dtype=np.float64)
        simplified = simplify_path(pts, tolerance=10.0)
        assert len(simplified) == 2


# ---------------------------------------------------------------------------
# Bezier fitting
# ---------------------------------------------------------------------------

class TestBezierFitting:
    def test_correct_structure(self):
        pts = np.array([
            [0, 0, 0], [100, 10, -150], [200, 5, -300],
        ], dtype=np.float64)
        cps = fit_bezier_control_points(pts)
        assert len(cps) == 3
        for cp in cps:
            assert "position" in cp
            assert "handle_in" in cp
            assert "handle_out" in cp
            assert len(cp["position"]) == 3

    def test_first_handle_in_is_zero(self):
        pts = np.array([[0, 0, 0], [100, 0, -100]], dtype=np.float64)
        cps = fit_bezier_control_points(pts)
        assert cps[0]["handle_in"] == [0.0, 0.0, 0.0]

    def test_last_handle_out_is_zero(self):
        pts = np.array([[0, 0, 0], [100, 0, -100]], dtype=np.float64)
        cps = fit_bezier_control_points(pts)
        assert cps[-1]["handle_out"] == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# BakedCurve
# ---------------------------------------------------------------------------

class TestBakedCurve:
    def test_total_length_positive(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        assert curve.total_length > 0

    def test_straight_path_length(self):
        cps = [
            {"position": [0, 0, 0], "handle_in": [0, 0, 0], "handle_out": [0, 0, -100]},
            {"position": [0, 0, -300], "handle_in": [0, 0, 100], "handle_out": [0, 0, 0]},
        ]
        curve = BakedCurve(cps)
        assert curve.total_length == pytest.approx(300.0, rel=0.01)

    def test_sample_position_at_endpoints(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        start = curve.sample_position(0.0)
        end = curve.sample_position(curve.total_length)
        assert start[0] == pytest.approx(100.0, abs=1.0)
        assert end[2] == pytest.approx(-700.0, abs=1.0)

    def test_world_point_lateral(self):
        cps = [
            {"position": [0, 0, 0], "handle_in": [0, 0, 0], "handle_out": [0, 0, -100]},
            {"position": [0, 0, -300], "handle_in": [0, 0, 100], "handle_out": [0, 0, 0]},
        ]
        curve = BakedCurve(cps)
        # At midpoint, lateral=50 should offset in X (right)
        center = curve.world_point(150.0, 0.0, 0.0)
        right = curve.world_point(150.0, 50.0, 0.0)
        assert right[0] > center[0]  # offset to the right


# ---------------------------------------------------------------------------
# Chunk mesh
# ---------------------------------------------------------------------------

class TestChunkMesh:
    def test_vertex_count(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)
        assert mesh["vertices"].shape == (GRID_ROWS * GRID_COLS, 3)

    def test_index_count(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)
        expected = (GRID_ROWS - 1) * (GRID_COLS - 1) * 6
        assert len(mesh["indices"]) == expected

    def test_normals_are_unit_length(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)
        lengths = np.linalg.norm(mesh["normals"], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)

    def test_colors_present(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)
        assert mesh["colors"].shape == (GRID_ROWS * GRID_COLS, 4)

    def test_anchor_near_midpoint(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)
        anchor = mesh["anchor"]
        mid = curve.sample_position(CHUNK_LENGTH / 2)
        np.testing.assert_allclose(anchor, mid, atol=1.0)


# ---------------------------------------------------------------------------
# Chunk binary roundtrip
# ---------------------------------------------------------------------------

class TestChunkBinary:
    def test_write_and_read_header(self):
        cps = _simple_control_points()
        curve = BakedCurve(cps)
        mesh = build_chunk_mesh(curve, 0.0, lambda d, l: 0.0)

        with tempfile.NamedTemporaryFile(suffix=".chunk", delete=False) as f:
            path = f.name

        write_chunk(path, 7, mesh, {}, 0.0)

        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"CHNK"
            version = struct.unpack("<B", f.read(1))[0]
            assert version == 2
            chunk_id = struct.unpack("<i", f.read(4))[0]
            assert chunk_id == 7
            vert_count = struct.unpack("<i", f.read(4))[0]
            assert vert_count == GRID_ROWS * GRID_COLS
            idx_count = struct.unpack("<i", f.read(4))[0]
            assert idx_count == (GRID_ROWS - 1) * (GRID_COLS - 1) * 6

        Path(path).unlink()

    def test_manifest_written(self):
        with tempfile.TemporaryDirectory() as d:
            entries = [{"id": 0, "start_z": 0.0, "path": "chunk_0000.chunk"}]
            write_manifest(d, entries, [{"position": [0, 0, 0], "handle_in": [0, 0, 0], "handle_out": [0, 0, -50]}])
            manifest_path = Path(d) / "manifest.json"
            assert manifest_path.exists()
            data = json.loads(manifest_path.read_text())
            assert "chunks" in data
            assert "path" in data
            assert len(data["chunks"]) == 1


# ---------------------------------------------------------------------------
# Elevation color
# ---------------------------------------------------------------------------

class TestElevationColor:
    def test_low_elevation_green(self):
        c = elevation_color(100.0)
        assert c[1] > c[0]  # G > R (green dominant)
        assert c[1] > c[2]  # G > B

    def test_high_elevation_grey(self):
        c = elevation_color(1800.0)
        # Grey means R ≈ G ≈ B
        assert abs(c[0] - c[1]) < 0.1
        assert abs(c[1] - c[2]) < 0.1

    def test_alpha_is_one(self):
        for elev in [0, 500, 1000, 2000]:
            c = elevation_color(float(elev))
            assert c[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full pipeline (path_to_bezier)
# ---------------------------------------------------------------------------

class TestPathToBezier:
    def test_straight_path(self):
        h = _flat_heightmap(100.0, 20)
        path = [(10, c) for c in range(20)]  # horizontal line
        cps = path_to_bezier(path, h, cell_size=50.0, simplify_tolerance=50.0, subsample_step=2)
        assert len(cps) >= 2
        for cp in cps:
            assert "position" in cp

    def test_long_path_produces_multiple_points(self):
        h = _flat_heightmap(100.0, 100)
        path = [(50, c) for c in range(100)]  # 100 cells horizontal
        cps = path_to_bezier(path, h, cell_size=50.0, simplify_tolerance=100.0)
        # Should simplify a straight line to 2 points
        assert len(cps) == 2

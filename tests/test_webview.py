"""Tests for the Flask-based webview extractor (Phase 1 -- backend skeleton).

The webview app is a Flask application that serves world data through a
JSON API. It uses WorldQuery (Layer B) exclusively -- never touching
RasterStore or EventStore directly.

Factory function: ``create_app(world_dir) -> Flask``
Endpoints tested:
    GET /api/versions          -- list of version dicts
    GET /api/world/<v>/metadata -- structural metadata for a version
"""

from __future__ import annotations

import numpy as np
import pytest

from bike_sim.world import World

# World grid constants (must match WorldQuery expectations).
GRID_SIZE = 1000


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def world_dir(tmp_path):
    """Create a world with one committed version and return its path.

    Writes a heightmap raster and registers a species + individual so that
    the metadata endpoint has meaningful counts to report.
    """
    world_path = tmp_path / "test_world"
    world = World.create(world_path, seed=42)

    # Write a raster layer so the layer index is non-empty.
    heightmap = np.random.default_rng(42).random((GRID_SIZE, GRID_SIZE))
    world.rasters.set_version(0)
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)

    # Register a species and an individual so counts are > 0.
    world.events.add_species("oak", genome={"height": 25.0}, appeared_year=0.0)
    world.events.add_individual(
        "oak_001", "oak", x=1000.0, y=1000.0, appeared_year=0.0
    )

    world.commit_version(trigger="test setup")
    world.save(world_path / "world.json")
    world.close()

    return world_path


@pytest.fixture()
def client(world_dir):
    """Create a Flask test client backed by the test world."""
    from bike_sim.extract.webview.app import create_app

    app = create_app(world_dir)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════


class TestWebviewApp:
    """Flask webview extractor -- backend skeleton tests."""

    def test_app_creation(self, world_dir):
        """create_app returns a Flask application instance."""
        from flask import Flask

        from bike_sim.extract.webview.app import create_app

        app = create_app(world_dir)
        assert isinstance(app, Flask)

    def test_versions_endpoint(self, client):
        """GET /api/versions returns 200 with a JSON list of version dicts."""
        resp = client.get("/api/versions")

        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) >= 1

        # Each entry should look like a version dict with at least these keys.
        entry = data[0]
        assert "version_id" in entry
        assert "trigger" in entry
        assert "tier_clocks" in entry

    def test_metadata_endpoint(self, client):
        """GET /api/world/0/metadata returns 200 with expected metadata keys."""
        resp = client.get("/api/world/0/metadata")

        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

        # Required keys from WorldQuery.get_world_metadata.
        for key in ("extent", "cell_size", "grid_size", "layers", "simulated_year"):
            assert key in data, f"Missing key: {key}"

        # Structural checks on returned values.
        assert data["cell_size"] == 50.0
        assert data["grid_size"] == 1000

        extent = data["extent"]
        assert extent["x_min"] == 0
        assert extent["x_max"] == 50000

        # Layers should be grouped by tier.
        layers = data["layers"]
        assert isinstance(layers, dict)
        assert "geology" in layers
        assert "heightmap" in layers["geology"]

    def test_metadata_invalid_version(self, client):
        """GET /api/world/999/metadata returns 404 for a non-existent version."""
        resp = client.get("/api/world/999/metadata")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Tile Rendering
# ═══════════════════════════════════════════════════════════════════


class TestTileRendering:
    """Tile endpoint tests — renders raster layers as 256x256 PNG tiles."""

    def test_tile_endpoint_returns_png(self, client):
        """GET a valid tile returns 200 with content-type image/png."""
        resp = client.get("/api/world/0/tiles/geology/heightmap/0/0/0.png")

        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_tile_is_256x256(self, client):
        """Decoded PNG tile has dimensions 256x256."""
        import io

        from PIL import Image

        resp = client.get("/api/world/0/tiles/geology/heightmap/0/0/0.png")
        assert resp.status_code == 200

        img = Image.open(io.BytesIO(resp.data))
        assert img.size == (256, 256)

    def test_tile_cached_on_disk(self, client):
        """After requesting a tile, the PNG file exists in the cache directory."""
        import pathlib

        resp = client.get("/api/world/0/tiles/geology/heightmap/0/0/0.png")
        assert resp.status_code == 200

        cache_dir = pathlib.Path(client.application.config["TILE_CACHE_DIR"])
        cached_file = cache_dir / "0" / "geology" / "heightmap" / "0" / "0" / "0.png"
        assert cached_file.exists()
        assert cached_file.stat().st_size > 0

    def test_tile_invalid_zoom(self, client):
        """Zoom level above maximum (5) returns 404."""
        resp = client.get("/api/world/0/tiles/geology/heightmap/5/0/0.png")
        assert resp.status_code == 404

    def test_tile_out_of_bounds(self, client):
        """Tile coordinates outside valid range return 404."""
        # Zoom 0 means 1x1 grid, so x=1 is out of bounds.
        resp = client.get("/api/world/0/tiles/geology/heightmap/0/1/0.png")
        assert resp.status_code == 404

        # Zoom 1 means 2x2 grid, so x=2 or y=2 is out of bounds.
        resp = client.get("/api/world/0/tiles/geology/heightmap/1/2/0.png")
        assert resp.status_code == 404

        resp = client.get("/api/world/0/tiles/geology/heightmap/1/0/2.png")
        assert resp.status_code == 404

    def test_tile_invalid_layer(self, client):
        """Request for a nonexistent layer returns 404."""
        resp = client.get("/api/world/0/tiles/geology/nonexistent/0/0/0.png")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Leaflet Map
# ═══════════════════════════════════════════════════════════════════


class TestLeafletMap:
    """Leaflet slippy map served at GET / — a single HTML page."""

    def test_index_returns_html(self, client):
        """GET / returns 200 with text/html content type."""
        resp = client.get("/")

        assert resp.status_code == 200
        assert "text/html" in resp.content_type

    def test_index_contains_leaflet(self, client):
        """The index page references the Leaflet library."""
        resp = client.get("/")
        body = resp.data.decode("utf-8").lower()

        assert "leaflet" in body or "l.map" in body

    def test_index_contains_tile_url_pattern(self, client):
        """The map template references the tile API endpoint."""
        resp = client.get("/")
        body = resp.data.decode("utf-8")

        assert "/api/world/" in body, "Map page should reference /api/world/ tile path"
        assert "/tiles/" in body, "Map page should reference /tiles/ in the tile URL"

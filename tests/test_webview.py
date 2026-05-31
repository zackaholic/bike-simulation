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


# ═══════════════════════════════════════════════════════════════════
# Phase 4 — Layer Toggle
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def world_dir_multi_layer(tmp_path):
    """Create a world with raster layers across all three tiers.

    Extends the single-layer pattern with:
    - geology/heightmap (continuous)
    - geology/bedrock_type (categorical integers 0-5)
    - climate_hydrology/temperature (continuous)
    """
    world_path = tmp_path / "test_world_multi"
    world = World.create(world_path, seed=42)
    rng = np.random.default_rng(42)

    world.rasters.set_version(0)

    # Geology layers
    heightmap = rng.random((GRID_SIZE, GRID_SIZE))
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)

    bedrock = rng.integers(0, 6, size=(GRID_SIZE, GRID_SIZE)).astype(np.float64)
    world.rasters.write_layer("geology", "bedrock_type", bedrock, tick_number=0)

    # Climate-hydrology layer
    temperature = rng.uniform(-10.0, 35.0, size=(GRID_SIZE, GRID_SIZE))
    world.rasters.write_layer(
        "climate_hydrology", "temperature", temperature, tick_number=0
    )

    # Register a species so commit succeeds with valid ecology state.
    world.events.add_species("oak", genome={"height": 25.0}, appeared_year=0.0)
    world.events.add_individual(
        "oak_001", "oak", x=1000.0, y=1000.0, appeared_year=0.0
    )

    world.commit_version(trigger="multi-layer test setup")
    world.save(world_path / "world.json")
    world.close()

    return world_path


@pytest.fixture()
def multi_client(world_dir_multi_layer):
    """Create a Flask test client backed by the multi-layer test world."""
    from bike_sim.extract.webview.app import create_app

    app = create_app(world_dir_multi_layer)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestLayerToggle:
    """Phase 4 — layer switching: base layers, overlays, and layer control UI."""

    def test_metadata_includes_all_tiers(self, multi_client):
        """Metadata endpoint returns layers grouped by all three simulation tiers.

        Even if a tier has no layers, its key should be present so the
        frontend can build a complete layer control.
        """
        resp = multi_client.get("/api/world/0/metadata")

        assert resp.status_code == 200
        data = resp.get_json()

        layers = data["layers"]
        assert isinstance(layers, dict)

        # All three tiers must be present as top-level keys.
        for tier in ("geology", "climate_hydrology", "ecology"):
            assert tier in layers, f"Missing tier key: {tier}"
            assert isinstance(layers[tier], list)

        # Geology should have both layers we wrote.
        assert "heightmap" in layers["geology"]
        assert "bedrock_type" in layers["geology"]

        # Climate-hydrology should have temperature.
        assert "temperature" in layers["climate_hydrology"]

        # Ecology may be empty (no density rasters written), but key exists.
        assert isinstance(layers["ecology"], list)

    def test_tiles_for_different_layers(self, multi_client):
        """Tiles can be fetched for layers across different tiers.

        Verifies that the tile endpoint serves valid PNGs for geology/heightmap,
        geology/bedrock_type, and climate_hydrology/temperature.
        """
        import io

        from PIL import Image

        layer_paths = [
            "geology/heightmap",
            "geology/bedrock_type",
            "climate_hydrology/temperature",
        ]

        for layer_path in layer_paths:
            url = f"/api/world/0/tiles/{layer_path}/0/0/0.png"
            resp = multi_client.get(url)

            assert resp.status_code == 200, f"Expected 200 for {layer_path}, got {resp.status_code}"
            assert resp.content_type == "image/png"

            img = Image.open(io.BytesIO(resp.data))
            assert img.size == (256, 256), f"Tile for {layer_path} should be 256x256"

    def test_index_contains_layer_control(self, multi_client):
        """GET / includes Leaflet layer control for switching between layers.

        The index page should contain evidence of L.control.layers or the
        metadata-driven layer switching mechanism.
        """
        resp = multi_client.get("/")

        assert resp.status_code == 200
        body = resp.data.decode("utf-8")

        # The page should contain layer control setup — either the Leaflet
        # built-in control or a custom layer switcher referencing the metadata.
        has_layer_control = (
            "L.control.layers" in body
            or "control.layers" in body.lower()
            or "/api/world/" in body and "metadata" in body
        )
        assert has_layer_control, (
            "Index page should contain L.control.layers or a metadata-driven "
            "layer switcher"
        )

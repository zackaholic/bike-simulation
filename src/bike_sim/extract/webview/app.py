"""Flask application for the webview world inspector.

A Layer C extractor that consumes WorldQuery (Layer B) exclusively.
Serves world data as JSON over a REST API; later phases add tile
rendering and an interactive Leaflet frontend.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template

from bike_sim.extract.webview.tiles import MAX_ZOOM, render_tile, tile_valid
from bike_sim.query.world_query import WorldQuery
from bike_sim.world import World

WORLD_EXTENT = 50_000.0


def create_app(world_dir: str | Path) -> Flask:
    """Create a Flask app serving data for the world at *world_dir*."""
    world_dir = Path(world_dir)
    app = Flask(__name__)

    # Tile cache lives alongside the world directory.
    cache_dir = world_dir / ".tile_cache"
    app.config["TILE_CACHE_DIR"] = str(cache_dir)

    world = World.open(world_dir)
    query = WorldQuery(world)

    @app.teardown_appcontext
    def _close_world(exc: BaseException | None) -> None:  # noqa: ARG001
        pass  # World stays open for the lifetime of the process.

    # ── Frontend ──────────────────────────────────────────────────

    @app.route("/")
    def index():
        current_version = world.current_version
        if current_version < 0:
            current_version = 0
        return render_template(
            "index.html",
            world_extent=int(WORLD_EXTENT),
            max_zoom=MAX_ZOOM,
            current_version=current_version,
        )

    # ── API routes ────────────────────────────────────────────────

    @app.route("/api/versions")
    def api_versions():
        return jsonify(world.list_versions())

    @app.route("/api/world/<int:version>/metadata")
    def api_metadata(version: int):
        try:
            meta = query.get_world_metadata(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404
        return jsonify(meta)

    # ── Tile routes ───────────────────────────────────────────────

    @app.route("/api/world/<int:version>/tiles/<tier>/<layer>/<int:z>/<int:x>/<int:y>.png")
    def api_tile(version: int, tier: str, layer: str, z: int, x: int, y: int):
        # Validate version
        try:
            world.get_version(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404

        # Validate zoom and coordinates
        if not tile_valid(z, x, y):
            return jsonify({"error": "Invalid tile coordinates"}), 404

        # Validate layer exists
        available = query.available_layers(tier)
        if layer not in available:
            return jsonify({"error": f"Layer {tier}/{layer} not found"}), 404

        png_bytes = render_tile(query, version, tier, layer, z, x, y, cache_dir)
        return Response(png_bytes, mimetype="image/png")

    return app

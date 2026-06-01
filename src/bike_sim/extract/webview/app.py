"""Flask application for the webview world inspector.

A Layer C extractor that consumes WorldQuery (Layer B) exclusively.
Serves world data as JSON over a REST API; later phases add tile
rendering and an interactive Leaflet frontend.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

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

    # ── Point inspection ─────────────────────────────────────────

    @app.route("/api/world/<int:version>/point")
    def api_point(version: int):
        try:
            world.get_version(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404

        x_str = request.args.get("x")
        y_str = request.args.get("y")
        if x_str is None or y_str is None:
            return jsonify({"error": "Missing required parameters: x, y"}), 400

        try:
            x = float(x_str)
            y = float(y_str)
        except ValueError:
            return jsonify({"error": "x and y must be numbers"}), 400

        return jsonify(query.query_point(version, x, y))

    # ── Individual routes ──────────────────────────────────────────

    @app.route("/api/world/<int:version>/individuals")
    def api_individuals(version: int):
        try:
            world.get_version(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404

        x_min = float(request.args.get("x_min", 0))
        y_min = float(request.args.get("y_min", 0))
        x_max = float(request.args.get("x_max", WORLD_EXTENT))
        y_max = float(request.args.get("y_max", WORLD_EXTENT))

        return jsonify(query.query_individuals_in_bbox(version, x_min, y_min, x_max, y_max))

    @app.route("/api/world/<int:version>/individual/<individual_id>")
    def api_individual_detail(version: int, individual_id: str):
        try:
            world.get_version(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404

        try:
            detail = query.get_individual_detail(version, individual_id)
        except KeyError:
            return jsonify({"error": f"Individual {individual_id} not found"}), 404

        return jsonify(detail)

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

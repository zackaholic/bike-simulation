"""Flask application for the webview world inspector.

A Layer C extractor that consumes WorldQuery (Layer B) exclusively.
Serves world data as JSON over a REST API; later phases add tile
rendering and an interactive Leaflet frontend.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from bike_sim.extract.webview.tiles import MAX_ZOOM, get_layer_info, render_tile, tile_valid
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

    # ── Layer info (for legends) ────────────────────────────────

    @app.route("/api/world/<int:version>/layer_info/<tier>/<layer>")
    def api_layer_info(version: int, tier: str, layer: str):
        try:
            world.get_version(version)
        except KeyError:
            return jsonify({"error": f"Version {version} not found"}), 404

        available = query.available_layers(tier)
        if layer not in available:
            return jsonify({"error": f"Layer {tier}/{layer} not found"}), 404

        return jsonify(get_layer_info(query, version, tier, layer))

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

    # ── Timeline routes ────────────────────────────────────────────

    @app.route("/api/timeline/species")
    def api_timeline_species():
        ancestor = request.args.get("ancestor")
        return jsonify(query.get_species_timeline(ancestor=ancestor))

    @app.route("/api/timeline/weather")
    def api_timeline_weather():
        return jsonify(query.get_weather_timeline())

    @app.route("/api/timeline/diversity")
    def api_timeline_diversity():
        return jsonify(query.get_diversity_timeline())

    @app.route("/api/timeline/speciation")
    def api_timeline_speciation():
        return jsonify(query.get_speciation_timeline())

    @app.route("/api/timeline/disturbance")
    def api_timeline_disturbance():
        return jsonify(query.get_disturbance_timeline())

    @app.route("/api/timeline/snapshots")
    def api_timeline_snapshots():
        versions = world.list_versions()
        snapshots = []
        for entry in versions:
            eco_clock = entry.get("tier_clocks", {}).get("ecology", {})
            year = eco_clock.get("simulated_year", 0.0) if isinstance(eco_clock, dict) else 0.0
            snapshots.append({
                "version_id": entry["version_id"],
                "year": year,
            })
        return jsonify(snapshots)

    # ── Ride experience ────────────────────────────────────────────

    @app.route("/api/ride/experience")
    def api_ride_experience():
        """Return ride experience data if a ride path exists."""
        ride_output = world_dir / "ride_output" / "ride_experience.json"
        if not ride_output.exists():
            return jsonify({"error": "No ride experience data. Run ride-experience first."}), 404
        import json
        return Response(ride_output.read_text(), mimetype="application/json")

    @app.route("/api/ride/path")
    def api_ride_path():
        """Return ride path as list of [row, col] pairs."""
        path_file = world_dir / "ride_output" / "ride_path.json"
        if not path_file.exists():
            return jsonify({"error": "No ride path. Run ride-experience first."}), 404
        import json
        return Response(path_file.read_text(), mimetype="application/json")

    return app

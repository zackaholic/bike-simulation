"""Run the webview dev server.

Usage::

    python -m bike_sim.extract.webview <world_dir> [--port 5000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bike_sim.extract.webview.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Webview world inspector")
    parser.add_argument("world_dir", type=Path, help="Path to a world directory")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    app = create_app(args.world_dir)
    print(f"Serving world at {args.world_dir} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

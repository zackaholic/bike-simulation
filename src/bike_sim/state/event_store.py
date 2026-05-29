"""SQLite-backed storage for species, distinguished individuals, and events.

This store handles the structured, non-raster data that the simulation produces:

  - **Species**: genome (trait vector as dict), lineage, history.
  - **Distinguished individuals**: specific plants the cyclist might notice,
    with sub-cell-precision positions and life histories.
  - **Events**: things that happened at a place and time (fire, flood, etc.),
    with optional JSON payloads for extra detail.

SQLite is a good fit here because this data is append-mostly, needs spatial
and temporal queries, and benefits from ACID guarantees when the simulation
writes in batches after each tick.

Spatial queries use simple bounding-box filtering in SQL. At our scale
(thousands of individuals, not millions), this is plenty fast. R-tree
indices can be added later if profiling shows a need.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


class EventStore:
    """SQLite-backed store for species, individuals, and events."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def create(cls, path: Path) -> EventStore:
        """Create a new EventStore database at the given path."""
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        store = cls(conn)
        store._create_tables()
        return store

    @classmethod
    def open(cls, path: Path) -> EventStore:
        """Open an existing EventStore database."""
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return cls(conn)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Species ──────────────────────────────────────────────────────

    def add_species(
        self,
        species_id: str,
        genome: dict,
        parent_id: str | None = None,
        appeared_year: float = 0.0,
    ) -> None:
        """Insert a new species with its genome trait vector."""
        self._conn.execute(
            """INSERT INTO species (species_id, genome_json, parent_id, appeared_year)
               VALUES (?, ?, ?, ?)""",
            (species_id, json.dumps(genome), parent_id, appeared_year),
        )
        self._conn.commit()

    def get_species(self, species_id: str) -> dict:
        """Look up a species by ID."""
        row = self._conn.execute(
            "SELECT * FROM species WHERE species_id = ?", (species_id,)
        ).fetchone()
        return {
            "species_id": row["species_id"],
            "genome": json.loads(row["genome_json"]),
            "parent_id": row["parent_id"],
            "appeared_year": row["appeared_year"],
        }

    def list_species(self) -> list[dict]:
        """Return all species as dicts with at least a 'species_id' key."""
        rows = self._conn.execute("SELECT * FROM species").fetchall()
        return [
            {
                "species_id": row["species_id"],
                "parent_id": row["parent_id"],
                "appeared_year": row["appeared_year"],
            }
            for row in rows
        ]

    # ── Distinguished individuals ────────────────────────────────────

    def add_individual(
        self,
        individual_id: str,
        species_id: str,
        x: float,
        y: float,
        appeared_year: float = 0.0,
    ) -> None:
        """Insert a distinguished individual at a specific position."""
        self._conn.execute(
            """INSERT INTO individuals (individual_id, species_id, x, y, appeared_year)
               VALUES (?, ?, ?, ?, ?)""",
            (individual_id, species_id, x, y, appeared_year),
        )
        self._conn.commit()

    def get_individual(self, individual_id: str) -> dict:
        """Look up an individual by ID."""
        row = self._conn.execute(
            "SELECT * FROM individuals WHERE individual_id = ?", (individual_id,)
        ).fetchone()
        return {
            "individual_id": row["individual_id"],
            "species_id": row["species_id"],
            "x": row["x"],
            "y": row["y"],
            "appeared_year": row["appeared_year"],
        }

    def find_individuals_near(self, x: float, y: float, radius: float) -> list[dict]:
        """Find all individuals within a given radius of a point.

        Uses a bounding-box pre-filter followed by exact distance check.
        """
        rows = self._conn.execute(
            """SELECT * FROM individuals
               WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?""",
            (x - radius, x + radius, y - radius, y + radius),
        ).fetchall()
        # Exact circular distance check
        results = []
        for row in rows:
            dx = row["x"] - x
            dy = row["y"] - y
            if math.sqrt(dx * dx + dy * dy) <= radius:
                results.append(
                    {
                        "individual_id": row["individual_id"],
                        "species_id": row["species_id"],
                        "x": row["x"],
                        "y": row["y"],
                        "appeared_year": row["appeared_year"],
                    }
                )
        return results

    # ── Events ───────────────────────────────────────────────────────

    def add_event(
        self,
        event_type: str,
        x: float,
        y: float,
        year: float,
        radius: float = 0.0,
        data: dict | None = None,
    ) -> None:
        """Record an event at a location and time, with optional JSON payload."""
        self._conn.execute(
            """INSERT INTO events (event_type, x, y, year, radius, data_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_type, x, y, year, radius, json.dumps(data) if data else None),
        )
        self._conn.commit()

    def get_events_in_region(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> list[dict]:
        """Find all events whose position falls within a bounding box."""
        rows = self._conn.execute(
            """SELECT * FROM events
               WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?""",
            (x_min, x_max, y_min, y_max),
        ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def get_events_in_time_range(self, year_start: float, year_end: float) -> list[dict]:
        """Find all events within a year range (inclusive)."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE year BETWEEN ? AND ?",
            (year_start, year_end),
        ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    # ── Internal ─────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS species (
                species_id   TEXT PRIMARY KEY,
                genome_json  TEXT NOT NULL,
                parent_id    TEXT,
                appeared_year REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS individuals (
                individual_id TEXT PRIMARY KEY,
                species_id   TEXT NOT NULL,
                x            REAL NOT NULL,
                y            REAL NOT NULL,
                appeared_year REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_individuals_xy
                ON individuals (x, y);

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                x           REAL NOT NULL,
                y           REAL NOT NULL,
                year        REAL NOT NULL,
                radius      REAL DEFAULT 0.0,
                data_json   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_xy
                ON events (x, y);
            CREATE INDEX IF NOT EXISTS idx_events_year
                ON events (year);
        """)
        self._conn.commit()

    @staticmethod
    def _event_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "event_type": row["event_type"],
            "x": row["x"],
            "y": row["y"],
            "year": row["year"],
            "radius": row["radius"],
            "data": json.loads(row["data_json"]) if row["data_json"] else None,
        }

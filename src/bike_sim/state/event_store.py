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
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = cls(conn)
        store._create_tables()
        store._migrate()
        return store

    @classmethod
    def open(cls, path: Path) -> EventStore:
        """Open an existing EventStore database."""
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = cls(conn)
        store._migrate()
        return store

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
        """Look up a species by ID.

        Raises KeyError if the species does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM species WHERE species_id = ?", (species_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Species {species_id!r} not found")
        return {
            "species_id": row["species_id"],
            "genome": json.loads(row["genome_json"]),
            "parent_id": row["parent_id"],
            "appeared_year": row["appeared_year"],
            "extinct_year": row["extinct_year"],
        }

    def mark_species_extinct(self, species_id: str, extinct_year: float) -> None:
        """Mark a species as extinct at the given year."""
        self._conn.execute(
            "UPDATE species SET extinct_year = ? WHERE species_id = ?",
            (extinct_year, species_id),
        )
        self._conn.commit()

    def list_species(self, alive_at_year: float | None = None) -> list[dict]:
        """Return all species as dicts with at least a 'species_id' key.

        If alive_at_year is given, filter to species where:
        appeared_year <= alive_at_year AND (extinct_year IS NULL OR extinct_year > alive_at_year)
        """
        if alive_at_year is not None:
            rows = self._conn.execute(
                """SELECT * FROM species
                   WHERE appeared_year <= ?
                     AND (extinct_year IS NULL OR extinct_year > ?)""",
                (alive_at_year, alive_at_year),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM species").fetchall()
        return [
            {
                "species_id": row["species_id"],
                "parent_id": row["parent_id"],
                "appeared_year": row["appeared_year"],
                "extinct_year": row["extinct_year"],
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
        """Look up an individual by ID.

        Raises KeyError if the individual does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM individuals WHERE individual_id = ?", (individual_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Individual {individual_id!r} not found")
        return {
            "individual_id": row["individual_id"],
            "species_id": row["species_id"],
            "x": row["x"],
            "y": row["y"],
            "appeared_year": row["appeared_year"],
            "died_year": row["died_year"],
            "state": row["state"],
        }

    def kill_individual(self, individual_id: str, died_year: float) -> None:
        """Mark an individual as dead at the given year."""
        self._conn.execute(
            "UPDATE individuals SET died_year = ? WHERE individual_id = ?",
            (died_year, individual_id),
        )
        self._conn.commit()

    def update_individual_state(self, individual_id: str, state: str) -> None:
        """Update an individual's lifecycle state (alive/snag/log/mound/removed)."""
        self._conn.execute(
            "UPDATE individuals SET state = ? WHERE individual_id = ?",
            (state, individual_id),
        )
        self._conn.commit()

    def find_individuals_near(
        self, x: float, y: float, radius: float, alive_at_year: float | None = None
    ) -> list[dict]:
        """Find all individuals within a given radius of a point.

        Uses a bounding-box pre-filter followed by exact distance check.

        If alive_at_year is given, filter to individuals where:
        appeared_year <= alive_at_year AND (died_year IS NULL OR died_year > alive_at_year)
        """
        if alive_at_year is not None:
            rows = self._conn.execute(
                """SELECT * FROM individuals
                   WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?
                     AND appeared_year <= ?
                     AND (died_year IS NULL OR died_year > ?)""",
                (x - radius, x + radius, y - radius, y + radius, alive_at_year, alive_at_year),
            ).fetchall()
        else:
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
                        "died_year": row["died_year"],
                        "state": row["state"],
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

    # ── Tick summaries ────────────────────────────────────────────────

    def write_tick_summary(
        self,
        tick: int,
        year: float,
        season: int,
        species_summaries: list[dict],
    ) -> None:
        """Batch-insert per-species summary rows for a single tick.

        Each dict in *species_summaries* must have keys:
        species_id, total_density, occupied_cells, mean_biomass_age.
        """
        rows = [
            (tick, year, season, s["species_id"], s["total_density"],
             s["occupied_cells"], s["mean_biomass_age"])
            for s in species_summaries
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO tick_summary
               (tick, year, season, species_id, total_density, occupied_cells, mean_biomass_age)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def write_tick_weather(
        self,
        tick: int,
        year: float,
        season: int,
        mean_temp: float,
        mean_precip: float,
        mean_drought: float,
    ) -> None:
        """Insert weather summary for a single tick."""
        self._conn.execute(
            """INSERT OR REPLACE INTO tick_weather
               (tick, year, season, mean_temp, mean_precip, mean_drought)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tick, year, season, mean_temp, mean_precip, mean_drought),
        )
        self._conn.commit()

    def get_tick_summaries(
        self,
        species_id: str | None = None,
        tick_start: int | None = None,
        tick_end: int | None = None,
    ) -> list[dict]:
        """Query tick summaries with optional species and tick-range filters."""
        clauses: list[str] = []
        params: list = []
        if species_id is not None:
            clauses.append("species_id = ?")
            params.append(species_id)
        if tick_start is not None:
            clauses.append("tick >= ?")
            params.append(tick_start)
        if tick_end is not None:
            clauses.append("tick <= ?")
            params.append(tick_end)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM tick_summary{where} ORDER BY tick, species_id", params
        ).fetchall()
        return [
            {
                "tick": row["tick"],
                "year": row["year"],
                "season": row["season"],
                "species_id": row["species_id"],
                "total_density": row["total_density"],
                "occupied_cells": row["occupied_cells"],
                "mean_biomass_age": row["mean_biomass_age"],
            }
            for row in rows
        ]

    def get_tick_weather(
        self,
        tick_start: int | None = None,
        tick_end: int | None = None,
    ) -> list[dict]:
        """Query tick weather summaries with optional tick-range filter."""
        clauses: list[str] = []
        params: list = []
        if tick_start is not None:
            clauses.append("tick >= ?")
            params.append(tick_start)
        if tick_end is not None:
            clauses.append("tick <= ?")
            params.append(tick_end)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM tick_weather{where} ORDER BY tick", params
        ).fetchall()
        return [
            {
                "tick": row["tick"],
                "year": row["year"],
                "season": row["season"],
                "mean_temp": row["mean_temp"],
                "mean_precip": row["mean_precip"],
                "mean_drought": row["mean_drought"],
            }
            for row in rows
        ]

    # ── Internal ─────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS species (
                species_id   TEXT PRIMARY KEY,
                genome_json  TEXT NOT NULL,
                parent_id    TEXT,
                appeared_year REAL DEFAULT 0.0,
                extinct_year REAL
            );

            CREATE TABLE IF NOT EXISTS individuals (
                individual_id TEXT PRIMARY KEY,
                species_id   TEXT NOT NULL,
                x            REAL NOT NULL,
                y            REAL NOT NULL,
                appeared_year REAL DEFAULT 0.0,
                died_year    REAL,
                state        TEXT DEFAULT 'alive'
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

            CREATE TABLE IF NOT EXISTS tick_summary (
                tick        INTEGER NOT NULL,
                year        REAL NOT NULL,
                season      INTEGER NOT NULL,
                species_id  TEXT NOT NULL,
                total_density REAL NOT NULL,
                occupied_cells INTEGER NOT NULL,
                mean_biomass_age REAL NOT NULL,
                PRIMARY KEY (tick, species_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tick_summary_species
                ON tick_summary (species_id);

            CREATE TABLE IF NOT EXISTS tick_weather (
                tick        INTEGER NOT NULL PRIMARY KEY,
                year        REAL NOT NULL,
                season      INTEGER NOT NULL,
                mean_temp   REAL NOT NULL,
                mean_precip REAL NOT NULL,
                mean_drought REAL NOT NULL
            );
        """)
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns/tables that may be missing from older databases."""
        for stmt in [
            "ALTER TABLE individuals ADD COLUMN died_year REAL",
            "ALTER TABLE species ADD COLUMN extinct_year REAL",
            "ALTER TABLE individuals ADD COLUMN state TEXT DEFAULT 'alive'",
        ]:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Ensure tick_summary and tick_weather tables exist in older databases.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tick_summary (
                tick        INTEGER NOT NULL,
                year        REAL NOT NULL,
                season      INTEGER NOT NULL,
                species_id  TEXT NOT NULL,
                total_density REAL NOT NULL,
                occupied_cells INTEGER NOT NULL,
                mean_biomass_age REAL NOT NULL,
                PRIMARY KEY (tick, species_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tick_summary_species
                ON tick_summary (species_id);

            CREATE TABLE IF NOT EXISTS tick_weather (
                tick        INTEGER NOT NULL PRIMARY KEY,
                year        REAL NOT NULL,
                season      INTEGER NOT NULL,
                mean_temp   REAL NOT NULL,
                mean_precip REAL NOT NULL,
                mean_drought REAL NOT NULL
            );
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

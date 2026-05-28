"""Tests for the SQLite-backed EventStore."""

import pytest

from bike_sim.state.event_store import EventStore


@pytest.fixture()
def store(tmp_path):
    """Create a fresh EventStore for each test."""
    db_path = tmp_path / "events.db"
    s = EventStore.create(db_path)
    yield s
    s.close()


# --- Species ---


def test_add_and_get_species(store):
    """Add a species with a genome dict, retrieve it, verify all fields."""
    genome = {"height": 12.5, "drought_tolerance": 0.8, "growth_rate": 0.03}
    store.add_species("pine_01", genome, parent_id=None, appeared_year=500.0)

    result = store.get_species("pine_01")

    assert result["species_id"] == "pine_01"
    assert result["genome"] == genome
    assert result["parent_id"] is None
    assert result["appeared_year"] == 500.0


def test_species_parent_lineage(store):
    """A child species stores its parent_id correctly."""
    store.add_species("grass_ancestor", {"height": 0.3}, appeared_year=0.0)
    store.add_species(
        "grass_variant",
        {"height": 0.4, "seed_mass": 0.01},
        parent_id="grass_ancestor",
        appeared_year=1000.0,
    )

    child = store.get_species("grass_variant")
    assert child["parent_id"] == "grass_ancestor"
    assert child["appeared_year"] == 1000.0


def test_list_species(store):
    """list_species returns all added species IDs."""
    store.add_species("sp_a", {"trait": 1.0})
    store.add_species("sp_b", {"trait": 2.0})
    store.add_species("sp_c", {"trait": 3.0})

    ids = store.list_species()
    assert set(ids) == {"sp_a", "sp_b", "sp_c"}


# --- Distinguished individuals ---


def test_add_and_get_individual(store):
    """Add an individual, retrieve it, verify position and species link."""
    store.add_species("oak_01", {"height": 20.0})
    store.add_individual("mother_oak", "oak_01", x=100.5, y=200.3, appeared_year=800.0)

    ind = store.get_individual("mother_oak")

    assert ind["individual_id"] == "mother_oak"
    assert ind["species_id"] == "oak_01"
    assert ind["x"] == pytest.approx(100.5)
    assert ind["y"] == pytest.approx(200.3)
    assert ind["appeared_year"] == 800.0


def test_find_individuals_near(store):
    """find_individuals_near returns only individuals within the given radius."""
    store.add_species("birch", {"height": 15.0})

    # Place individuals at known positions around (50, 50).
    store.add_individual("close_1", "birch", x=50.0, y=51.0)  # dist = 1.0
    store.add_individual("close_2", "birch", x=49.0, y=50.0)  # dist = 1.0
    store.add_individual("far_1", "birch", x=60.0, y=50.0)  # dist = 10.0
    store.add_individual("far_2", "birch", x=50.0, y=70.0)  # dist = 20.0

    nearby = store.find_individuals_near(50.0, 50.0, radius=5.0)
    nearby_ids = {ind["individual_id"] for ind in nearby}

    assert nearby_ids == {"close_1", "close_2"}
    assert "far_1" not in nearby_ids
    assert "far_2" not in nearby_ids


# --- Events ---


def test_events_spatial_query(store):
    """get_events_in_region returns only events within the bounding box."""
    store.add_event("fire", x=10.0, y=10.0, year=100.0)
    store.add_event("fire", x=50.0, y=50.0, year=100.0)
    store.add_event("flood", x=90.0, y=90.0, year=100.0)

    # Query a region that includes only the first event.
    events = store.get_events_in_region(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0)
    assert len(events) == 1
    assert events[0]["event_type"] == "fire"
    assert events[0]["x"] == pytest.approx(10.0)
    assert events[0]["y"] == pytest.approx(10.0)

    # Query a region that includes two events.
    events = store.get_events_in_region(x_min=0.0, y_min=0.0, x_max=55.0, y_max=55.0)
    assert len(events) == 2


def test_events_time_range_query(store):
    """get_events_in_time_range returns only events within the year range."""
    store.add_event("eruption", x=0.0, y=0.0, year=100.0)
    store.add_event("fire", x=0.0, y=0.0, year=500.0)
    store.add_event("drought", x=0.0, y=0.0, year=1000.0)

    events = store.get_events_in_time_range(year_start=400.0, year_end=600.0)
    assert len(events) == 1
    assert events[0]["event_type"] == "fire"

    # Inclusive boundaries: events exactly at the boundary should be returned.
    events = store.get_events_in_time_range(year_start=100.0, year_end=1000.0)
    assert len(events) == 3


def test_event_data_round_trip(store):
    """The optional data dict round-trips correctly through JSON serialization."""
    payload = {
        "intensity": 0.85,
        "affected_species": ["pine_01", "oak_01"],
        "nested": {"depth": 3, "note": "deep burn"},
    }
    store.add_event("fire", x=25.0, y=25.0, year=300.0, radius=5.0, data=payload)

    events = store.get_events_in_region(x_min=20.0, y_min=20.0, x_max=30.0, y_max=30.0)
    assert len(events) == 1
    event = events[0]

    assert event["data"] == payload
    assert event["data"]["intensity"] == pytest.approx(0.85)
    assert event["data"]["affected_species"] == ["pine_01", "oak_01"]
    assert event["data"]["nested"]["depth"] == 3
    assert event["radius"] == pytest.approx(5.0)


# --- Lifecycle ---


def test_open_existing_database(tmp_path):
    """EventStore.open loads an existing database created by EventStore.create."""
    db_path = tmp_path / "persist.db"
    s = EventStore.create(db_path)
    s.add_species("fern", {"frond_count": 12.0}, appeared_year=0.0)
    s.close()

    s2 = EventStore.open(db_path)
    result = s2.get_species("fern")
    assert result["species_id"] == "fern"
    assert result["genome"] == {"frond_count": 12.0}
    s2.close()

"""Tests for WeatherSystem — deterministic per-season weather from overlapping sinusoidal cycles."""

import math

import numpy as np
import pytest

from bike_sim.weather import WeatherSystem, WeatherCycle, SeasonalWeather


# ---------- Fixtures ----------


@pytest.fixture
def heightmap():
    """Simple gradient heightmap for testing."""
    h = np.zeros((100, 100), dtype=np.float64)
    # Elevation increases with y (north) and x (east)
    y = np.linspace(0, 1000, 100).reshape(-1, 1)
    x = np.linspace(0, 500, 100).reshape(1, -1)
    return h + y + x * 0.3


@pytest.fixture
def system(heightmap):
    """WeatherSystem with seed 42."""
    return WeatherSystem(world_seed=42, geology_heightmap=heightmap, grid_size=100, cell_size=50.0)


@pytest.fixture
def system_alt(heightmap):
    """WeatherSystem with a different seed."""
    return WeatherSystem(world_seed=999, geology_heightmap=heightmap, grid_size=100, cell_size=50.0)


# ---------- 1. Determinism from seed ----------


class TestDeterminism:
    def test_same_seed_same_weather(self, heightmap):
        """Same seed + same year/season must produce identical arrays."""
        sys_a = WeatherSystem(world_seed=42, geology_heightmap=heightmap, grid_size=100, cell_size=50.0)
        sys_b = WeatherSystem(world_seed=42, geology_heightmap=heightmap, grid_size=100, cell_size=50.0)

        wa = sys_a.generate(year=100.0, season=2)
        wb = sys_b.generate(year=100.0, season=2)

        np.testing.assert_array_equal(wa.temperature, wb.temperature)
        np.testing.assert_array_equal(wa.precipitation, wb.precipitation)
        np.testing.assert_array_equal(wa.frost_severity, wb.frost_severity)
        assert wa.storm_intensity == wb.storm_intensity

    def test_different_seeds_different_weather(self, system, system_alt):
        """Different seeds must produce different weather."""
        wa = system.generate(year=100.0, season=2)
        wb = system_alt.generate(year=100.0, season=2)

        # At least one of temperature or precipitation should differ
        assert not np.array_equal(wa.temperature, wb.temperature) or \
               not np.array_equal(wa.precipitation, wb.precipitation)

    def test_repeated_calls_same_result(self, system):
        """Calling generate twice with same args returns identical results."""
        wa = system.generate(year=50.0, season=0)
        wb = system.generate(year=50.0, season=0)
        np.testing.assert_array_equal(wa.temperature, wb.temperature)
        np.testing.assert_array_equal(wa.precipitation, wb.precipitation)


# ---------- 2. Cycle generation ----------


class TestCycleGeneration:
    def test_cycle_count(self, system):
        """Should generate approximately 9 cycles."""
        cycles = system.get_cycles()
        assert isinstance(cycles, list)
        assert 5 <= len(cycles) <= 15, f"Expected ~9 cycles, got {len(cycles)}"

    def test_positive_periods(self, system):
        """All cycles must have positive periods."""
        for cycle in system.get_cycles():
            assert cycle.period > 0, f"Cycle has non-positive period: {cycle.period}"

    def test_phases_in_range(self, system):
        """All phases must be in [0, 2*pi]."""
        for cycle in system.get_cycles():
            assert 0 <= cycle.phase <= 2 * math.pi, \
                f"Phase {cycle.phase} outside [0, 2*pi]"

    def test_valid_targets(self, system):
        """Cycle targets must be 'temperature', 'precipitation', or 'both'."""
        valid_targets = {"temperature", "precipitation", "both"}
        for cycle in system.get_cycles():
            assert cycle.target in valid_targets, \
                f"Invalid target: {cycle.target}"

    def test_compound_cycle_correlation(self, system):
        """Compound ('both') cycles must have correlation in [-1, 1]."""
        for cycle in system.get_cycles():
            if cycle.target == "both":
                assert -1 <= cycle.correlation <= 1, \
                    f"Correlation {cycle.correlation} outside [-1, 1]"

    def test_different_seeds_different_cycles(self, system, system_alt):
        """Different seeds should produce different cycle parameters."""
        ca = system.get_cycles()
        cb = system_alt.get_cycles()

        # Compare periods — at least some should differ
        periods_a = [c.period for c in ca]
        periods_b = [c.period for c in cb]
        assert periods_a != periods_b


# ---------- 3. Cycle evaluation / superposition ----------


class TestCycleEvaluation:
    def test_temperature_anomaly_varies(self, system):
        """Temperature anomaly should not be constant over time."""
        temps = []
        for year in [0, 10, 50, 200]:
            w = system.generate(year=float(year), season=2)
            temps.append(w.temperature.mean())
        # Not all the same
        assert len(set(round(t, 6) for t in temps)) > 1, \
            "Temperature anomaly appears constant across years"

    def test_precipitation_multiplier_positive(self, system):
        """Precipitation must always be positive (exp of log anomaly)."""
        for year in [0, 50, 100, 500]:
            for season in range(4):
                w = system.generate(year=float(year), season=season)
                assert np.all(w.precipitation >= 0), \
                    f"Negative precipitation at year={year}, season={season}"

    def test_anomalies_nonzero_at_time_zero(self, system):
        """At year 0, anomalies should generally be nonzero (phases don't all cancel)."""
        w = system.generate(year=0.0, season=0)
        # Temperature should not be exactly the baseline everywhere
        # (with multiple cycles and random phases this is almost certain)
        assert w.temperature.std() > 0 or w.temperature.mean() != 0


# ---------- 4. Seasonal modulation ----------


class TestSeasonalModulation:
    def test_summer_warmer_than_winter(self, system):
        """Summer (season=2) should generally be warmer than winter (season=0)."""
        summer = system.generate(year=100.0, season=2)
        winter = system.generate(year=100.0, season=0)
        assert summer.temperature.mean() > winter.temperature.mean(), \
            f"Summer mean {summer.temperature.mean():.1f} not warmer than winter {winter.temperature.mean():.1f}"

    def test_summer_drier_than_winter(self, system):
        """Summer should generally have lower precipitation than winter (seasonal scalars)."""
        summer = system.generate(year=100.0, season=2)
        winter = system.generate(year=100.0, season=0)
        assert summer.precipitation.mean() < winter.precipitation.mean(), \
            f"Summer precip {summer.precipitation.mean():.1f} not lower than winter {winter.precipitation.mean():.1f}"

    def test_all_four_seasons_produce_output(self, system):
        """All four seasons (0-3) should produce valid SeasonalWeather."""
        for season in range(4):
            w = system.generate(year=100.0, season=season)
            assert isinstance(w, SeasonalWeather)
            assert w.season == season
            assert w.temperature.shape == (100, 100)


# ---------- 5. Spatial variation ----------


class TestSpatialVariation:
    def test_temperature_decreases_with_elevation(self, system):
        """Higher elevations should have lower temperatures (lapse rate)."""
        w = system.generate(year=100.0, season=2)
        # Bottom row (low elevation) vs top row (high elevation)
        low_elev_temp = w.temperature[0, :].mean()
        high_elev_temp = w.temperature[-1, :].mean()
        assert high_elev_temp < low_elev_temp, \
            f"High elevation temp {high_elev_temp:.1f} not lower than low elevation {low_elev_temp:.1f}"

    def test_higher_cells_colder(self, system):
        """Individual high-elevation cells should be colder than low-elevation cells."""
        w = system.generate(year=100.0, season=2)
        # Corner comparison: (0,0) is lowest elevation, (99,99) is highest
        assert w.temperature[99, 99] < w.temperature[0, 0]

    def test_precipitation_not_uniform(self, system):
        """Precipitation should vary spatially (not a single value everywhere)."""
        w = system.generate(year=100.0, season=2)
        assert w.precipitation.std() > 0, "Precipitation is spatially uniform"


# ---------- 6. Frost model ----------


class TestFrostModel:
    def test_frost_non_negative(self, system):
        """Frost severity must be >= 0 everywhere."""
        for season in range(4):
            w = system.generate(year=100.0, season=season)
            assert np.all(w.frost_severity >= 0), \
                f"Negative frost severity in season {season}"

    def test_frost_higher_at_elevation(self, system):
        """Frost severity should be higher at higher (colder) elevations."""
        w = system.generate(year=100.0, season=0)  # winter
        low_frost = w.frost_severity[0, :].mean()
        high_frost = w.frost_severity[-1, :].mean()
        assert high_frost > low_frost, \
            f"High elevation frost {high_frost:.3f} not greater than low {low_frost:.3f}"

    def test_winter_more_frost_than_summer(self, system):
        """Winter/spring should have higher frost severity than summer."""
        winter = system.generate(year=100.0, season=0)
        summer = system.generate(year=100.0, season=2)
        assert winter.frost_severity.mean() > summer.frost_severity.mean(), \
            f"Winter frost {winter.frost_severity.mean():.3f} not greater than summer {summer.frost_severity.mean():.3f}"


# ---------- 7. Storm intensity ----------


class TestStormIntensity:
    def test_storm_non_negative(self, system):
        """Storm intensity must be >= 0."""
        for year in [0, 50, 200]:
            for season in range(4):
                w = system.generate(year=float(year), season=season)
                assert w.storm_intensity >= 0, \
                    f"Negative storm intensity at year={year}, season={season}"

    def test_storm_higher_with_high_precip(self, system):
        """Storm intensity should be higher when precipitation multiplier > 1."""
        # Collect storm intensities and mean precipitations across many time points
        storms_by_precip = []
        for year in range(0, 500, 10):
            for season in range(4):
                w = system.generate(year=float(year), season=season)
                storms_by_precip.append((w.precipitation.mean(), w.storm_intensity))

        # Split into wet and dry halves by precipitation
        storms_by_precip.sort(key=lambda x: x[0])
        mid = len(storms_by_precip) // 2
        dry_storms = [s for _, s in storms_by_precip[:mid]]
        wet_storms = [s for _, s in storms_by_precip[mid:]]

        assert np.mean(wet_storms) > np.mean(dry_storms), \
            "Wet periods don't have higher storm intensity than dry periods"


# ---------- 8. Output shape and types ----------


class TestOutputShapeAndTypes:
    def test_array_shapes(self, system):
        """All spatial arrays must have shape (grid_size, grid_size)."""
        w = system.generate(year=100.0, season=2)
        assert w.temperature.shape == (100, 100)
        assert w.precipitation.shape == (100, 100)
        assert w.frost_severity.shape == (100, 100)

    def test_array_dtypes(self, system):
        """All arrays should be float64."""
        w = system.generate(year=100.0, season=2)
        assert w.temperature.dtype == np.float64
        assert w.precipitation.dtype == np.float64
        assert w.frost_severity.dtype == np.float64

    def test_storm_intensity_is_scalar(self, system):
        """storm_intensity should be a float scalar, not an array."""
        w = system.generate(year=100.0, season=2)
        assert isinstance(w.storm_intensity, float)

    def test_season_field_matches_input(self, system):
        """The season field on output must match the input season."""
        for season in range(4):
            w = system.generate(year=100.0, season=season)
            assert w.season == season


# ---------- 9. Precipitation non-negativity ----------


class TestPrecipitationNonNegativity:
    def test_precipitation_never_negative(self, system):
        """Precipitation must be >= 0 even in extreme dry conditions."""
        for year in [0, 100, 250, 500, 1000]:
            for season in range(4):
                w = system.generate(year=float(year), season=season)
                assert np.all(w.precipitation >= 0), \
                    f"Negative precipitation at year={year}, season={season}: " \
                    f"min={w.precipitation.min():.4f}"

    def test_precipitation_extreme_years(self, system):
        """Check non-negativity across a wide range of years."""
        for year in np.linspace(0, 5000, 50):
            w = system.generate(year=float(year), season=0)
            assert np.all(w.precipitation >= 0), \
                f"Negative precipitation at year={year:.0f}"


# ---------- 10. Long-term cycle effects ----------


class TestLongTermCycleEffects:
    def test_weather_differs_at_distant_years(self, system):
        """Weather at year 0 vs year 500 should differ (long cycles shift)."""
        w0 = system.generate(year=0.0, season=2)
        w500 = system.generate(year=500.0, season=2)
        assert not np.allclose(w0.temperature, w500.temperature), \
            "Temperature identical at year 0 and year 500"
        assert not np.allclose(w0.precipitation, w500.precipitation), \
            "Precipitation identical at year 0 and year 500"

    def test_weather_differs_at_nearby_years(self, system):
        """Weather at year 0 vs year 3 should differ (short cycles shift)."""
        w0 = system.generate(year=0.0, season=2)
        w3 = system.generate(year=3.0, season=2)
        assert not np.allclose(w0.temperature, w3.temperature), \
            "Temperature identical at year 0 and year 3"

    def test_gradual_change(self, system):
        """Adjacent years should have bounded temperature differences (continuous, not lurching)."""
        diffs = []
        for yr in range(50, 60):
            w0 = system.generate(year=float(yr), season=2)
            w1 = system.generate(year=float(yr + 1), season=2)
            diffs.append(np.abs(w0.temperature - w1.temperature).mean())

        # Year-to-year change should be modest — no single year jumps more than 5°C mean
        assert max(diffs) < 5.0, \
            f"Year-to-year temperature jump too large: {max(diffs):.3f}°C"

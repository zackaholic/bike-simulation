"""Tests for per-season erosion (Step 4).

Tests cover erode_seasonal, thermal_diffusion, and SeasonalErosionParams.
"""

import numpy as np
import pytest

from bike_sim.tiers.erosion import (
    SeasonalErosionParams,
    erode_seasonal,
    thermal_diffusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def terrain():
    """Simple gradient terrain for erosion testing."""
    rng = np.random.default_rng(42)
    # Sloped terrain (higher in NW corner)
    y = np.linspace(500, 0, 100).reshape(-1, 1)
    x = np.linspace(300, 0, 100).reshape(1, -1)
    heightmap = (y + x + rng.uniform(0, 20, (100, 100))).astype(np.float64)
    sediment = rng.uniform(0, 3, (100, 100)).astype(np.float64)
    return heightmap, sediment


@pytest.fixture
def flow_and_precip():
    """Flow accumulation and precipitation for a simple terrain."""
    rng = np.random.default_rng(42)
    flow = rng.uniform(1, 500, (100, 100)).astype(np.float64)
    # Higher flow in valleys (lower elevation)
    y = np.linspace(0, 1, 100).reshape(-1, 1)
    flow *= 1 + y * 5  # more flow at bottom
    precip = np.full((100, 100), 800.0, dtype=np.float64)
    return flow, precip


@pytest.fixture
def bedrock():
    """Mixed bedrock types."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 6, (100, 100)).astype(np.int32)


# ---------------------------------------------------------------------------
# 1. erode_seasonal basics
# ---------------------------------------------------------------------------


class TestErodeSeasonalBasics:
    def test_erode_seasonal_reduces_total_elevation(
        self, terrain, flow_and_precip, bedrock
    ):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip
        surface_before = (heightmap + sediment).sum()

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock,
        )

        surface_after = (heightmap + sediment).sum()
        assert surface_after < surface_before

    def test_erode_seasonal_modifies_in_place(
        self, terrain, flow_and_precip, bedrock
    ):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip
        hm_id = id(heightmap)
        sed_id = id(sediment)

        result = erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock,
        )

        assert result is None
        assert id(heightmap) == hm_id
        assert id(sediment) == sed_id

    def test_sediment_non_negative(self, terrain, flow_and_precip, bedrock):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=1.0,
            bedrock_type=bedrock,
        )

        assert np.all(sediment >= 0)

    def test_heightmap_non_negative(self, terrain, flow_and_precip, bedrock):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=1.0,
            bedrock_type=bedrock,
        )

        assert np.all(heightmap >= 0)


# ---------------------------------------------------------------------------
# 2. Storm intensity effects
# ---------------------------------------------------------------------------


class TestStormIntensity:
    def test_higher_storm_more_erosion(self, flow_and_precip, bedrock):
        rng = np.random.default_rng(99)
        flow, precip = flow_and_precip

        # Create two identical terrains
        base_h = np.linspace(500, 0, 100).reshape(-1, 1) + np.linspace(
            300, 0, 100
        ).reshape(1, -1)
        base_h = base_h + rng.uniform(0, 20, (100, 100))

        h_low = base_h.copy().astype(np.float64)
        s_low = rng.uniform(0, 3, (100, 100)).astype(np.float64)
        h_high = h_low.copy()
        s_high = s_low.copy()

        surface_before = (h_low + s_low).sum()

        erode_seasonal(
            h_low, s_low, flow, precip, storm_intensity=0.0, bedrock_type=bedrock
        )
        erode_seasonal(
            h_high, s_high, flow, precip, storm_intensity=2.0, bedrock_type=bedrock
        )

        change_low = surface_before - (h_low + s_low).sum()
        change_high = surface_before - (h_high + s_high).sum()
        # More intense storm removes more material
        assert change_high > change_low

    def test_zero_storm_still_erodes(self, terrain, flow_and_precip, bedrock):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip
        surface_before = (heightmap + sediment).sum()

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=0.0,
            bedrock_type=bedrock,
        )

        surface_after = (heightmap + sediment).sum()
        assert surface_after < surface_before


# ---------------------------------------------------------------------------
# 3. Flow accumulation effects
# ---------------------------------------------------------------------------


class TestFlowAccumulation:
    def test_higher_flow_more_erosion(self, bedrock):
        rng = np.random.default_rng(77)
        # Uniform sloped terrain
        h = np.full((100, 100), 200.0, dtype=np.float64)
        s = np.full((100, 100), 2.0, dtype=np.float64)
        precip = np.full((100, 100), 800.0, dtype=np.float64)

        # Low flow everywhere
        flow_low = np.full((100, 100), 5.0, dtype=np.float64)
        # High flow everywhere
        flow_high = np.full((100, 100), 500.0, dtype=np.float64)

        h_low = h.copy()
        s_low = s.copy()
        h_high = h.copy()
        s_high = s.copy()

        surface_before = (h + s).sum()

        erode_seasonal(
            h_low, s_low, flow_low, precip, storm_intensity=0.5, bedrock_type=bedrock
        )
        erode_seasonal(
            h_high,
            s_high,
            flow_high,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock,
        )

        change_low = surface_before - (h_low + s_low).sum()
        change_high = surface_before - (h_high + s_high).sum()
        assert change_high > change_low


# ---------------------------------------------------------------------------
# 4. Bedrock erodibility
# ---------------------------------------------------------------------------


class TestBedrockErodibility:
    def test_soft_bedrock_erodes_more(self, flow_and_precip):
        flow, precip = flow_and_precip
        rng = np.random.default_rng(55)

        h_base = (
            np.linspace(500, 0, 100).reshape(-1, 1)
            + np.linspace(300, 0, 100).reshape(1, -1)
            + rng.uniform(0, 20, (100, 100))
        ).astype(np.float64)

        # Shale (type 2, erodibility 1.2) vs granite (type 3, erodibility 0.3)
        bedrock_shale = np.full((100, 100), 2, dtype=np.int32)
        bedrock_granite = np.full((100, 100), 3, dtype=np.int32)

        h_shale = h_base.copy()
        s_shale = np.full((100, 100), 2.0, dtype=np.float64)
        h_granite = h_base.copy()
        s_granite = np.full((100, 100), 2.0, dtype=np.float64)

        surface_before = (h_base + 2.0).sum()

        erode_seasonal(
            h_shale,
            s_shale,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock_shale,
        )
        erode_seasonal(
            h_granite,
            s_granite,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock_granite,
        )

        change_shale = surface_before - (h_shale + s_shale).sum()
        change_granite = surface_before - (h_granite + s_granite).sum()
        assert change_shale > change_granite


# ---------------------------------------------------------------------------
# 5. Deposition
# ---------------------------------------------------------------------------


class TestDeposition:
    def test_deposition_occurs(self, terrain, flow_and_precip, bedrock):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip
        sediment_before = sediment.copy()

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock,
        )

        # Some cells should have gained sediment (deposition)
        gained = sediment > sediment_before
        assert gained.any(), "Expected some cells to gain sediment via deposition"


# ---------------------------------------------------------------------------
# 6. Calibration (approximate)
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_erosion_magnitude_reasonable(self, terrain, flow_and_precip, bedrock):
        heightmap, sediment = terrain
        flow, precip = flow_and_precip
        surface_before = heightmap + sediment

        erode_seasonal(
            heightmap,
            sediment,
            flow,
            precip,
            storm_intensity=0.5,
            bedrock_type=bedrock,
        )

        surface_after = heightmap + sediment
        mean_change = np.abs(surface_before - surface_after).mean()
        # A single tick should not erode more than ~10mm (0.01m) on average
        assert mean_change < 0.01, (
            f"Mean erosion {mean_change:.6f}m exceeds 10mm threshold"
        )


# ---------------------------------------------------------------------------
# 7. thermal_diffusion
# ---------------------------------------------------------------------------


class TestThermalDiffusion:
    def test_diffusion_smooths_sharp_features(self):
        # Create flat terrain with a sediment spike (diffusion acts on sediment)
        heightmap = np.full((100, 100), 100.0, dtype=np.float64)
        sediment = np.full((100, 100), 1.0, dtype=np.float64)
        sediment[50, 50] = 50.0  # spike in sediment

        for _ in range(1000):
            thermal_diffusion(heightmap, sediment, diffusion_rate=1e-3)

        # Spike should have flattened substantially
        assert sediment[50, 50] < 30.0, (
            f"Sediment spike at {sediment[50, 50]:.1f} did not flatten enough"
        )

    def test_diffusion_magnitude_tiny(self):
        rng = np.random.default_rng(42)
        heightmap = rng.uniform(100, 500, (100, 100)).astype(np.float64)
        sediment = rng.uniform(0, 3, (100, 100)).astype(np.float64)
        hm_before = heightmap.copy()

        thermal_diffusion(heightmap, sediment, diffusion_rate=1e-6)

        max_change = np.abs(heightmap - hm_before).max()
        assert max_change < 0.01, (
            f"Single diffusion step changed terrain by {max_change:.6f}m"
        )

    def test_diffusion_sediment_non_negative(self):
        rng = np.random.default_rng(42)
        heightmap = rng.uniform(100, 500, (100, 100)).astype(np.float64)
        # Very thin sediment to stress test
        sediment = rng.uniform(0, 0.01, (100, 100)).astype(np.float64)

        for _ in range(100):
            thermal_diffusion(heightmap, sediment, diffusion_rate=1e-4)

        assert np.all(sediment >= 0)


# ---------------------------------------------------------------------------
# 8. SeasonalErosionParams
# ---------------------------------------------------------------------------


class TestSeasonalErosionParams:
    def test_default_params_exist(self):
        params = SeasonalErosionParams()
        # Should have meaningful attributes
        assert hasattr(params, "erosion_scale")
        assert hasattr(params, "deposition_fraction")
        assert params.erosion_scale > 0
        assert 0 < params.deposition_fraction < 1

    def test_custom_params(self):
        params = SeasonalErosionParams(
            erosion_scale=2.0, deposition_fraction=0.8
        )
        assert params.erosion_scale == 2.0
        assert params.deposition_fraction == 0.8

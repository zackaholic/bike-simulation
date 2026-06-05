"""Tests for the particle-based erosion system.

Tests use small 100x100 synthetic heightmaps and reduced particle counts
for speed. The erosion module is pure numpy with no bike_sim dependencies.
"""

import numpy as np
import numpy.testing as npt
import pytest

from bike_sim.tiers.erosion import (
    BEDROCK_ERODIBILITY,
    ErosionParams,
    erode_hydraulic,
    erode_thermal,
)


# ---------- Fixtures ----------


@pytest.fixture
def small_params():
    """ErosionParams with reduced particles for fast testing."""
    return ErosionParams(num_particles=5_000, max_lifetime=30)


@pytest.fixture
def tilted_plane():
    """100x100 heightmap tilted linearly from 100m (left) to 0m (right)."""
    return np.linspace(100.0, 0.0, 100).reshape(1, 100).repeat(100, axis=0)


@pytest.fixture
def dome():
    """100x100 gentle dome heightmap peaking at centre."""
    y, x = np.mgrid[:100, :100]
    cx, cy = 50.0, 50.0
    return 50.0 - 0.01 * ((x - cx) ** 2 + (y - cy) ** 2)


# ---------- Tests ----------


def test_particle_on_slope(small_params, tilted_plane):
    """Erosion on a tilted plane removes more material where flow has built up.

    On a left-to-right slope, flow accumulates toward the right (downhill).
    Particles that have traveled further accumulate more water and erode more,
    so mid-slope regions (where particles have gathered speed and water) should
    show greater erosion than the very top of the slope. Overall the heightmap
    must be modified.
    """
    hm = tilted_plane.copy()
    bedrock = np.ones((100, 100), dtype=np.int32)  # uniform type 1
    precip = np.ones((100, 100), dtype=np.float64)
    flow = np.ones((100, 100), dtype=np.float64)
    sediment = np.zeros((100, 100), dtype=np.float64)
    rng = np.random.default_rng(42)

    eroded, sed_out = erode_hydraulic(
        hm, bedrock, precip, flow, sediment, rng, small_params
    )

    # Heightmap must have changed
    assert not np.array_equal(eroded, tilted_plane)

    # Compare erosion in upper quarter vs lower quarter (column-wise).
    # Upper quarter = columns 0-24 (high elevation, start of flow).
    # Lower quarter = columns 50-74 (mid-slope, accumulated flow).
    erosion = tilted_plane - eroded
    upper_loss = erosion[:, :25].sum()
    mid_loss = erosion[:, 50:75].sum()
    assert mid_loss > upper_loss, (
        f"Mid-slope erosion ({mid_loss:.3f}) should exceed upper-slope ({upper_loss:.3f})"
    )


def test_particle_deterministic(small_params, tilted_plane):
    """Two runs with identical RNG state produce bit-identical results."""
    bedrock = np.ones((100, 100), dtype=np.int32)
    precip = np.ones((100, 100), dtype=np.float64)
    flow = np.ones((100, 100), dtype=np.float64)
    sediment = np.zeros((100, 100), dtype=np.float64)

    eroded_a, sed_a = erode_hydraulic(
        tilted_plane.copy(), bedrock, precip, flow, sediment.copy(),
        np.random.default_rng(42), small_params,
    )
    eroded_b, sed_b = erode_hydraulic(
        tilted_plane.copy(), bedrock, precip, flow, sediment.copy(),
        np.random.default_rng(42), small_params,
    )

    npt.assert_array_equal(eroded_a, eroded_b)
    npt.assert_array_equal(sed_a, sed_b)


def test_erodibility_scaling(small_params, dome):
    """Soft bedrock (higher erodibility) erodes more than hard bedrock.

    Left half = type 2 (shale, erodibility 1.2).
    Right half = type 3 (granite, erodibility 0.3).
    """
    hm = dome.copy()
    bedrock = np.full((100, 100), 3, dtype=np.int32)  # hard everywhere
    bedrock[:, :50] = 2  # soft on the left half
    precip = np.ones((100, 100), dtype=np.float64)
    flow = np.ones((100, 100), dtype=np.float64)
    sediment = np.zeros((100, 100), dtype=np.float64)
    rng = np.random.default_rng(99)

    eroded, _ = erode_hydraulic(
        hm, bedrock, precip, flow, sediment, rng, small_params
    )

    erosion = dome - eroded
    soft_loss = erosion[:, :50].sum()
    hard_loss = erosion[:, 50:].sum()

    assert soft_loss > hard_loss, (
        f"Soft-rock erosion ({soft_loss:.3f}) should exceed hard-rock ({hard_loss:.3f})"
    )


def test_sediment_erodes_before_bedrock(small_params, tilted_plane):
    """When thick sediment is present, erosion removes sediment not bedrock.

    With 10m of sediment everywhere and a small number of particles, the
    sediment layer should absorb the bulk of erosive work.
    """
    hm_original = tilted_plane.copy()
    hm = tilted_plane.copy()
    bedrock = np.ones((100, 100), dtype=np.int32)
    precip = np.ones((100, 100), dtype=np.float64)
    flow = np.ones((100, 100), dtype=np.float64)
    sediment_before = np.full((100, 100), 10.0, dtype=np.float64)
    rng = np.random.default_rng(7)

    eroded, sed_after = erode_hydraulic(
        hm, bedrock, precip, flow, sediment_before.copy(), rng, small_params
    )

    sediment_removed = (sediment_before - sed_after).clip(min=0).sum()
    bedrock_removed = (hm_original - eroded).clip(min=0).sum()

    assert sediment_removed > bedrock_removed, (
        f"Sediment removal ({sediment_removed:.3f}) should far exceed "
        f"bedrock removal ({bedrock_removed:.3f})"
    )


def test_thermal_reduces_steep_slopes():
    """Thermal erosion smooths a sharp cliff, reducing maximum slope.

    A step function (left=100m, right=0m) should become smoother after
    thermal erosion: maximum height difference between adjacent cells
    decreases, and the transition zone widens.
    """
    hm = np.zeros((100, 100), dtype=np.float64)
    hm[:, :50] = 100.0  # sharp cliff at column 50
    sediment = np.zeros((100, 100), dtype=np.float64)

    # Maximum adjacent-cell height difference before
    diff_before = np.abs(np.diff(hm, axis=1)).max()

    erode_thermal(hm, sediment, ErosionParams())

    diff_after = np.abs(np.diff(hm, axis=1)).max()

    assert diff_after < diff_before, (
        f"Max slope should decrease: before={diff_before:.1f}, after={diff_after:.1f}"
    )

    # Transition zone should be wider: count columns with intermediate heights
    col_means = hm[50, :]  # sample one row
    transition = np.count_nonzero((col_means > 5.0) & (col_means < 95.0))
    assert transition > 1, "Cliff should spread to more than one column of transition"


def test_mass_conservation(small_params, dome):
    """Material removed from the heightmap >= sediment deposited.

    The difference accounts for sediment carried off the domain edge by
    particles. Both the eroded heightmap and sediment must be non-negative.
    """
    hm_original = dome.copy()
    hm = dome.copy()
    bedrock = np.ones((100, 100), dtype=np.int32)
    precip = np.ones((100, 100), dtype=np.float64)
    flow = np.ones((100, 100), dtype=np.float64)
    sediment = np.zeros((100, 100), dtype=np.float64)
    rng = np.random.default_rng(123)

    eroded, sed_out = erode_hydraulic(
        hm, bedrock, precip, flow, sediment, rng, small_params
    )

    total_removed = (hm_original - eroded).sum()
    total_deposited = sed_out.sum()

    assert total_removed >= 0, "Net erosion should be non-negative"
    assert total_deposited >= 0, "Total sediment should be non-negative"
    assert total_removed >= total_deposited, (
        f"Material removed ({total_removed:.3f}) must be >= sediment deposited "
        f"({total_deposited:.3f}); difference is edge-lost sediment"
    )

    # No negative values in outputs
    assert (eroded >= 0).all(), "Eroded heightmap has negative values"
    assert (sed_out >= 0).all(), "Sediment has negative values"

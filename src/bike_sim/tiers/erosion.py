"""Particle-based hydraulic erosion and thermal weathering.

This module implements the erosion sub-system used by the climate-hydrology
tier. It replaces the earlier grid-diffusion placeholder with two physically
motivated processes:

**Hydraulic erosion** drops virtual water particles onto the terrain surface.
Each particle flows downhill, picking up sediment where it moves fast over
steep terrain and depositing it where it slows down. Over many thousands of
particles this carves river channels, alluvial fans, and talus slopes — the
kind of features that make a landscape feel *shaped by water* rather than
stamped from noise.

**Thermal erosion** (talus creep) redistributes material on slopes that exceed
the angle of repose. This softens cliffs into scree slopes and fills narrow
gullies, complementing the channelised action of hydraulic erosion.

Both routines are pure numpy computation — no imports from bike_sim. The
caller (``ClimateHydrologyTier``) is responsible for reading/writing raster
layers and passing in a seeded RNG. The inner particle loop is written in a
numba-ready style (arrays and scalars only, no Python objects) so it can be
JIT-compiled in a future optimisation pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ErosionParams:
    """All tuneable knobs for the erosion sub-system."""

    # Hydraulic particle parameters
    num_particles: int = 70_000
    max_lifetime: int = 50
    inertia: float = 0.1
    capacity_factor: float = 2.0
    deposition_rate: float = 0.3
    erosion_rate: float = 0.1
    evaporation_rate: float = 0.03
    gravity: float = 4.0
    min_slope: float = 0.01
    erosion_radius: int = 3

    # Thermal erosion parameters
    thermal_passes: int = 3
    talus_slope: float = 0.6
    creep_rate: float = 0.15


BEDROCK_ERODIBILITY: dict[int, float] = {
    0: 0.8,   # moderate (metamorphic)
    1: 1.0,   # reference (sandstone)
    2: 1.2,   # soft (shale)
    3: 0.3,   # hard (granite)
    4: 0.5,   # moderately hard (limestone)
    5: 0.6,   # moderate (basalt)
}


# ------------------------------------------------------------------
# Gradient helper
# ------------------------------------------------------------------

@njit(cache=True)
def _compute_gradient_bilinear(
    heightmap: np.ndarray, sediment: np.ndarray, x: float, y: float,
) -> tuple[float, float]:
    """Bilinear-interpolated gradient at continuous position (*x*, *y*).

    Computes the gradient of the combined surface (heightmap + sediment)
    without allocating a full-sized temporary array — only the four corner
    values are summed.

    *x* is the column (horizontal) coordinate, *y* is the row (vertical)
    coordinate, matching numpy's ``[row, col]`` indexing.

    Returns ``(gx, gy)`` — the gradient in the x and y directions.  The
    gradient points *uphill* (from low to high); the caller negates to
    move downhill.

    JIT-compiled with ``@numba.njit``: only arrays and scalars.
    """
    rows, cols = heightmap.shape

    # Integer coords of the top-left cell of the 2×2 neighbourhood.
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, cols - 1)
    y1 = min(y0 + 1, rows - 1)

    # Fractional offsets within the cell.
    fx = x - x0
    fy = y - y0

    # Combined surface heights at four corners — no full-array allocation.
    h00 = heightmap[y0, x0] + sediment[y0, x0]
    h10 = heightmap[y0, x1] + sediment[y0, x1]
    h01 = heightmap[y1, x0] + sediment[y1, x0]
    h11 = heightmap[y1, x1] + sediment[y1, x1]

    # Gradient via partial derivatives of the bilinear interpolant.
    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
    gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

    return gx, gy


# ------------------------------------------------------------------
# Particle simulation (numba-ready inner loop)
# ------------------------------------------------------------------

@njit(cache=True)
def _simulate_particle(
    heightmap: np.ndarray,
    sediment: np.ndarray,
    erodibility: np.ndarray,
    pos_x: float,
    pos_y: float,
    inertia: float,
    capacity_factor: float,
    deposition_rate: float,
    erosion_rate: float,
    evaporation_rate: float,
    gravity: float,
    min_slope: float,
    max_lifetime: int,
    erosion_radius: int,
) -> None:
    """Simulate a single water droplet eroding and depositing sediment.

    Modifies *heightmap* and *sediment* **in-place**.

    All parameters are numpy arrays or scalars — no Python objects — and the
    function is wrapped with ``@numba.njit`` for a large speedup over the pure
    Python particle loop.
    """
    rows, cols = heightmap.shape

    dir_x = 0.0
    dir_y = 0.0
    speed = 1.0
    water = 1.0
    sed_load = 0.0

    for step in range(max_lifetime):
        # 1. Integer cell coords.
        ci = int(pos_x)
        cj = int(pos_y)

        # 2. Compute gradient on the combined surface (no full-array alloc).
        gx, gy = _compute_gradient_bilinear(heightmap, sediment, pos_x, pos_y)

        # 3. Update direction (inertia blend).
        dir_x = dir_x * inertia + gx * (1 - inertia)
        dir_y = dir_y * inertia + gy * (1 - inertia)

        # Normalise.
        length = (dir_x * dir_x + dir_y * dir_y) ** 0.5
        if length < 1e-10:
            # Zero gradient — pick a pseudo-random direction from position hash.
            hash_val = ((ci * 6364136223846793005 + cj * 1442695040888963407 + step) & 0xFFFFFFFF)
            angle = (hash_val / 0xFFFFFFFF) * 2.0 * 3.141592653589793
            dir_x = np.cos(angle)
            dir_y = np.sin(angle)
        else:
            dir_x /= length
            dir_y /= length

        # 4. Move.
        new_x = pos_x - dir_x  # negative because gradient points uphill
        new_y = pos_y - dir_y

        # 5. Kill if out of bounds (1-cell margin).
        if new_x < 1 or new_x >= cols - 2 or new_y < 1 or new_y >= rows - 2:
            break

        # 6. Height difference.
        old_ci = int(pos_x)
        old_cj = int(pos_y)
        old_h = heightmap[old_cj, old_ci] + sediment[old_cj, old_ci]

        new_ci = int(new_x)
        new_cj = int(new_y)
        new_h = heightmap[new_cj, new_ci] + sediment[new_cj, new_ci]
        delta_h = new_h - old_h

        # 7. Update speed.
        speed = (max(speed * speed - delta_h * gravity, 0.0)) ** 0.5

        # 8. Compute sediment capacity.
        capacity = max(-delta_h, min_slope) * speed * water * capacity_factor

        # Bilinear weights for the old position (for deposition/erosion).
        ox0 = int(pos_x)
        oy0 = int(pos_y)
        ox1 = min(ox0 + 1, cols - 1)
        oy1 = min(oy0 + 1, rows - 1)
        ofx = pos_x - ox0
        ofy = pos_y - oy0
        w00 = (1 - ofx) * (1 - ofy)
        w10 = ofx * (1 - ofy)
        w01 = (1 - ofx) * ofy
        w11 = ofx * ofy

        if sed_load > capacity:
            # 9. Deposit excess sediment.
            deposit = (sed_load - capacity) * deposition_rate
            sediment[oy0, ox0] += deposit * w00
            sediment[oy0, ox1] += deposit * w10
            sediment[oy1, ox0] += deposit * w01
            sediment[oy1, ox1] += deposit * w11
            sed_load -= deposit
        else:
            # 10. Erode terrain.
            erode_amount = min(
                (capacity - sed_load) * erosion_rate,
                max(-delta_h + 0.001, 0.0),
            )

            if erode_amount > 1e-10:
                # Distribute erosion over cells within erosion_radius.
                total_weight = 0.0
                r = erosion_radius

                # Pre-compute bounding box.
                imin = max(0, old_cj - r)
                imax = min(rows - 1, old_cj + r)
                jmin = max(0, old_ci - r)
                jmax = min(cols - 1, old_ci + r)

                # First pass: total weight.
                for iy in range(imin, imax + 1):
                    for ix in range(jmin, jmax + 1):
                        dist = ((ix - pos_x) ** 2 + (iy - pos_y) ** 2) ** 0.5
                        w = max(0.0, r - dist)
                        total_weight += w

                if total_weight > 1e-10:
                    # Second pass: apply erosion.
                    for iy in range(imin, imax + 1):
                        for ix in range(jmin, jmax + 1):
                            dist = ((ix - pos_x) ** 2 + (iy - pos_y) ** 2) ** 0.5
                            w = max(0.0, r - dist)
                            if w < 1e-10:
                                continue
                            frac = w / total_weight
                            cell_erode = erode_amount * frac

                            # Erode sediment first, then bedrock.
                            if sediment[iy, ix] >= cell_erode:
                                sediment[iy, ix] -= cell_erode
                            else:
                                remainder = cell_erode - sediment[iy, ix]
                                sediment[iy, ix] = 0.0
                                heightmap[iy, ix] -= remainder * erodibility[iy, ix]

                    sed_load += erode_amount

        # 11. Evaporate.
        water *= (1 - evaporation_rate)

        # 12. Kill if water too low.
        if water < 0.01:
            break

        # Advance position.
        pos_x = new_x
        pos_y = new_y


# ------------------------------------------------------------------
# Public API: hydraulic erosion
# ------------------------------------------------------------------

def erode_hydraulic(
    heightmap: np.ndarray,
    bedrock_type: np.ndarray,
    precipitation: np.ndarray,
    flow_acc: np.ndarray,
    sediment: np.ndarray,
    rng: np.random.Generator,
    params: ErosionParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run particle-based hydraulic erosion over the terrain.

    Parameters
    ----------
    heightmap : (H, W) float64
        Bedrock surface elevation in metres.  Not modified.
    bedrock_type : (H, W) int32
        Bedrock-type index (0–5) per cell, from the geology tier.
    precipitation : (H, W) float64
        Annual precipitation in mm.
    flow_acc : (H, W) float64
        D8 flow accumulation (cell counts).
    sediment : (H, W) float64
        Existing sediment depth in metres.  Copied internally.
    rng : numpy.random.Generator
        Seeded RNG for reproducibility.
    params : ErosionParams, optional
        Tuneable knobs.  Defaults to ``ErosionParams()``.

    Returns
    -------
    eroded_heightmap : (H, W) float64
        Modified bedrock surface.
    sediment_depth : (H, W) float64
        Updated sediment layer.
    """
    if params is None:
        params = ErosionParams()

    rows, cols = heightmap.shape

    # 1. Build erodibility map.
    erodibility = np.ones_like(heightmap, dtype=np.float64)
    for code, factor in BEDROCK_ERODIBILITY.items():
        erodibility[bedrock_type == code] = factor

    # 2. Working copies.
    eroded = heightmap.copy()
    sed = sediment.copy()

    # 3. Spawn weights from precipitation and flow accumulation.
    weights = precipitation * np.sqrt(flow_acc + 1.0)
    weights_flat = weights.ravel()
    total = weights_flat.sum()
    if total < 1e-10:
        return eroded, sed
    probs = weights_flat / total

    # 4. Sample spawn positions.
    flat_indices = rng.choice(rows * cols, size=params.num_particles, p=probs)
    spawn_rows, spawn_cols = np.divmod(flat_indices, cols)

    # Small random offset within each cell for sub-cell variation.
    offsets_x = rng.uniform(0.0, 1.0, size=params.num_particles)
    offsets_y = rng.uniform(0.0, 1.0, size=params.num_particles)

    # 5. Simulate each particle.
    for i in range(params.num_particles):
        px = float(spawn_cols[i]) + offsets_x[i]
        py = float(spawn_rows[i]) + offsets_y[i]

        # Clamp to safe bounds (1-cell margin).
        px = max(1.0, min(px, cols - 2.0))
        py = max(1.0, min(py, rows - 2.0))

        _simulate_particle(
            eroded, sed, erodibility,
            px, py,
            params.inertia,
            params.capacity_factor,
            params.deposition_rate,
            params.erosion_rate,
            params.evaporation_rate,
            params.gravity,
            params.min_slope,
            params.max_lifetime,
            params.erosion_radius,
        )

    # 6. Clip to valid ranges.
    np.clip(eroded, 0, None, out=eroded)
    np.clip(sed, 0, None, out=sed)

    return eroded, sed


# ------------------------------------------------------------------
# Public API: thermal erosion
# ------------------------------------------------------------------

def erode_thermal(
    heightmap: np.ndarray,
    sediment: np.ndarray,
    params: ErosionParams | None = None,
) -> None:
    """Thermal erosion (talus creep) — modifies *heightmap* and *sediment* in-place.

    Material on slopes steeper than the talus angle is redistributed to
    neighbours.  Sediment is moved first; if the sediment layer is exhausted,
    bedrock is consumed.

    Fully vectorised numpy — no Python loops over cells.

    Parameters
    ----------
    heightmap : (H, W) float64
        Bedrock surface elevation.  Modified in-place.
    sediment : (H, W) float64
        Sediment depth.  Modified in-place.
    params : ErosionParams, optional
        Tuneable knobs.
    """
    if params is None:
        params = ErosionParams()

    rows, cols = heightmap.shape

    # Neighbour offsets: (dy, dx, cell_distance).
    neighbours = [
        (-1,  0, 1.0),     # N
        ( 1,  0, 1.0),     # S
        ( 0, -1, 1.0),     # W
        ( 0,  1, 1.0),     # E
        (-1, -1, 1.414),   # NW
        (-1,  1, 1.414),   # NE
        ( 1, -1, 1.414),   # SW
        ( 1,  1, 1.414),   # SE
    ]

    for _ in range(params.thermal_passes):
        surface = heightmap + sediment

        # Pad with edge values so boundary cells have valid neighbours.
        padded = np.pad(surface, 1, mode="edge")

        # Accumulate total excess and per-direction contributions.
        total_excess = np.zeros((rows, cols), dtype=np.float64)
        direction_excess = []

        for dy, dx, cell_dist in neighbours:
            # Neighbour heights (shifted view into padded array).
            nb = padded[1 + dy : rows + 1 + dy, 1 + dx : cols + 1 + dx]
            diff = surface - nb
            threshold = params.talus_slope * cell_dist
            excess = np.maximum(diff - threshold, 0.0)
            direction_excess.append(excess)
            total_excess += excess

        # Nothing to move if no cell exceeds the talus angle.
        if total_excess.max() < 1e-10:
            continue

        # Amount to remove from each source cell (shared across directions).
        # Move creep_rate of the total excess, split proportionally.
        move_total = total_excess * params.creep_rate

        # Source: remove material from sediment first, then bedrock.
        from_sed = np.minimum(move_total, sediment)
        from_rock = move_total - from_sed
        sediment -= from_sed
        heightmap -= from_rock

        # Distribute to each neighbour proportional to that direction's excess.
        for idx, (dy, dx, _cell_dist) in enumerate(neighbours):
            exc = direction_excess[idx]
            # Fraction of total excess going to this direction.
            safe_total = np.where(total_excess > 1e-10, total_excess, 1.0)
            frac = np.where(total_excess > 1e-10, exc / safe_total, 0.0)
            contrib = move_total * frac

            # Add material to neighbour cells (as sediment).
            # Use slicing to shift: contrib placed at neighbour positions.
            # Target row range: [max(0,dy) .. rows+min(0,dy))
            # Source row range:  [max(0,-dy) .. rows+min(0,-dy))
            sr0 = max(0, -dy)
            sr1 = rows + min(0, -dy)
            sc0 = max(0, -dx)
            sc1 = cols + min(0, -dx)
            tr0 = max(0, dy)
            tr1 = rows + min(0, dy)
            tc0 = max(0, dx)
            tc1 = cols + min(0, dx)

            sediment[tr0:tr1, tc0:tc1] += contrib[sr0:sr1, sc0:sc1]

        # Clip to valid ranges.
        np.clip(sediment, 0, None, out=sediment)
        np.clip(heightmap, 0, None, out=heightmap)


# ------------------------------------------------------------------
# Per-season erosion (lightweight, called every seasonal tick)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SeasonalErosionParams:
    """Parameters for lightweight per-season erosion."""

    # Erosion strength: target ~1-2m mean erosion per 1000yr (4000 ticks).
    # Each tick: erosion_scale * flow_norm * precip_norm * (1+storm*2) * slope * erodibility
    # With typical values: ~0.3 * 1.0 * 2.0 * 0.05 * 0.8 = 0.024 effective multiplier
    # So 0.01 * 0.024 ≈ 0.00024m per tick → ~1m per 4000 ticks
    erosion_scale: float = 0.01  # m of erosion per unit erosion_potential

    # Deposition: fraction of eroded material deposited where slope decreases
    deposition_fraction: float = 0.7

    # Thermal diffusion rate: ~0.001mm per tick
    diffusion_rate: float = 1e-6  # m per tick (applied via laplacian)

    # Storm scale factor for storm_intensity → erosion multiplier
    storm_scale: float = 2.0


def erode_seasonal(
    heightmap: np.ndarray,
    sediment: np.ndarray,
    flow_accumulation: np.ndarray,
    precipitation: np.ndarray,
    storm_intensity: float,
    bedrock_type: np.ndarray,
    params: SeasonalErosionParams | None = None,
) -> None:
    """Lightweight per-season erosion. Modifies heightmap and sediment in-place.

    Algorithm:
    1. Compute erosion potential per cell = flow_acc * precip * storm * slope * erodibility
    2. Erode: remove material proportional to potential (sediment first, then bedrock)
    3. Deposit: material deposited where slope decreases (simple downhill redistribution)

    Calibrated so ~4000 ticks produces ~11m mean erosion (matching the batch system).
    """
    if params is None:
        params = SeasonalErosionParams()

    rows, cols = heightmap.shape
    combined = heightmap + sediment

    # 1. Compute slope magnitude
    dy, dx = np.gradient(combined, 50.0)  # 50m cell size
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.clip(slope, 0.001, None)  # minimum slope to prevent div-by-zero

    # 2. Compute erodibility from bedrock type
    erodibility = np.ones_like(heightmap)
    for btype, erod_val in BEDROCK_ERODIBILITY.items():
        erodibility = np.where(bedrock_type == btype, erod_val, erodibility)

    # 3. Normalize flow accumulation and precipitation
    flow_norm = np.log1p(flow_accumulation) / np.log1p(flow_accumulation.max() + 1)
    precip_norm = precipitation / (precipitation.mean() + 1e-10)

    # 4. Erosion potential
    erosion_potential = (
        flow_norm
        * precip_norm
        * (1.0 + storm_intensity * params.storm_scale)
        * slope
        * erodibility
        * params.erosion_scale
    )

    # 5. Erode: remove from sediment first, then bedrock
    erode_from_sed = np.minimum(erosion_potential, sediment)
    remaining = erosion_potential - erode_from_sed
    erode_from_rock = remaining * erodibility  # harder rock resists more

    sediment -= erode_from_sed
    heightmap -= erode_from_rock

    total_eroded = erode_from_sed + erode_from_rock

    # 6. Deposit where slope is low (simple: deposit proportional to inverse slope)
    # Material moves downhill; deposited where slope decreases
    deposition = total_eroded * params.deposition_fraction
    # Smooth the deposition to simulate downhill transport
    # Use a simple 3x3 averaging to spread deposits
    padded = np.pad(deposition, 1, mode="edge")
    smoothed_dep = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    # Deposit preferentially where slope is low
    inverse_slope = 1.0 / (slope + 0.1)
    deposit_weight = inverse_slope / (inverse_slope.mean() + 1e-10)
    deposit_weight = np.clip(deposit_weight, 0, 3.0)  # cap to prevent extreme deposits

    sediment += smoothed_dep * deposit_weight

    # Ensure non-negative
    np.clip(sediment, 0, None, out=sediment)
    np.clip(heightmap, 0, None, out=heightmap)


def thermal_diffusion(
    heightmap: np.ndarray,
    sediment: np.ndarray,
    diffusion_rate: float = 1e-6,
) -> None:
    """Tiny diffusion applied each tick — imperceptible per ride, visible over years.

    Approximates laplacian diffusion on the combined surface.
    Over 4000 ticks (~1000 years), produces ~4mm of smoothing on sharp features.
    Over 400K ticks (~100K years), visible skyline changes.
    """
    combined = heightmap + sediment

    # Laplacian via 3x3 kernel: center - average of neighbors
    padded = np.pad(combined, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]  # N + S
        + padded[1:-1, :-2] + padded[1:-1, 2:]  # W + E
        - 4.0 * padded[1:-1, 1:-1]
    )

    # Apply diffusion to sediment (not bedrock — sediment moves easier)
    change = laplacian * diffusion_rate
    sediment += change

    # Ensure non-negative
    np.clip(sediment, 0, None, out=sediment)

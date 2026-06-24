"""Cubic Bezier spline with arc-length parameterization.

Matches Godot 4's Curve3D convention exactly:
  - Control point handles are relative to the point position
  - sample_transform(distance) matches sample_baked_with_rotation()
  - Coordinate system: X=right, Y=up, Z=backward (forward = -Z)

Ported from bike-trainer-godot/world-builder/bezier.py.
"""

from __future__ import annotations

import numpy as np


def _cubic_bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    t = float(np.clip(t, 0, 1))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def _cubic_bezier_tangent(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    t = float(np.clip(t, 0.0001, 0.9999))
    return 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


class BakedCurve:
    """Arc-length parameterized cubic Bezier spline.

    Control points format::

        [
            {"position": [x,y,z], "handle_in": [x,y,z], "handle_out": [x,y,z]},
            ...
        ]

    Handles are relative to position — same as Godot's add_point(pos, in, out).
    """

    SAMPLES_PER_SEGMENT: int = 500

    def __init__(self, control_points: list[dict], bake_interval: float = 1.0) -> None:
        self.control_points = control_points
        self.bake_interval = bake_interval
        self._bake()

    def _bake(self) -> None:
        positions: list[np.ndarray] = []
        tangents: list[np.ndarray] = []
        cumulative: list[float] = [0.0]

        n_segments = len(self.control_points) - 1
        for seg in range(n_segments):
            cp0 = self.control_points[seg]
            cp1 = self.control_points[seg + 1]

            p0 = np.array(cp0["position"], dtype=float)
            p1 = p0 + np.array(cp0["handle_out"], dtype=float)
            p2 = np.array(cp1["position"], dtype=float) + np.array(cp1["handle_in"], dtype=float)
            p3 = np.array(cp1["position"], dtype=float)

            n = self.SAMPLES_PER_SEGMENT
            for i in range(n):
                t = i / n
                pos = _cubic_bezier(p0, p1, p2, p3, t)
                tan = _cubic_bezier_tangent(p0, p1, p2, p3, t)
                positions.append(pos)
                tangents.append(tan)
                if len(positions) > 1:
                    dist = np.linalg.norm(positions[-1] - positions[-2])
                    cumulative.append(cumulative[-1] + dist)

        # Final point of last segment
        cp0 = self.control_points[-2]
        cp1 = self.control_points[-1]
        p0 = np.array(cp0["position"], dtype=float)
        p1 = p0 + np.array(cp0["handle_out"], dtype=float)
        p2 = np.array(cp1["position"], dtype=float) + np.array(cp1["handle_in"], dtype=float)
        p3 = np.array(cp1["position"], dtype=float)
        positions.append(_cubic_bezier(p0, p1, p2, p3, 1.0))
        tangents.append(_cubic_bezier_tangent(p0, p1, p2, p3, 1.0))
        cumulative.append(cumulative[-1] + np.linalg.norm(positions[-1] - positions[-2]))

        self._positions = np.array(positions)
        self._tangents = np.array(tangents)
        self._arc_lengths = np.array(cumulative)
        self.total_length: float = float(self._arc_lengths[-1])

    def _index_at(self, distance: float) -> int:
        distance = float(np.clip(distance, 0.0, self.total_length))
        idx = int(np.searchsorted(self._arc_lengths, distance))
        return min(idx, len(self._positions) - 1)

    def sample_position(self, distance: float) -> np.ndarray:
        """World position at *distance* meters along path."""
        return self._positions[self._index_at(distance)].copy()

    def sample_transform(self, distance: float) -> dict[str, np.ndarray]:
        """Coordinate frame at *distance* meters along path.

        Returns dict with keys:
            origin  — position [x, y, z]
            forward — direction of travel (−Z dominant)
            right   — lateral right (+X dominant)
            up      — local up (+Y dominant)
        """
        idx = self._index_at(distance)
        pos = self._positions[idx]
        tan = self._tangents[idx]

        tan_len = np.linalg.norm(tan)
        forward = tan / tan_len if tan_len > 1e-8 else np.array([0.0, 0.0, -1.0])

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_len

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        return {"origin": pos, "forward": forward, "right": right, "up": up}

    def world_point(self, distance: float, lateral: float, height: float) -> np.ndarray:
        """Convert path-space (distance, lateral offset, height) to world position.

        lateral > 0 = rider's right side.
        """
        xf = self.sample_transform(distance)
        return xf["origin"] + xf["right"] * lateral + xf["up"] * height

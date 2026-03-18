#!/usr/bin/env python3
"""
Shared utility functions for path-tracking controllers.

All functions assume the **vehicle-frame** convention:
  - Vehicle is at the origin (0, 0), heading along +X.
  - Waypoints are expressed relative to the vehicle.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

# =============================================================================
# Constants
# =============================================================================
MIN_SEGMENT_LENGTH_M = 1e-9
MIN_CURVATURE_DENOM = 1e-9
DEFAULT_CURVATURE_CLAMP = 0.5          # rad/m  (generous default)


# =============================================================================
# Closest-Point Search
# =============================================================================

def find_closest_waypoint_idx(
    positions: npt.NDArray[np.float64],
) -> int:
    """
    Index of the waypoint closest to the vehicle (origin).

    Args:
        positions: (N, 2) array of waypoint positions in vehicle frame.

    Returns:
        Index into `positions` of the nearest point.
    """
    dists_sq = positions[:, 0] ** 2 + positions[:, 1] ** 2
    return int(np.argmin(dists_sq))


# =============================================================================
# Tracking Errors
# =============================================================================

def cross_track_error(
    prev_pt: npt.NDArray[np.float64],
    next_pt: npt.NDArray[np.float64],
) -> float:
    """
    Signed perpendicular distance from the vehicle to a path segment.

    Uses the 2-D cross-product formula.  The vehicle is assumed to be at
    the origin in the local frame.

    Args:
        prev_pt: Start of the path segment (2,).
        next_pt: End of the path segment (2,).

    Returns:
        Cross-track error in metres.
        Positive  =>  vehicle is to the **left** of the path.
        Negative  =>  vehicle is to the **right** of the path.
    """
    path_vec = next_pt - prev_pt
    seg_len = np.linalg.norm(path_vec)
    if seg_len < MIN_SEGMENT_LENGTH_M:
        return 0.0

    car_vec = -prev_pt                              # prev_pt → origin
    cross = path_vec[0] * car_vec[1] - path_vec[1] * car_vec[0]
    return float(cross / seg_len)


def heading_error(
    prev_pt: npt.NDArray[np.float64],
    next_pt: npt.NDArray[np.float64],
) -> float:
    """
    Heading error between the vehicle and a path segment.

    Defined as  vehicle_heading − path_heading.  In the vehicle frame the
    vehicle heading is 0, so:

        e_psi = −atan2(path_vec.y, path_vec.x)

    Args:
        prev_pt: Start of the path segment (2,).
        next_pt: End of the path segment (2,).

    Returns:
        Heading error in radians.
        Positive  =>  vehicle is yawed **clockwise** of the path.
    """
    path_vec = next_pt - prev_pt
    seg_len = np.linalg.norm(path_vec)
    if seg_len < MIN_SEGMENT_LENGTH_M:
        return 0.0

    return -float(np.arctan2(path_vec[1], path_vec[0]))


def compute_tracking_errors(
    positions: npt.NDArray[np.float64],
    closest_idx: int,
) -> Tuple[float, float]:
    """
    Lateral and heading errors relative to the nearest path segment.

    Convenience wrapper that picks the correct segment around
    `closest_idx` and calls :func:`cross_track_error` and
    :func:`heading_error`.

    Args:
        positions: (N, 2) waypoint array in vehicle frame.
        closest_idx: Index of the closest waypoint.

    Returns:
        (e_y, e_psi)
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0, 0.0

    if closest_idx == 0:
        prev_pt = positions[0]
        next_pt = positions[1]
    else:
        prev_pt = positions[closest_idx - 1]
        next_pt = positions[closest_idx]

    e_y = cross_track_error(prev_pt, next_pt)
    e_psi = heading_error(prev_pt, next_pt)
    return e_y, e_psi


# =============================================================================
# Path Lookahead
# =============================================================================

def find_lookahead_idx(
    positions: npt.NDArray[np.float64],
    start_idx: int,
    lookahead_dist: float,
) -> int:
    """
    Walk forward along the path from ``start_idx`` by ``lookahead_dist`` metres.

    Accumulates segment arc-lengths until the total exceeds the desired
    lookahead distance, then returns that waypoint index.  Clamps to the
    last waypoint if the path is shorter than the requested distance.

    Args:
        positions: (N, 2) waypoint array in vehicle frame.
        start_idx: Index to begin walking from (typically the closest
                   waypoint to the vehicle).
        lookahead_dist: Desired lookahead distance in metres.

    Returns:
        Index of the waypoint at (or just past) the lookahead distance.
    """
    n = positions.shape[0]
    accum = 0.0
    idx = start_idx
    while idx < n - 1:
        step = float(np.linalg.norm(positions[idx + 1] - positions[idx]))
        accum += step
        if accum >= lookahead_dist:
            return idx + 1
        idx += 1
    return n - 1


# =============================================================================
# Curvature
# =============================================================================

def estimate_horizon_curvature(
    positions: npt.NDArray[np.float64],
    closest_idx: int,
    clamp: float = DEFAULT_CURVATURE_CLAMP,
) -> float:
    """
    Menger curvature of the visible path horizon.

    Uses three waypoints — the nearest to the vehicle, the furthest, and the
    midpoint between them — to capture the overall arc of the horizon rather
    than local oscillations.

    Args:
        positions: (N, 2) waypoint array in vehicle frame.
        closest_idx: Index of the waypoint closest to the vehicle.
        clamp: Maximum absolute curvature (rad/m).

    Returns:
        Curvature in rad/m, clamped to ``[-clamp, clamp]``.
    """
    n = positions.shape[0]
    if n < 3:
        return 0.0

    i_near = closest_idx
    i_far = n - 1
    if i_far <= i_near:
        return 0.0

    i_mid = (i_near + i_far) // 2
    if i_mid == i_near or i_mid == i_far:
        return 0.0

    p0, p1, p2 = positions[i_near], positions[i_mid], positions[i_far]

    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)

    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

    denom = a * b * c
    if denom < MIN_CURVATURE_DENOM:
        return 0.0

    kappa = 2.0 * cross / denom
    return float(np.clip(kappa, -clamp, clamp))


def estimate_curvature(
    positions: npt.NDArray[np.float64],
    idx: int,
    clamp: float = DEFAULT_CURVATURE_CLAMP,
) -> float:
    """
    Menger curvature from three consecutive waypoints.

    Uses the circumscribed-circle formula::

        kappa = 2 * signed_area / (|a| * |b| * |c|)

    Sign follows the right-hand rule (+  = left turn).

    Args:
        positions: (N, 2) waypoint array in vehicle frame.
        idx: Index of the centre point.
        clamp: Maximum absolute curvature (rad/m).

    Returns:
        Curvature in rad/m, clamped to ``[-clamp, clamp]``.
    """
    n = positions.shape[0]
    if n < 3:
        return 0.0

    i0 = max(0, idx - 1)
    i2 = min(n - 1, idx + 1)
    if i0 == idx or idx == i2:
        return 0.0

    p0, p1, p2 = positions[i0], positions[idx], positions[i2]

    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)

    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

    denom = a * b * c
    if denom < MIN_CURVATURE_DENOM:
        return 0.0

    kappa = 2.0 * cross / denom
    return float(np.clip(kappa, -clamp, clamp))

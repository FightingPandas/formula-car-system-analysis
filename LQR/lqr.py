#!/usr/bin/env python3
"""
Gain-Scheduled LQR Controller.

Coupled lateral-longitudinal controller that uses a pre-computed look-up table
of LQR gain matrices, indexed by (velocity, curvature).  At each control step
the 2x5 gain matrix K is bilinearly interpolated from the LuT, then applied
with feedforward + feedback:

    u = u_0  +  (-K @ delta_x)

State vector: x = [e_vx, v_y, yaw_rate, e_y, e_psi]^T
Input vector: u = [F_x, delta]^T

The F_x output is converted to per-wheel motor torque so that the downstream
control message matches the convention used by the PID throttle controller.

Requires:
    A pre-computed LuT at  <package>/data/lqr_lut.pkl
    Generate it with:  ros2 run controls lqr_generate_lut
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from controls.base_controller import ControlOutput, CoupledControllerBase
from controls.utils import (
    compute_tracking_errors,
    estimate_curvature,
    estimate_horizon_curvature,
    find_closest_waypoint_idx,
    find_lookahead_idx,
)
from orchestrator.vehicle_config import VehicleConfig

# =============================================================================
# Constants
# =============================================================================
LUT_FILENAME = "lqr_lut.pkl"
LUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", LUT_FILENAME)

MIN_REFERENCE_V_MPS = 1.0


# =============================================================================
# Controller
# =============================================================================

class GainScheduledLQR(CoupledControllerBase):
    """
    Gain-scheduled LQR for coupled path tracking.

    Loads a pre-computed look-up table of 2x5 gain matrices K(V, kappa) and
    interpolates bilinearly at runtime.  Feedforward terms u_0 and equilibrium
    state x_0 are computed analytically from the bicycle model.

    An optional *kinematic* feedforward term can be enabled to anticipate
    upcoming curvature changes before the feedback loop reacts::

        delta_ff = L * (kappa_target - kappa_current)

    where ``kappa_target`` is the local curvature at a lookahead point on the
    path.  This eliminates the initial lag when entering a turn from a
    straight (or vice versa) because the controller doesn't have to wait for
    cross-track and heading errors to build up.

    Args:
        lut_path: Path to the pickled LuT file.
        use_feedforward: Enable the kinematic lookahead feedforward.
        feedforward_lookahead_time_sec: How far ahead (in seconds) to sample
            the target curvature.  Converted to metres at runtime via
            ``lookahead_dist = lookahead_time * vx``.

    Raises:
        FileNotFoundError: If the LuT file does not exist.
    """

    def __init__(
        self,
        lut_path: str = LUT_PATH,
        use_feedforward: bool = True,
        feedforward_lookahead_time_sec: float = 0.5,
    ) -> None:
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(
                f"LQR LuT not found at {lut_path}.  "
                "Run:  ros2 run controls lqr_generate_lut"
            )

        self._vc = VehicleConfig()
        self._use_feedforward = use_feedforward
        self._ff_lookahead_time_sec = feedforward_lookahead_time_sec

        self.ey_injection: float = 0.0
        self._debug: dict = {}

        # -----------------------------------------------------------------
        # Load LuT
        # -----------------------------------------------------------------
        with open(lut_path, "rb") as f:
            lut = pickle.load(f)

        self._V_grid: np.ndarray = lut["V_grid"]
        self._kappa_grid: np.ndarray = lut["kappa_grid"]
        K_table: np.ndarray = lut["K_table"]           # (nV, nK, 2, 5)

        # -----------------------------------------------------------------
        # Build one RegularGridInterpolator per K-matrix element (2x5 = 10)
        # -----------------------------------------------------------------
        self._K_interps: List[List[RegularGridInterpolator]] = []
        for row in range(2):
            row_interps: List[RegularGridInterpolator] = []
            for col in range(5):
                interp = RegularGridInterpolator(
                    (self._V_grid, self._kappa_grid),
                    K_table[:, :, row, col],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,        # nearest-neighbour extrapolation
                )
                row_interps.append(interp)
            self._K_interps.append(row_interps)

    # =====================================================================
    # Public interface (CoupledControllerBase)
    # =====================================================================

    def compute(
        self,
        path: List[Tuple[npt.NDArray[np.float64], float, float]],
        current_velocity: float,
        vehicle_state: Optional[npt.NDArray[np.float64]] = None,
    ) -> ControlOutput:
        """
        Compute coupled (steering, throttle) command.

        Args:
            path: Waypoints in **vehicle frame** as
                  [(position, v_des, curvature), ...].
            current_velocity: Scalar speed (m/s) — not used directly; vx
                              comes from vehicle_state for accuracy.
            vehicle_state: np.array([vx, vy, yaw_rate]) in body frame.

        Returns:
            ControlOutput with steering (rad) and throttle (motor torque Nm).
        """
        if len(path) < 2 or vehicle_state is None:
            return ControlOutput(steering=0.0, throttle=0.0)

        vc = self._vc
        vx = float(vehicle_state[0])
        vy = float(vehicle_state[1])
        yaw_rate = float(vehicle_state[2])

        # -----------------------------------------------------------------
        # Reference point: closest waypoint on the path
        # -----------------------------------------------------------------
        positions = np.array([pt[0] for pt in path])
        closest_idx = find_closest_waypoint_idx(positions)

        reference_V = max(float(path[closest_idx][1]), MIN_REFERENCE_V_MPS)
        kappa = estimate_horizon_curvature(positions, closest_idx)

        # -----------------------------------------------------------------
        # Tracking errors (shared helpers)
        # -----------------------------------------------------------------
        e_y, e_psi = compute_tracking_errors(positions, closest_idx)
        e_y += self.ey_injection
        e_vx = vx - reference_V

        # Measured state
        x_meas = np.array([e_vx, vy, yaw_rate, e_y, e_psi])

        # -----------------------------------------------------------------
        # Equilibrium state  x_0  (steady-state at reference V, kappa)
        # -----------------------------------------------------------------
        v_y0 = reference_V * (
            vc.L_r * kappa
            - (vc.m * vc.L_f * reference_V**2 * kappa) / (vc.L * vc.C_r)
        )
        yaw_rate0 = kappa * reference_V
        e_psi0 = -v_y0 / reference_V

        x_0 = np.array([0.0, v_y0, yaw_rate0, 0.0, e_psi0])

        # -----------------------------------------------------------------
        # Feedforward inputs  u_0
        # -----------------------------------------------------------------
        F_x0 = (
            0.5 * vc.rho * vc.Cd * vc.A * reference_V**2
            - vc.m * kappa * reference_V * v_y0
        )
        delta_0 = (
            vc.L * kappa
            + (vc.m * reference_V**2 * kappa / vc.L)
            * (vc.L_r / vc.C_f - vc.L_f / vc.C_r)
        )
        u_0 = np.array([F_x0, delta_0])

        # -----------------------------------------------------------------
        # Feedback:  delta_u = -K @ (x - x_0)
        # -----------------------------------------------------------------
        K = self._interpolate_K(reference_V, kappa)
        delta_x = x_meas - x_0
        delta_u = -K @ delta_x

        u_cmd = u_0 + delta_u
        F_x = float(u_cmd[0])
        delta = float(u_cmd[1])

        # -----------------------------------------------------------------
        # Kinematic feedforward: anticipate upcoming curvature changes
        #
        # The existing u_0 already contains the steady-state steering for
        # the *current* operating-point curvature (kappa).  This extra term
        # looks further down the path and adds the *incremental* kinematic
        # steering needed for the curvature that is coming:
        #
        #   delta_ff = L * (kappa_target − kappa)
        #
        # In steady-state (constant curvature) this is zero, so it won't
        # double-count.  On a straight approaching a turn it equals the full
        # L * kappa_target, giving the step-response anticipation we want.
        # -----------------------------------------------------------------
        if self._use_feedforward:
            lookahead_dist = self._ff_lookahead_time_sec * max(vx, MIN_REFERENCE_V_MPS)
            la_idx = find_lookahead_idx(positions, closest_idx, lookahead_dist)
            kappa_target = estimate_curvature(positions, la_idx)
            delta_ff = vc.L * (kappa_target - kappa)
            delta += delta_ff

        # -----------------------------------------------------------------
        # Output conversion
        # -----------------------------------------------------------------
        delta = float(np.clip(delta, -vc.steer_max_rad, vc.steer_max_rad))

        # F_x (N) → per-wheel motor torque (Nm) to match PID convention.
        # Simulator applies:  T_wheel = T_motor * GR
        # So:  T_motor = F_x * R / (4 * GR)
        torque = F_x * vc.wheel_radius / (4.0 * vc.GR)

        self._debug = {
            "x_meas": x_meas.copy(),
            "F_x": F_x,
            "delta": delta,
            "kappa": kappa,
            "ref_V": reference_V,
        }

        return ControlOutput(steering=delta, throttle=torque)

    def reset(self) -> None:
        """LQR is memoryless — nothing to reset."""

    # =====================================================================
    # Private helpers
    # =====================================================================

    def _interpolate_K(self, V: float, kappa: float) -> np.ndarray:
        """
        Bilinearly interpolate the 2x5 K matrix from the LuT.

        Args:
            V: Longitudinal velocity (m/s).
            kappa: Path curvature (rad/m).

        Returns:
            2x5 gain matrix.
        """
        K = np.zeros((2, 5))
        pt = np.array([[V, kappa]])
        for row in range(2):
            for col in range(5):
                K[row, col] = float(self._K_interps[row][col](pt))
        return K


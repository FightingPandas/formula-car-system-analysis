#!/usr/bin/env python3
"""
Offline LQR Gain Look-Up Table Generator.

Sweeps a grid of (velocity, curvature) operating points and solves the
Continuous Algebraic Riccati Equation (CARE) at each to produce a pre-computed
look-up table of 2x5 LQR gain matrices K.

State vector: x = [e_vx, v_y, yaw_rate, e_y, e_psi]^T
Input vector: u = [F_x, delta]^T

The A matrix is parameterized by (V, kappa); B is constant.  Bryson's Rule
provides the Q and R weighting matrices.

Usage:
    ros2 run controls lqr_generate_lut

Output:
    <package>/data/lqr_lut.pkl   (pickle with V_grid, kappa_grid, K_table, Q, R)

To modify tuning, edit the Q and R matrices below and re-run.
"""

import os
import pickle
import sys

import numpy as np
from scipy.linalg import solve_continuous_are

from orchestrator.vehicle_config import VehicleConfig

# =============================================================================
# Grid Parameters
# =============================================================================
V_MIN_MPS = 1.0
V_MAX_MPS = 25.0
V_STEP_MPS = 1.0

KAPPA_MIN_RADPM = -0.1
KAPPA_MAX_RADPM = 0.1
KAPPA_STEP_RADPM = 0.01

# =============================================================================
# Bryson's Rule Tuning Matrices
# =============================================================================
#                  e_vx   v_y   yaw_rate  e_y   e_psi
Q = np.diag([      1.0,   4.0,  0.5,      16.0, 16.0])

#                  F_x        delta
R = np.diag([      4.44e-7,   3.0])


# =============================================================================
# System Matrices
# =============================================================================

def build_A(
    V: float,
    kappa: float,
    vc: VehicleConfig,
) -> np.ndarray:
    """
    Build the continuous-time A matrix for a given operating point.

    Args:
        V: Longitudinal velocity (m/s).  Must be > 0.
        kappa: Path curvature (rad/m).
        vc: Vehicle configuration dataclass.

    Returns:
        5x5 numpy array (continuous-time state matrix).
    """
    m = vc.m
    L = vc.L
    L_f = vc.L_f
    L_r = vc.L_r
    C_af = vc.C_f
    C_ar = vc.C_r
    I_z = vc.I_z
    rho = vc.rho
    C_d = vc.Cd
    A_front = vc.A

    # Row 1 (e_vx dynamics)
    A11 = -(A_front * C_d * rho * V) / m
    A13 = (
        kappa * V
        * (C_ar * L_f * L_r + C_ar * L_r**2 - L_f * V**2 * m)
        / (C_ar * L)
    )

    # Row 2 (v_y dynamics)
    A21 = (
        kappa
        * (
            C_af * C_ar * L_f**2
            + 2 * C_af * C_ar * L_f * L_r
            + C_af * C_ar * L_r**2
            - C_af * L_f * V**2 * m
            - 2 * C_ar * L_f * V**2 * m
            - C_ar * L_r * V**2 * m
        )
        / (C_ar * V * m * L)
    )
    A22 = -(C_af + C_ar) / (V * m)
    A23 = (-C_af * L_f + C_ar * L_r - V**2 * m) / (V * m)

    # Row 3 (yaw_rate dynamics)
    A31 = (
        kappa * L_f
        * (
            C_af * C_ar * L_f**2
            + 2 * C_af * C_ar * L_f * L_r
            + C_af * C_ar * L_r**2
            - C_af * L_f * V**2 * m
            + C_ar * L_r * V**2 * m
        )
        / (C_ar * I_z * V * L)
    )
    A32 = (-C_af * L_f + C_ar * L_r) / (I_z * V)
    A33 = -(C_af * L_f**2 + C_ar * L_r**2) / (I_z * V)

    # Equilibrium states used in rows 4-5
    v_y0 = V * (
        L_r * kappa - (m * L_f * V**2 * kappa) / (L * C_ar)
    )
    e_psi0 = -v_y0 / V

    A = np.array([
        [A11,    kappa * V, A13, 0.0, 0.0                ],
        [A21,    A22,       A23, 0.0, 0.0                ],
        [A31,    A32,       A33, 0.0, 0.0                ],
        [e_psi0, 1.0,       0.0, 0.0, V - v_y0 * e_psi0 ],
        [-kappa, 0.0,       1.0, 0.0, 0.0                ],
    ])
    return A


def build_B(vc: VehicleConfig) -> np.ndarray:
    """
    Build the (constant) B input matrix.

    Args:
        vc: Vehicle configuration dataclass.

    Returns:
        5x2 numpy array (continuous-time input matrix).
    """
    m = vc.m
    C_af = vc.C_f
    L_f = vc.L_f
    I_z = vc.I_z

    B = np.array([
        [1.0 / m,  0.0                   ],
        [0.0,      C_af / m              ],
        [0.0,      (L_f * C_af) / I_z    ],
        [0.0,      0.0                   ],
        [0.0,      0.0                   ],
    ])
    return B


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Generate the LQR gain LuT and save to disk."""
    vc = VehicleConfig()

    V_grid = np.arange(V_MIN_MPS, V_MAX_MPS + V_STEP_MPS / 2, V_STEP_MPS)
    kappa_grid = np.arange(
        KAPPA_MIN_RADPM, KAPPA_MAX_RADPM + KAPPA_STEP_RADPM / 2, KAPPA_STEP_RADPM
    )

    B = build_B(vc)
    K_table = np.full((len(V_grid), len(kappa_grid), 2, 5), np.nan)

    n_total = len(V_grid) * len(kappa_grid)
    n_solved = 0
    n_failed = 0

    print(f"Generating LQR LuT:  {len(V_grid)} V  x  {len(kappa_grid)} kappa  =  {n_total} points")
    print(f"  V     : [{V_grid[0]:.1f}, {V_grid[-1]:.1f}] m/s   (step {V_STEP_MPS})")
    print(f"  kappa : [{kappa_grid[0]:.3f}, {kappa_grid[-1]:.3f}] rad/m  (step {KAPPA_STEP_RADPM})")
    print(f"  Q diag: {np.diag(Q).tolist()}")
    print(f"  R diag: {np.diag(R).tolist()}")
    print()

    for i, V in enumerate(V_grid):
        row_ok = 0
        for j, kappa in enumerate(kappa_grid):
            try:
                A = build_A(V, kappa, vc)
                P = solve_continuous_are(A, B, Q, R)
                K = np.linalg.solve(R, B.T @ P)
                K_table[i, j] = K
                n_solved += 1
                row_ok += 1
            except (np.linalg.LinAlgError, ValueError):
                n_failed += 1

        pct = 100.0 * (i + 1) / len(V_grid)
        print(f"  V={V:5.1f} m/s :  {row_ok}/{len(kappa_grid)} solved   [{pct:5.1f}%]")

    # ---- Save ----
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "lqr_lut.pkl")

    lut = {
        "V_grid": V_grid,
        "kappa_grid": kappa_grid,
        "K_table": K_table,
        "Q": Q,
        "R": R,
    }

    with open(out_path, "wb") as f:
        pickle.dump(lut, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print(f"Saved LuT  ->  {out_path}")
    print(f"  Solved: {n_solved}/{n_total}   Failed: {n_failed}/{n_total}")

    if n_failed > 0:
        nan_pct = 100.0 * n_failed / n_total
        print(f"  WARNING: {nan_pct:.1f}% of grid points failed to stabilize (filled with NaN)")
        sys.exit(1)


if __name__ == "__main__":
    main()

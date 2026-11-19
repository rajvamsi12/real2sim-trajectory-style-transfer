"""
il/mapping_rhd.py

Corrected drone-pixel -> CARLA mapping (10 Nov 2026). The point pairs in "drone pixel to carla coordinates.txt" send each
real carriageway onto the CARLA carriageway on the same side of its road, so Gwalior's left-hand traffic runs against
CARLA's lanes (measured: 95.6 % of moving steps). Here each region takes its partner region's CARLA points (1<->2, 3<->4,
5<->6) in the same point order: a mirror across each road, which turns left-hand into right-hand traffic and keeps the
median-side lane next to the median. Drone pixels and region polygons are unchanged.

Does not touch timemethod.py. use_rhd() replaces timemethod.transformation_matrices' values in place, so every module that
already imported the dict (scenario_checks, simulate_and_record_scenarios, ...) sees the new matrices in that process.

  python il/mapping_rhd.py          # print the matrices, their residuals and the old/new determinants
"""

import numpy as np

# drone pixel (u, -v) -> CARLA (x, y); pixels as in "drone pixel to carla coordinates.txt", CARLA targets swapped by pair
PAIRS_RHD = {
    "Region_1": [((1243, -895), (-30.8, 11.4)), ((1293, -895), (-30.8, 18.1)),
                 ((1274, -1510), (69.2, 11.7)), ((1324, -1510), (69.2, 19.1))],
    "Region_2": [((1363, -892), (-30.8, 23.0)), ((1413, -892), (-30.8, 30.0)),
                 ((1338, -1509), (69.2, 22.9)), ((1394, -1509), (69.2, 30.0))],
    "Region_3": [((1432, -735), (-39.4, 4.4)), ((1435, -785), (-47.0, 4.4)),
                 ((1938.5, -677), (-39.4, -72.5)), ((1943, -724), (-46.9, -72.5))],
    "Region_4": [((1435, -798), (-47.0, 4.4)), ((1440, -848), (-54.7, 4.4)),
                 ((1943.5, -738), (-46.9, -72.5)), ((1948, -788), (-54.7, -72.5))],
    "Region_5": [((1190, -644), (-38.1, 35.1)), ((1166, -696), (-46.8, 35.1)),
                 ((569.5, -297), (-38.1, 141.8)), ((548, -344), (-46.8, 141.8))],
    "Region_6": [((1159, -704), (-46.8, 35.1)), ((1137, -755), (-55.3, 35.2)),
                 ((541, -352.5), (-46.8, 141.8)), ((520.5, -401), (-55.3, 141.8))],
}


def fit(pairs):
    """Least-squares 2x3 affine (same form as timemethod.transformation_matrices) from 4 point pairs."""
    A = np.array([[u, v, 1.0] for (u, v), _ in pairs])
    B = np.array([c for _, c in pairs])
    return np.linalg.lstsq(A, B, rcond=None)[0].T


TRANSFORMATION_MATRICES_RHD = {r: fit(p) for r, p in PAIRS_RHD.items()}


def use_rhd():
    """Switch timemethod (and everything that imported its dict) to the corrected matrices for this process."""
    import timemethod
    for r, M in TRANSFORMATION_MATRICES_RHD.items():
        timemethod.transformation_matrices[r] = M
    return timemethod


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import timemethod
    for r, M in TRANSFORMATION_MATRICES_RHD.items():
        res = max(np.hypot(*(M @ [u, v, 1] - np.array(c))) for (u, v), c in PAIRS_RHD[r])
        old = np.asarray(timemethod.transformation_matrices[r])
        print(f"{r}: max residual {res:.2f} m, det old {np.linalg.det(old[:, :2]):+.4f} new {np.linalg.det(M[:, :2]):+.4f}")
        print("   ", np.array2string(M, precision=6, suppress_small=False).replace("\n", "\n    "))

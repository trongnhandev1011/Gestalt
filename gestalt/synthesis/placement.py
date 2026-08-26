# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cell placement as a hard-core point process: positions drawn uniformly at random inside a
field, rejecting a draw that falls closer than a minimum separation to an already-placed cell.
This is what "positions keep a minimum separation set by the requested cell count" (the method
paragraph, sec/5_synthesis.tex) refers to.
"""
import numpy as np


def fit_min_separation(xy_per_tile):
    """Fit the minimum-separation constant from real tiles: for each tile, its nearest-neighbour
    spacing scales as roughly c / sqrt(n_cells) over a field of fixed area, so c is that
    spacing times sqrt(n_cells), averaged (median, robust to a few crowded/sparse tiles) across
    every real tile supplied.

    Args:
        xy_per_tile (list[np.ndarray]): one (n_i, 2) array of real cell positions per tile,
            already in the same normalized coordinate frame as the field to be generated.

    Returns:
        float: the fitted constant c, for use with `hardcore_positions(n, c=c)`.
    """
    consts = []
    for xy in xy_per_tile:
        n = len(xy)
        if n < 2:
            continue
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        consts.append(np.median(nn) * np.sqrt(n))
    return float(np.median(consts))


def hardcore_positions(n, c, bounds=(-1.0, 1.0), rng=None, max_tries_per_cell=200):
    """Sample n cell positions in a square field by sequential rejection.

    Args:
        n (int): number of cells to place.
        c (float): minimum-separation constant; the effective minimum distance for this draw
            is c / sqrt(n), so denser requests pack tighter, matching real tissue at higher
            cell counts (see `fit_min_separation`).
        bounds (tuple): (lo, hi) for both axes.
        rng (np.random.Generator, optional): source of randomness.
        max_tries_per_cell (int): rejection budget per cell before halving the separation and
            retrying, so a pathologically dense request still terminates.

    Returns:
        np.ndarray: (n, 2) positions.
    """
    rng = rng or np.random.default_rng()
    lo, hi = bounds
    min_sep = c / max(np.sqrt(n), 1e-6)
    pts = np.empty((0, 2))
    while len(pts) < n:
        placed_this_round = False
        for _ in range(max_tries_per_cell):
            cand = rng.uniform(lo, hi, size=2)
            if len(pts) == 0 or np.linalg.norm(pts - cand, axis=1).min() >= min_sep:
                pts = np.vstack([pts, cand])
                placed_this_round = True
                break
        if not placed_this_round:
            min_sep *= 0.5  # field is saturated at this spacing; relax and keep going
    return pts[:n].astype(np.float32)

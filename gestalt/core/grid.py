# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid geometry for the 1024px pipeline: a 1024px field is a 4x4 arrangement of 16 sub-tiles of
256px each. The readout's query positions and the field builder both read the layout from here.
"""
import numpy as np

GRID = 4
TILE = 256
NSUB = GRID * GRID
LXY_IS_XY = True  # local_xy stored as (x,y). If the convention check below ever fails, flip this.


def make_qpos():
    c = (np.arange(GRID) + 0.5) / GRID * 2 - 1.0                     # [-.75,-.25,.25,.75]
    order = [(dy, dx) for dy in range(GRID) for dx in range(GRID)]   # k = dy*GRID+dx (row-major, y outer)
    return np.array([[c[dx], c[dy]] if LXY_IS_XY else [c[dy], c[dx]] for dy, dx in order], np.float32)


QPOS = make_qpos()  # (16,2) fixed grid centers in [-1,1]


def offset(dy, dx):
    return np.array([dx * TILE, dy * TILE], np.float32) if LXY_IS_XY else np.array([dy * TILE, dx * TILE], np.float32)
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Objective 1: predict a frozen foundation model's tile embedding.

The shared trunk pools to one tile vector, which a de-standardisation head maps to the
foundation model's dimension. Trained by direct supervision against the real embedding.
"""

import torch
import torch.nn as nn

from ...core.backbone import GATLayer, GeneTokenEncoder, build_knn
from ...core.grid import QPOS
from ...core.readout import GridReadout, QueryReadout


class _DeStd(nn.Module):
    """Map the pooled tile vector into the target embedding space.

    Output weights are zero-initialised so the head starts as a no-op. When cfg.NUM_SLIDES is
    set, a learned per-slide additive term is applied in raw target space.

    Args:
        cfg (Config): Model configuration.
        num_slides (int, optional): Number of training slides. Enables the per-slide term.
    """
    def __init__(self, cfg, num_slides=None):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(cfg.HIDDEN), nn.Linear(cfg.HIDDEN, cfg.UNI_DIM))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('uni_mean', torch.zeros(cfg.UNI_DIM))
        self.register_buffer('uni_std', torch.ones(cfg.UNI_DIM))
        self.slide_bias = None
        if num_slides:
            self.slide_bias = nn.Embedding(num_slides, cfg.UNI_DIM)
            nn.init.zeros_(self.slide_bias.weight)

    def forward(self, h, slide_id=None):
        out = self.head(h) * self.uni_std + self.uni_mean
        if self.slide_bias is not None and slide_id is not None:
            out = out + self.slide_bias(slide_id)
        return out


class GeneTokenCells256(nn.Module):
    """Predict a foundation model's embedding for one 256px tile.

    Args:
        cfg (Config): Model configuration.
    """

    def __init__(self, cfg):
        super().__init__()
        H = cfg.HIDDEN
        self.enc = GeneTokenEncoder(cfg)
        self.pos = nn.Linear(2, H)
        self.blocks = nn.ModuleList([GATLayer(H, cfg.N_HEADS) for _ in range(cfg.N_BLOCKS)])
        self.read = QueryReadout(H, cfg.N_HEADS)
        self.dec = _DeStd(cfg, num_slides=getattr(cfg, 'NUM_SLIDES', None))
        self.pairbias_k = getattr(cfg, 'PAIRBIAS_K', 8)

    def forward(self, b):
        xy = b['xy']
        cm = b['cmask']
        x = self.enc(b['gene_ids'], b['gene_vals'], b['gmask'])
        # stash the per-cell encoder output so an auxiliary masked-gene-reconstruction head can
        # reuse it without a second forward pass. Read-only side effect; nothing else touches it.
        self._last_cells = x
        x = (x + self.pos(xy)) * cm[..., None]
        ni, ef, nv = build_knn(xy, cm, min(self.pairbias_k, xy.shape[1] - 1))
        for blk in self.blocks:
            x = blk(x, ni, ef, nv) * cm[..., None]
        r = self.read(x, xy, cm)
        # Expose the pre-head tile vector for analyses that operate on the backbone
        # representation, not the predicted embedding.
        self._last_tile = r
        return self.dec(r, slide_id=b.get('slide_id'))


class GeneTokenCells1024(nn.Module):
    """Predict 16 sub-region embeddings for one 1024px field.

    Args:
        cfg (Config): Model configuration.
    """

    def __init__(self, cfg):
        super().__init__()
        H = cfg.HIDDEN
        self.enc = GeneTokenEncoder(cfg)
        self.pos = nn.Linear(2, H)
        self.blocks = nn.ModuleList([GATLayer(H, cfg.N_HEADS) for _ in range(cfg.N_BLOCKS)])
        self.read = GridReadout(H, cfg.N_HEADS, QPOS, getattr(cfg, 'READOUT_DECAY', 0.0))
        self.dec = _DeStd(cfg, num_slides=getattr(cfg, 'NUM_SLIDES', None))  # runs per-query for free
        self.pairbias_k = getattr(cfg, 'PAIRBIAS_K', 8)

    def forward(self, b):
        xy = b['xy']
        cm = b['cmask']
        x = self.enc(b['gene_ids'], b['gene_vals'], b['gmask']) + self.pos(xy)
        x = x * cm[..., None]
        ni, ef, nv = build_knn(xy, cm, min(self.pairbias_k, xy.shape[1] - 1))
        for blk in self.blocks:
            x = blk(x, ni, ef, nv) * cm[..., None]
        return self.dec(self.read(x, xy, cm), slide_id=b.get('slide_id'))  # (B,16,1536)


# ---------------------------------------------------------------- flow-matching head (256 only)

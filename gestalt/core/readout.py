# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tile readouts.

QueryReadout pools all cells into one tile vector. GridReadout emits 16 sub-region
vectors for a 1024px field.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from .backbone import RelPosBias


class QueryReadout(nn.Module):
    """Pool every cell in a tile into a single vector with one learned query.

    Attention is unrestricted by k and carries its own relative-position bias.

    Args:
        dim (int): Token width.
        heads (int): Number of attention heads.
    """


    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.tok = nn.Parameter(torch.randn(dim) * 0.02)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.n = nn.LayerNorm(dim)
        self.bias = RelPosBias(heads)

    def forward(self, cell, cell_xy, cmask):
        B, L, D = cell.shape
        q = self.q(self.n(self.tok)[None, None].expand(B, 1, D)).view(B, 1, self.h, self.d).transpose(1, 2)
        kv = self.kv(cell).view(B, L, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        qxy = cell_xy.new_zeros(B, 1, 2)
        dd = qxy[:, :, None, :] - cell_xy[:, None, :, :]
        dist = dd.norm(dim=-1, keepdim=True)
        bias = self.bias.mlp(torch.cat([dd, dist], -1)).permute(0, 3, 1, 2)
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(self.d) + bias
        lg = lg.masked_fill(~cmask[:, None, None, :], -1e4)
        return (lg.softmax(-1) @ v).transpose(1, 2).reshape(B, D)


class GridReadout(nn.Module):
    """Pool cells into 16 sub-region vectors, one per 256px quadrant of a 1024px field.

    decay > 0 adds a per-head -gamma*||offset||^2 term to the attention logits. Without it the
    learned RelPosBias MLP is the only thing concentrating a query, and it localizes only
    partly, so the 16 conditioning vectors come out more alike than the 16 real sub-regions,
    blending appearance across the field. gamma is softplus-positive so it can only penalise
    distance, and it stays close to its initial value in practice.

    Args:
        dim (int): Token width.
        heads (int): Number of attention heads.
        qpos (np.ndarray): Query positions, shape (16, 2).
        decay (float): Initial per-head distance-penalty strength. 0 disables the penalty
            entirely (the parameter is then left absent so decay=0 checkpoints load strictly).
    """


    def __init__(self, dim, heads, qpos, decay=0.0):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.G = qpos.shape[0]
        self.tok = nn.Parameter(torch.randn(self.G, dim) * 0.02)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.n = nn.LayerNorm(dim)
        self.bias = RelPosBias(heads)
        self.use_decay = decay > 0
        if self.use_decay:
            # softplus^-1(decay), so the parameter starts at exactly the requested strength.
            self.log_decay = nn.Parameter(torch.full((heads,), math.log(math.expm1(decay))))
        self.register_buffer('qpos', torch.as_tensor(qpos, dtype=torch.float32))  # fixed, not learned

    def forward(self, cell, cell_xy, cmask):
        B, L, D = cell.shape
        q = self.q(self.n(self.tok)[None].expand(B, self.G, D)).view(B, self.G, self.h, self.d).transpose(1, 2)
        kv = self.kv(cell).view(B, L, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        qxy = self.qpos[None].expand(B, self.G, 2)
        dd = qxy[:, :, None, :] - cell_xy[:, None, :, :]
        dist = dd.norm(dim=-1, keepdim=True)
        bias = self.bias.mlp(torch.cat([dd, dist], -1)).permute(0, 3, 1, 2)  # (B,h,16,L)
        if self.use_decay:
            gamma = F.softplus(self.log_decay).view(1, -1, 1, 1)             # (1,h,1,1)
            bias = bias - gamma * dist.squeeze(-1)[:, None, :, :] ** 2       # (B,1,16,L)
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(self.d) + bias
        lg = lg.masked_fill(~cmask[:, None, None, :], -1e4)
        return (lg.softmax(-1) @ v).transpose(1, 2).reshape(B, self.G, D)  # (B,16,H)


# ---------------------------------------------------------------- gene-token encoder

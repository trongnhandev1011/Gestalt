# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared backbone: per-cell gene encoder and sparse k-NN graph attention.

Every conditioning objective uses this trunk unchanged; only the head and the loss differ.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


class RelPosBias(nn.Module):
    """Learned attention bias from the relative offset between two cells.

    Args:
        heads (int): Number of attention heads to produce a bias for.
        hid (int): Hidden width of the bias MLP.
    """

    def __init__(self, heads: int, hid: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hid), nn.GELU(), nn.Linear(hid, heads))

    def forward(self, xy):
        d = xy[:, :, None, :] - xy[:, None, :, :]
        dist = d.norm(dim=-1, keepdim=True)
        return self.mlp(torch.cat([d, dist], -1)).permute(0, 3, 1, 2)


def build_knn(xy: torch.Tensor, cmask: torch.Tensor, k: int) -> Tuple[torch.Tensor, ...]:
    """Build the k-nearest-neighbour graph over cell centroids.

    Args:
        xy (torch.Tensor): Cell coordinates, shape (B, L, 2).
        cmask (torch.Tensor): Valid-cell mask, shape (B, L).
        k (int): Neighbours per cell, excluding the cell itself.

    Returns:
        Neighbour indices (B, L, k+1), edge features (B, L, k+1, 3) holding the relative
        offset and its norm, and a neighbour validity mask (B, L, k+1).
    """
    B, L, _ = xy.shape
    d = torch.cdist(xy.float(), xy.float())
    BIG = 1e9
    d = d.masked_fill(~cmask[:, None, :], BIG).masked_fill(torch.eye(L, device=xy.device, dtype=torch.bool)[None], BIG)
    k_ = min(k, L - 1)
    _, ni = torch.topk(d, k_, dim=-1, largest=False)
    si = torch.arange(L, device=xy.device)[None, :, None].expand(B, L, 1)
    ni = torch.cat([si, ni], -1)
    bi = torch.arange(B, device=xy.device)[:, None, None]
    nv = cmask[bi, ni]
    rel = xy[bi, ni] - xy[:, :, None, :]
    dist = rel.norm(dim=-1, keepdim=True)
    return ni, torch.cat([rel, dist], -1), nv


class GATLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.edge = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, heads))
        self.o = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, ni, ef, nv):
        B, L, D = x.shape
        k = ni.shape[-1]
        h = self.n1(x)
        q, kf, vf = self.qkv(h).chunk(3, -1)
        bi = torch.arange(B, device=x.device)[:, None, None]
        kn = kf[bi, ni].view(B, L, k, self.h, self.d)
        vn = vf[bi, ni].view(B, L, k, self.h, self.d)
        q = q.view(B, L, self.h, self.d)
        sc = (q[:, :, None] * kn).sum(-1) / math.sqrt(self.d) + self.edge(ef)
        sc = sc.masked_fill(~nv[..., None], -1e4)
        out = ((sc.softmax(2))[..., None] * vn).sum(2).reshape(B, L, D)
        x = x + self.o(out)
        x = x + self.ff(self.n2(x))
        return x


class GeneSelfAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.n = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, gmask):
        M, Lg, D = x.shape
        h = self.n(x)
        q, k, v = self.qkv(h).view(M, Lg, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        sc = sc.masked_fill(~gmask[:, None, None, :], -1e4)
        out = (sc.softmax(-1) @ v).transpose(1, 2).reshape(M, Lg, D)
        x = x + self.o(out)
        x = x + self.ff(self.n2(x))
        return x


class GeneTokenEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.HIDDEN
        self.gene_emb = nn.Embedding(cfg.GENE_VOCAB + 1, H, padding_idx=cfg.GENE_VOCAB)
        self.val_proj = nn.Sequential(nn.Linear(1, H), nn.GELU(), nn.Linear(H, H))
        self.in_norm = nn.LayerNorm(H)
        self.layers = nn.ModuleList([GeneSelfAttn(H, cfg.N_HEADS) for _ in range(cfg.N_GENE_LAYERS)])
        self.pool_q = nn.Parameter(torch.randn(H) * 0.02)
        self.pq = nn.Linear(H, H, bias=False)
        self.pkv = nn.Linear(H, 2 * H, bias=False)
        self.pn = nn.LayerNorm(H)
        self.h = cfg.N_HEADS
        self.d = H // cfg.N_HEADS

    def forward(self, gene_ids, gene_vals, gmask):
        B, Lc, Lg = gene_ids.shape
        H = self.gene_emb.embedding_dim
        x = self.gene_emb(gene_ids) + self.val_proj(gene_vals.unsqueeze(-1))
        x = self.in_norm(x).view(B * Lc, Lg, H)
        gm = gmask.view(B * Lc, Lg)
        empty = ~gm.any(1)
        gm_safe = gm.clone()
        gm_safe[empty, 0] = True  # NaN guard for fully-padded cells
        for l in self.layers:
            x = l(x, gm_safe)
        q = self.pq(self.pn(self.pool_q)[None, None].expand(B * Lc, 1, H)).view(B * Lc, 1, self.h, self.d).transpose(1, 2)
        kv = self.pkv(x).view(B * Lc, Lg, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        lg = lg.masked_fill(~gm_safe[:, None, None, :], -1e4)
        cell = (lg.softmax(-1) @ v).transpose(1, 2).reshape(B * Lc, H).masked_fill(empty[:, None], 0.0)
        return cell.view(B, Lc, H)

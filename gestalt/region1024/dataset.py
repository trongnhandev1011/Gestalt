# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset for 1024px region synthesis."""
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.config import samples_tag


def _construct_field(positions, real_emb, Xs, cols, uni_dim):
    """Assemble one (per_cell, xy, tgt, tmask) field.

    Args:
        positions: {pos in 0..15: (window_idx, meta)} for the 4x4 layout.
    """
    per_cell, xy_list = [], []
    tgt = np.zeros((16, uni_dim), np.float32)
    tmask = np.zeros(16, bool)
    for pos, (wi, m) in positions.items():
        row, col = pos // 4, pos % 4
        tgt[pos] = real_emb[wi]
        tmask[pos] = True
        for ci, local in zip(m['cell_idx'], m['local_xy']):
            r = Xs.getrow(ci)
            if r.nnz == 0:
                continue
            gids, val = cols[r.indices], r.data.astype(np.float32)
            per_cell.append((gids, val))
            # place the subtile-local coordinate in the field's [0,1024) space
            xy_list.append([local[0] + col * 256, local[1] + row * 256])
    if not per_cell:
        return None
    xy = (np.array(xy_list, np.float32) / 1024.0) * 2 - 1
    return per_cell, xy, tgt, tmask


def _build_fields_random(cfg, G2I, adatas, val_frac=0.2, train_overlap_stride=2, test_overlap_stride=1):
    """Group a non-overlap 256px lattice into 1024px fields of 4x4 subtiles.

    Windows are split into train and val blocks, and each side slides a 4x4 window over its own
    blocks at `train_overlap_stride` / `test_overlap_stride`. A field is kept only when every
    window it covers belongs to that side, which keeps train and val tissue-disjoint; positions
    absent from the lattice are off-tissue and enter the field as tmask=False. cfg.OUT must
    point at a non-overlap lattice.
    """
    import h5py

    tr_fields, va_fields, ncell = [], [], []
    for s in cfg.SAMPLES:
        A = adatas[s]
        layer = A.layers[cfg.GENE_LAYER] if cfg.GENE_LAYER in A.layers else A.X
        keep = [i for i, g in enumerate(A.var_names) if str(g) in G2I]
        cols = np.array([G2I[str(A.var_names[i])] for i in keep], dtype=np.int64)
        Xs = layer.tocsr()[:, keep]

        meta = json.load(open(f'{cfg.OUT}/{s}_meta.json'))
        with h5py.File(f'{cfg.OUT}/{s}_tiles256_p5_uni.h5', 'r') as f:
            real_emb = f['uni'][:]  # (n_windows, 1536), same row order as meta

        by_pos = {}          # (gy,gx) -> (window_idx, meta)
        disjoint = {}         # (GY,GX) -> {pos: (wi,m)}, stride-4 partition
        for wi, m in enumerate(meta):
            gy, gx = m['gy'], m['gx']
            by_pos[(gy, gx)] = (wi, m)
            GY, GX = gy // 4, gx // 4
            pos = (gy % 4) * 4 + (gx % 4)
            disjoint.setdefault((GY, GX), {})[pos] = (wi, m)
        max_gy = max(m['gy'] for m in meta)
        max_gx = max(m['gx'] for m in meta)

        # split the disjoint (clean) blocks into train/val -- val NEVER gets densified.
        block_keys = list(disjoint.keys())
        rng = np.random.RandomState(cfg.SEED)
        perm = rng.permutation(len(block_keys))
        n_val = max(1, int(val_frac * len(block_keys)))
        val_keys = {block_keys[i] for i in perm[:n_val]}
        train_keys = {block_keys[i] for i in perm[n_val:]}

        val_windows = {(m['gy'], m['gx']) for k in val_keys for _, m in disjoint[k].values()}

        # val: dense slide at test_overlap_stride, keep a candidate only if NONE of its (up to
        # 16) real windows belongs to a train-assigned block (inverse of the train-side check).
        n_val_full = 0
        n_val_this_sample = 0
        for gy0 in range(0, max_gy + 1, test_overlap_stride):
            for gx0 in range(0, max_gx + 1, test_overlap_stride):
                positions, touches_train = {}, False
                for dy in range(4):
                    for dx in range(4):
                        key = (gy0 + dy, gx0 + dx)
                        if key in by_pos:
                            if key not in val_windows:
                                touches_train = True
                                break
                            positions[dy * 4 + dx] = by_pos[key]
                    if touches_train:
                        break
                if touches_train or not positions:
                    continue
                field = _construct_field(positions, real_emb, Xs, cols, cfg.UNI_DIM)
                if field is None:
                    continue
                va_fields.append(field)
                ncell.append(len(field[0]))
                n_val_this_sample += 1
                if len(positions) == 16:
                    n_val_full += 1

        # train: dense slide at stride train_overlap_stride over the full grid, keep a candidate
        # only if none of its (up to 16) windows belong to a val block.
        n_disjoint_train = 0
        n_slide_train = 0
        for gy0 in range(0, max_gy + 1, train_overlap_stride):
            for gx0 in range(0, max_gx + 1, train_overlap_stride):
                positions, touches_val = {}, False
                for dy in range(4):
                    for dx in range(4):
                        key = (gy0 + dy, gx0 + dx)
                        if key in val_windows:
                            touches_val = True
                            break
                        if key in by_pos:
                            positions[dy * 4 + dx] = by_pos[key]
                    if touches_val:
                        break
                if touches_val or not positions:
                    continue
                field = _construct_field(positions, real_emb, Xs, cols, cfg.UNI_DIM)
                if field is None:
                    continue
                tr_fields.append(field)
                ncell.append(len(field[0]))
                n_slide_train += 1
                if (gy0 // 4, gx0 // 4) in train_keys and gy0 % 4 == 0 and gx0 % 4 == 0:
                    n_disjoint_train += 1

        print(f'  {s}: {len(meta)} lattice windows | val: {len(val_keys)} disjoint blocks worth of '
              f'tissue, densified by stride-{test_overlap_stride} sliding into {n_val_this_sample} '
              f'fields ({n_val_full} fully-covered 16/16, {n_val_this_sample - n_val_full} partial) '
              f'| train: {n_slide_train} fields from stride-{train_overlap_stride} sliding '
              f'({n_disjoint_train} of which are the original disjoint blocks)')

    return tr_fields, va_fields, np.array(ncell)


def _build_fields(cfg, G2I, adatas):
    raise NotImplementedError(
        'Spatial-split field construction not implemented -- only random-split is used for the '
        '1024px lattice work so far. Same building blocks as _build_fields_random, just needs a '
        'per-sample spatial holdout instead of a pooled random permutation.'
    )


class TileDS1024(Dataset):
    def __init__(self, samples, augment, cfg, gene_mu, gene_sd, PAD):
        self.s, self.aug, self.cfg = samples, augment, cfg
        self.gene_mu, self.gene_sd, self.PAD = gene_mu, gene_sd, PAD

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        cfg = self.cfg
        per_cell, xy, tgt, tmask = self.s[i]
        N = cfg.N_MAX_CELLS_1024
        na = len(per_cell)
        if na > N:
            rng = np.random.default_rng() if self.aug else np.random.default_rng(i)
            sel = rng.choice(na, N, replace=False)
            per_cell = [per_cell[j] for j in sel]
            xy = xy[sel]
        n = min(len(per_cell), N)
        if self.aug:
            xy = xy + np.random.randn(*xy.shape) * cfg.AUG_JITTER * 2
        GID = np.full((N, cfg.K_MAX_GENES), self.PAD, np.int64)
        VAL = np.zeros((N, cfg.K_MAX_GENES), np.float32)
        GM = np.zeros((N, cfg.K_MAX_GENES), bool)
        XY = np.zeros((N, 2), np.float32)
        CM = np.zeros(N, bool)
        for c in range(n):
            gids, val = per_cell[c]
            if self.aug and len(gids) > cfg.AUG_MINKEEP:
                keep = np.random.rand(len(gids)) >= np.random.uniform(0, cfg.AUG_DROP)
                if keep.sum() >= cfg.AUG_MINKEEP:
                    gids, val = gids[keep], val[keep]
            k = min(len(gids), cfg.K_MAX_GENES)
            srt = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
            GID[c, :k] = gids[srt]
            VAL[c, :k] = (val[srt] - self.gene_mu[gids[srt]]) / self.gene_sd[gids[srt]]
            GM[c, :k] = True
        XY[:n] = xy[:n]
        CM[:n] = True
        return {'gene_ids': torch.from_numpy(GID), 'gene_vals': torch.from_numpy(VAL), 'gmask': torch.from_numpy(GM),
                'xy': torch.from_numpy(XY), 'cmask': torch.from_numpy(CM),
                'tgt': torch.from_numpy(tgt), 'tmask': torch.from_numpy(tmask)}


def load_or_build_1024(cfg, G2I, gene_mu, gene_sd, rank, is_ddp, barrier_fn,
                        train_overlap_stride=2, test_overlap_stride=1):
    """Build the field cache on rank 0 if it is absent, then load it on every rank.

    The cache filename encodes cfg.TEST_SAMPLE, the training slide set, and both stride
    settings, so each configuration keeps its own file.
    """
    loo_tag = f'_loo_{cfg.TEST_SAMPLE}' if getattr(cfg, 'TEST_SAMPLE', None) else ''
    samp_tag = samples_tag(cfg)
    ov_tag = '' if train_overlap_stride == 2 else f'_ov{train_overlap_stride}'
    test_tag = '' if test_overlap_stride == 4 else f'_test{test_overlap_stride}'
    cache_path = f'{cfg.OUT}/fields_1024_v1{loo_tag}{samp_tag}{ov_tag}{test_tag}.pkl'
    if rank == 0 and not os.path.exists(cache_path):
        import anndata as ad
        adatas = {s: ad.read_h5ad(f'{cfg.OUT}/{s}_adata.h5ad') for s in cfg.SAMPLES}
        if True:
            tr1024, va1024, ncell = _build_fields_random(cfg, G2I, adatas,
                                                           train_overlap_stride=train_overlap_stride,
                                                           test_overlap_stride=test_overlap_stride)
        else:
            tr1024, va1024, ncell = _build_fields(cfg, G2I, adatas)
        cfg.N_MAX_CELLS_1024 = int(np.quantile(ncell, 0.95))
        pickle.dump({'tr1024': tr1024, 'va1024': va1024, 'ncell': ncell, 'N_MAX_CELLS_1024': cfg.N_MAX_CELLS_1024},
                    open(cache_path, 'wb'), protocol=4)
        print(f'[rank0] built + cached 1024: train {len(tr1024)} val {len(va1024)}')
    barrier_fn(is_ddp)

    d7 = pickle.load(open(cache_path, 'rb'))
    tr1024, va1024 = d7['tr1024'], d7['va1024']
    cfg.N_MAX_CELLS_1024 = d7['N_MAX_CELLS_1024']
    PAD = cfg.GENE_VOCAB
    train_ds = TileDS1024(tr1024, True, cfg, gene_mu, gene_sd, PAD)
    val_ds = TileDS1024(va1024, False, cfg, gene_mu, gene_sd, PAD)
    return train_ds, val_ds, va1024

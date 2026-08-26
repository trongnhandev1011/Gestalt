# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tile dataset and the train/val split cache for the 256px experiments.

The cache is keyed by (pool, conditioning signal, gene vocab) and is built once,
on first use. Let one job create it before launching others that share the same key.
"""
import json
import os
import pickle

import anndata as ad
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ..core.config import samples_tag, fm_tag, fm_spec
from ..core.vocab import build_vocab

CACHE_DIR = os.environ.get('GESTALT_OUT', './data/tiles')
os.makedirs(CACHE_DIR, exist_ok=True)


def fm_h5(cfg, sample):
    """Path + in-file dataset key for sample's target-embedding cache, per cfg.FM."""
    fm = getattr(cfg, 'FM', 'uni2h')
    spec = fm_spec(fm)
    return f'{cfg.OUT}/{sample}_tiles256_p5_{spec["suffix"]}.h5', spec['h5_key']


def sample_sparse(A, cfg, G2I):
    layer = A.layers[cfg.GENE_LAYER] if cfg.GENE_LAYER in A.layers else A.X
    keep = [i for i, g in enumerate(A.var_names) if str(g) in G2I]
    cols = np.array([G2I[str(A.var_names[i])] for i in keep], dtype=np.int64)
    return layer.tocsr()[:, keep], cols


def explode_sparse(metas, Xs, cols, UNI, grid2row, accum, sum_g, sumsq_g, cnt_g):
    out = []
    for d in metas:
        row = grid2row.get((d['gy'], d['gx']))
        if row is None:
            continue
        ci = np.asarray(d['cell_idx'], np.int64)
        xy = (np.asarray(d['local_xy'], np.float32) / 256.0) * 2 - 1
        per_cell = []
        for c in ci:
            r = Xs.getrow(c)
            gids = cols[r.indices]
            val = r.data.astype(np.float32)
            per_cell.append((gids, val))
            if accum:
                np.add.at(sum_g, gids, val)
                np.add.at(sumsq_g, gids, val * val)
                np.add.at(cnt_g, gids, 1)
        out.append((per_cell, xy.astype(np.float32), UNI[row].astype(np.float32)))
    return out


def _build_256_cache(cfg, G2I):
    sum_g = np.zeros(cfg.GENE_VOCAB)
    sumsq_g = np.zeros(cfg.GENE_VOCAB)
    cnt_g = np.zeros(cfg.GENE_VOCAB)
    tr_s, va_s = [], []
    tr_slide, va_slide = [], []  # parallel arrays: which cfg.SAMPLES index each record came from
    train_samples = [s for s in cfg.SAMPLES if s != getattr(cfg, 'TEST_SAMPLE', None)]
    for s in train_samples:
        A = ad.read_h5ad(f'{cfg.OUT}/{s}_adata.h5ad')
        Xs, cols = sample_sparse(A, cfg, G2I)
        meta = json.load(open(f'{cfg.OUT}/{s}_meta.json'))
        if cfg.CAP_TILES and len(meta) > cfg.CAP_TILES:
            meta = list(np.random.RandomState(cfg.SPLIT_SEED).permutation(meta))[:cfg.CAP_TILES]
        uni_path, uni_key = fm_h5(cfg, s)
        with h5py.File(uni_path, 'r') as g:
            UNI = g[uni_key][:]
        with h5py.File(f'{cfg.OUT}/{s}_tiles256_p5.h5', 'r') as g:
            coords = g['coords'][:]
            nat = int(g.attrs['native'])
        grid2row = {(int(y0 // nat), int(x0 // nat)): i for i, (y0, x0, _) in enumerate(coords)}
        x0s = np.array([d['x0'] for d in meta])
        thr = np.quantile(x0s, 1 - cfg.CV_HOLDOUT_FRAC)
        sidx = cfg.SAMPLES.index(s)
        tr_chunk = explode_sparse([d for d in meta if d['x0'] < thr], Xs, cols, UNI, grid2row, True, sum_g, sumsq_g, cnt_g)
        va_chunk = explode_sparse([d for d in meta if d['x0'] >= thr], Xs, cols, UNI, grid2row, False, sum_g, sumsq_g, cnt_g)
        tr_s += tr_chunk
        va_s += va_chunk
        tr_slide += [sidx] * len(tr_chunk)
        va_slide += [sidx] * len(va_chunk)
    gene_mu = (sum_g / np.maximum(cnt_g, 1)).astype(np.float32)
    gene_sd = np.sqrt(np.maximum(sumsq_g / np.maximum(cnt_g, 1) - gene_mu ** 2, 1e-6)).astype(np.float32)
    K_MAX_GENES = 87

    # Target-embedding statistics and N_MAX_CELLS are computed from the training fold only.
    ut = np.stack([s[2] for s in tr_s])
    umu = ut.mean(0).astype(np.float32)
    usd = ut.std(0).clip(1e-6).astype(np.float32)
    n_max_cells = int(max(len(s[0]) for s in tr_s)) + 8

    return (tr_s, va_s, gene_mu, gene_sd, K_MAX_GENES, umu, usd, n_max_cells,
            np.array(tr_slide, np.int64), np.array(va_slide, np.int64))


def _build_256_cache_random(cfg, G2I, val_frac=0.2):
    """Pool every tile across all samples and take one seeded random 80/20 split, with no
    spatial or slide awareness."""
    sum_g = np.zeros(cfg.GENE_VOCAB)
    sumsq_g = np.zeros(cfg.GENE_VOCAB)
    cnt_g = np.zeros(cfg.GENE_VOCAB)
    all_records = []
    all_slide = []
    train_samples = [s for s in cfg.SAMPLES if s != getattr(cfg, 'TEST_SAMPLE', None)]
    for s in train_samples:
        A = ad.read_h5ad(f'{cfg.OUT}/{s}_adata.h5ad')
        Xs, cols = sample_sparse(A, cfg, G2I)
        meta = json.load(open(f'{cfg.OUT}/{s}_meta.json'))
        if cfg.CAP_TILES and len(meta) > cfg.CAP_TILES:
            meta = list(np.random.RandomState(cfg.SPLIT_SEED).permutation(meta))[:cfg.CAP_TILES]
        uni_path, uni_key = fm_h5(cfg, s)
        with h5py.File(uni_path, 'r') as g:
            UNI = g[uni_key][:]
        with h5py.File(f'{cfg.OUT}/{s}_tiles256_p5.h5', 'r') as g:
            coords = g['coords'][:]
            nat = int(g.attrs['native'])
        grid2row = {(int(y0 // nat), int(x0 // nat)): i for i, (y0, x0, _) in enumerate(coords)}
        # accum=False: gene stats get computed after the split, from the train portion only
        chunk = explode_sparse(meta, Xs, cols, UNI, grid2row, False, sum_g, sumsq_g, cnt_g)
        all_records += chunk
        all_slide += [cfg.SAMPLES.index(s)] * len(chunk)

    n = len(all_records)
    perm = np.random.RandomState(cfg.SPLIT_SEED).permutation(n)
    n_train = int((1 - val_frac) * n)
    tr_s = [all_records[i] for i in perm[:n_train]]
    va_s = [all_records[i] for i in perm[n_train:]]
    tr_slide = [all_slide[i] for i in perm[:n_train]]
    va_slide = [all_slide[i] for i in perm[n_train:]]

    for per_cell, _, _ in tr_s:
        for gids, val in per_cell:
            np.add.at(sum_g, gids, val)
            np.add.at(sumsq_g, gids, val * val)
            np.add.at(cnt_g, gids, 1)
    gene_mu = (sum_g / np.maximum(cnt_g, 1)).astype(np.float32)
    gene_sd = np.sqrt(np.maximum(sumsq_g / np.maximum(cnt_g, 1) - gene_mu ** 2, 1e-6)).astype(np.float32)
    K_MAX_GENES = 87

    ut = np.stack([s[2] for s in tr_s])
    umu = ut.mean(0).astype(np.float32)
    usd = ut.std(0).clip(1e-6).astype(np.float32)
    n_max_cells = int(max(len(s[0]) for s in tr_s)) + 8

    return (tr_s, va_s, gene_mu, gene_sd, K_MAX_GENES, umu, usd, n_max_cells,
            np.array(tr_slide, np.int64), np.array(va_slide, np.int64))


class TileDS(Dataset):
    def __init__(self, samples, augment, cfg, gene_mu, gene_sd, PAD, slide_ids=None):
        self.s, self.aug, self.cfg = samples, augment, cfg
        self.gene_mu, self.gene_sd, self.PAD = gene_mu, gene_sd, PAD
        self.slide_ids = slide_ids  # optional (N,) int array, which cfg.SAMPLES index per record

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        cfg = self.cfg
        per_cell, xy, tgt = self.s[i]
        n = min(len(per_cell), cfg.N_MAX_CELLS)
        if self.aug:
            xy = xy + np.random.randn(*xy.shape) * cfg.AUG_JITTER * 2
        GID = np.full((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), self.PAD, np.int64)
        VAL = np.zeros((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), np.float32)
        GM = np.zeros((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), bool)
        XY = np.zeros((cfg.N_MAX_CELLS, 2), np.float32)
        CM = np.zeros(cfg.N_MAX_CELLS, bool)
        for c in range(n):
            gids, val = per_cell[c]
            if self.aug and len(gids) > cfg.AUG_MINKEEP:
                keep = np.random.rand(len(gids)) >= np.random.uniform(0, cfg.AUG_DROP)
                if keep.sum() >= cfg.AUG_MINKEEP:
                    gids, val = gids[keep], val[keep]
            k = min(len(gids), cfg.K_MAX_GENES)
            sel = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
            GID[c, :k] = gids[sel]
            VAL[c, :k] = (val[sel] - self.gene_mu[gids[sel]]) / self.gene_sd[gids[sel]]
            GM[c, :k] = True
        XY[:n] = xy[:n]
        CM[:n] = True
        out = {'gene_ids': torch.from_numpy(GID), 'gene_vals': torch.from_numpy(VAL), 'gmask': torch.from_numpy(GM),
               'xy': torch.from_numpy(XY), 'cmask': torch.from_numpy(CM), 'tgt': torch.from_numpy(tgt)}
        if self.slide_ids is not None:
            out['slide_id'] = torch.tensor(self.slide_ids[i], dtype=torch.long)
        return out


def load_or_build_256(cfg, G2I, rank, is_ddp, barrier_fn):
    """Build the tile cache on rank 0 if it is absent, then load it on every rank.

    gene_mu, gene_sd, umu, usd and N_MAX_CELLS are computed inside the builder from the training
    tiles of this configuration, so they are train-only by construction.
    """
    loo_tag = f'_loo_{cfg.TEST_SAMPLE}' if getattr(cfg, 'TEST_SAMPLE', None) else ''
    samp_tag = samples_tag(cfg)
    split_seed = getattr(cfg, 'SPLIT_SEED', cfg.SEED)
    cache_path = (f'{cfg.OUT}/tr_va_256_v1_random_split{split_seed}{loo_tag}{samp_tag}'
                  f'{fm_tag(cfg)}_v{cfg.GENE_VOCAB}_sid2.pkl')
    if rank == 0 and not os.path.exists(cache_path):
        builder = _build_256_cache_random
        tr_s, va_s, gene_mu, gene_sd, K_MAX_GENES, umu, usd, n_max_cells, tr_slide, va_slide = builder(cfg, G2I)
        pickle.dump({'tr_s': tr_s, 'va_s': va_s, 'gene_mu': gene_mu, 'gene_sd': gene_sd,
                     'K_MAX_GENES': K_MAX_GENES, 'umu': umu, 'usd': usd,
                     'N_MAX_CELLS': n_max_cells, 'tr_slide': tr_slide, 'va_slide': va_slide,
                     'split_seed': split_seed},
                    open(cache_path, 'wb'), protocol=4)
        print(f'[rank0] built + cached 256: train {len(tr_s)} val {len(va_s)}')
    barrier_fn(is_ddp)

    d6 = pickle.load(open(cache_path, 'rb'))
    if d6.get('split_seed') != split_seed:
        raise RuntimeError(f'cache split seed {d6.get("split_seed")} does not match requested '
                           f'--split-seed {split_seed}: {cache_path}')
    tr_s, va_s, gene_mu, gene_sd = d6['tr_s'], d6['va_s'], d6['gene_mu'], d6['gene_sd']
    cfg.K_MAX_GENES = d6['K_MAX_GENES']
    cfg.N_MAX_CELLS = d6['N_MAX_CELLS']
    umu = torch.tensor(d6['umu'])
    usd = torch.tensor(d6['usd'])
    PAD = cfg.GENE_VOCAB
    train_ds = TileDS(tr_s, True, cfg, gene_mu, gene_sd, PAD, slide_ids=d6['tr_slide'])
    val_ds = TileDS(va_s, False, cfg, gene_mu, gene_sd, PAD, slide_ids=d6['va_slide'])
    return train_ds, val_ds, va_s, umu, usd


def load_test_sample_256(cfg, G2I, gene_mu, gene_sd, sample, rank=0, is_ddp=False, barrier_fn=lambda is_ddp: None):
    """All tiles of one sample, held out entirely from training -- for genuine leave-one-slide-out
    evaluation. Distinct from the intra-sample val split (`va_s` above) used for checkpoint
    selection during training: that val split still comes from samples the model trained on.
    """
    exp = os.path.basename(cfg.OUT.rstrip('/'))
    cache_path = f'{CACHE_DIR}/test_{sample}_{exp}_v{cfg.GENE_VOCAB}_256_v1{fm_tag(cfg)}.pkl'
    if rank == 0 and not os.path.exists(cache_path):
        sum_g = np.zeros(cfg.GENE_VOCAB)
        sumsq_g = np.zeros(cfg.GENE_VOCAB)
        cnt_g = np.zeros(cfg.GENE_VOCAB)
        A = ad.read_h5ad(f'{cfg.OUT}/{sample}_adata.h5ad')
        Xs, cols = sample_sparse(A, cfg, G2I)
        meta = json.load(open(f'{cfg.OUT}/{sample}_meta.json'))
        uni_path, uni_key = fm_h5(cfg, sample)
        with h5py.File(uni_path, 'r') as g:
            UNI = g[uni_key][:]
        with h5py.File(f'{cfg.OUT}/{sample}_tiles256_p5.h5', 'r') as g:
            coords = g['coords'][:]
            nat = int(g.attrs['native'])
        grid2row = {(int(y0 // nat), int(x0 // nat)): i for i, (y0, x0, _) in enumerate(coords)}
        # accum=False: this sample's gene stats must never leak into gene_mu/gene_sd
        test_s = explode_sparse(meta, Xs, cols, UNI, grid2row, False, sum_g, sumsq_g, cnt_g)
        pickle.dump({'test_s': test_s}, open(cache_path, 'wb'), protocol=4)
        print(f'[rank0] built + cached test sample {sample}: {len(test_s)} tiles')
    barrier_fn(is_ddp)

    d = pickle.load(open(cache_path, 'rb'))
    return TileDS(d['test_s'], False, cfg, gene_mu, gene_sd, cfg.GENE_VOCAB)

def load_vocab(cfg):
    """Build the gene vocabulary from the training slides and set cfg.GENE_VOCAB.

    The vocabulary is the union of gene panels over training slides only, so it depends on
    (cfg.OUT, cfg.SAMPLES, cfg.TEST_SAMPLE). cfg.GENE_VOCAB sizes the gene embedding table, so
    a model constructed after this call gets the right size.

    Returns:
        (GENES, G2I): the ordered gene list and its name -> index map.
    """
    GENES, G2I = build_vocab(cfg)
    cfg.GENE_VOCAB = len(GENES)
    cfg.GENES = GENES  # exposes gene identity to models that need it (e.g. external gene-set priors)
    return GENES, G2I

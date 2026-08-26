# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gene-importance evaluation."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.stats import hypergeom

from gestalt.core.config import make_cfg_256
from gestalt.models.mapper.model import GeneTokenCells256
from gestalt.tile256.dataset import sample_sparse, load_or_build_256

dev = 'cuda'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--samples', required=True, help='comma-separated sample names to score')
    p.add_argument('--ckpt', required=True, nargs='+',
                   help='one checkpoint (scored on every sample) or one per sample, in order')
    p.add_argument('--out', default='./data/tiles', help='tile/adata directory')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--n-tiles', type=int, default=300)
    p.add_argument('--topk', type=int, nargs='+', default=[50, 100])
    return p.parse_args()


def _load(ckpt_path, OUT):
    ck_d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = make_cfg_256(seed=0, fm='uni2h')
    for k, v in ck_d['cfg'].items():
        setattr(cfg, k, v)
    cfg.OUT = OUT
    GENES = ck_d['genes']
    return ck_d, cfg, GENES, {g: i for i, g in enumerate(GENES)}


def build_batch(d, Xs, cols, gene_mu, gene_sd, Lc, Lg, PAD):
    ci = np.asarray(d['cell_idx'])
    xy = (np.asarray(d['local_xy'], np.float32) / 256.0) * 2 - 1
    n = min(len(ci), Lc)
    GID = np.full((1, Lc, Lg), PAD, np.int64)
    VAL = np.zeros((1, Lc, Lg), np.float32)
    GM = np.zeros((1, Lc, Lg), bool)
    XY = np.zeros((1, Lc, 2), np.float32)
    CM = np.zeros((1, Lc), bool)
    for c in range(n):
        r = Xs.getrow(ci[c])
        gids, val = cols[r.indices], r.data.astype(np.float32)
        k = min(len(gids), Lg)
        sel = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
        GID[0, c, :k] = gids[sel]
        VAL[0, c, :k] = (val[sel] - gene_mu[gids[sel]]) / gene_sd[gids[sel]]
        GM[0, c, :k] = True
    XY[0, :n] = xy[:n]
    CM[0, :n] = True
    return GID, VAL, GM, XY, CM, n


def compute_importance(ckpt_path, sample, OUT, n_tiles, tile_seed=0):
    ck_d, cfg, GENES, G2I = _load(ckpt_path, OUT)
    model = GeneTokenCells256(cfg).to(dev).eval()
    model.load_state_dict(ck_d['model'])

    A = ad.read_h5ad(f'{OUT}/{sample}_adata.h5ad')
    Xs, cols = sample_sparse(A, cfg, G2I)
    meta = json.load(open(f'{OUT}/{sample}_meta.json'))
    train_ds, _, _, _, _ = load_or_build_256(cfg, G2I, 0, False, lambda i: None)
    gene_mu, gene_sd = train_ds.gene_mu, train_ds.gene_sd
    PAD, Lc, Lg = cfg.GENE_VOCAB, cfg.N_MAX_CELLS, cfg.K_MAX_GENES

    rng = np.random.RandomState(tile_seed)
    cand = [d for d in meta if len(d['cell_idx']) > 0]
    picked = rng.choice(len(cand), size=min(n_tiles, len(cand)), replace=False)

    isum = np.zeros(len(GENES), np.float64)
    icnt = np.zeros(len(GENES), np.float64)
    for idx in picked:
        GID, VAL, GM, XY, CM, _ = build_batch(cand[idx], Xs, cols, gene_mu, gene_sd, Lc, Lg, PAD)
        val_t = torch.from_numpy(VAL).to(dev).requires_grad_(True)
        batch = {'gene_ids': torch.from_numpy(GID).to(dev), 'gene_vals': val_t,
                 'gmask': torch.from_numpy(GM).to(dev), 'xy': torch.from_numpy(XY).to(dev),
                 'cmask': torch.from_numpy(CM).to(dev)}
        loss = (model(batch)[0] ** 2).sum()
        grad = torch.autograd.grad(loss, val_t)[0][0].detach().cpu().numpy()
        np.add.at(isum, GID[0][GM[0]], np.abs(grad[GM[0]]))
        np.add.at(icnt, GID[0][GM[0]], 1)

    icnt[icnt == 0] = 1
    return GENES, isum / icnt


def main():
    a = parse_args()
    samples = a.samples.split(',')
    if len(a.ckpt) not in (1, len(samples)):
        raise SystemExit(f'--ckpt must be one checkpoint or one per sample '
                         f'({len(samples)}), got {len(a.ckpt)}')
    per_sample = len(a.ckpt) == len(samples)
    Path(a.save_dir).mkdir(parents=True, exist_ok=True)

    ranks, panels = {}, {}
    for i, s in enumerate(samples):
        ck = a.ckpt[i] if per_sample else a.ckpt[0]
        print(f'=== importance on {s} ===', flush=True)
        GENES, imp = compute_importance(ck, s, a.out, a.n_tiles)
        order = np.argsort(-imp)
        ranks[s] = [GENES[j] for j in order]
        panels[s] = set(GENES)
        print(f'  top-10 = {ranks[s][:10]}', flush=True)
        json.dump({'genes': ranks[s], 'importance': imp[order].tolist()},
                  open(f'{a.save_dir}/{s}_ranking.json', 'w'))

    background = set.intersection(*panels.values())
    N = len(background)
    print(f'background: {N} genes')

    out = {'n_background_genes': int(N), 'pairs': []}
    for A_, B_ in combinations(samples, 2):
        ra = [g for g in ranks[A_] if g in background]
        rb = [g for g in ranks[B_] if g in background]
        for K in a.topk:
            if K > N:
                continue
            ov = len(set(ra[:K]) & set(rb[:K]))
            p = hypergeom.sf(ov - 1, N, K, K)
            chance = 100.0 * K / N
            print(f'{A_} vs {B_}, top-{K}: overlap={ov}/{K} ({100*ov/K:.0f}%), '
                  f'chance={chance:.0f}%, p={p:.2e}', flush=True)
            out['pairs'].append({'a': A_, 'b': B_, 'k': K, 'overlap': int(ov),
                                 'pct': 100.0 * ov / K, 'chance_pct': chance,
                                 'pvalue': float(p)})
    json.dump(out, open(f'{a.save_dir}/summary.json', 'w'), indent=2)
    print(f'\nsaved -> {a.save_dir}/summary.json')


if __name__ == '__main__':
    main()

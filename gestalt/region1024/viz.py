#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Region-synthesis visualisation (1024px fields).
Driven through viz.py; not called directly.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import filters, measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ..core.config import make_cfg_1024
from ..models.mapper import GeneTokenCells1024
from ..tile256.dataset import load_vocab, load_or_build_256
from .dataset import load_or_build_1024
from ..core.grid import GRID, TILE, NSUB, offset


def _no_barrier(is_ddp):
    pass


def to_np(t):
    return (t.numpy().transpose(1, 2, 0) * 255).astype('uint8')


def nuc_dia(rgb):
    h = rgb.astype(float)[:, :, 2] - rgb.astype(float)[:, :, 0]
    try:
        m = h > filters.threshold_otsu(h)
    except Exception:
        return np.nan
    ar = [p.area for p in measure.regionprops(measure.label(m)) if 20 < p.area < 2000]
    return np.sqrt(np.median(ar) / np.pi) * 2 if ar else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample', default='TENX117')
    ap.add_argument('--n-show', type=int, default=20)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=1.5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pixcell-lora', default=None,
                     help='PixCell-1024 LoRA adapter directory; omit for the base decoder')
    ap.add_argument('--test-sample', default=None,
                     help='slide the checkpoint was trained holding out')
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    cfg = make_cfg_1024()
    cfg.TEST_SAMPLE = args.test_sample
    GENES, G2I = load_vocab(cfg)
    # load_or_build_256 returns (train_ds, val_ds, va_s, umu, usd) -- only the train_ds's
    # gene_mu/gene_sd are needed here (the model's dec.uni_mean/uni_std come from the checkpoint).
    train256, _, _, _, _ = load_or_build_256(cfg, G2I, rank=0, is_ddp=False,
                                             barrier_fn=_no_barrier)
    gene_mu, gene_sd, PAD = train256.gene_mu, train256.gene_sd, cfg.GENE_VOCAB
    # sets cfg.N_MAX_CELLS_1024 from the (split-tagged) fields cache
    load_or_build_1024(cfg, G2I, gene_mu, gene_sd, rank=0, is_ddp=False, barrier_fn=_no_barrier)
    loo_tag = f'_loo_{args.test_sample}' if args.test_sample else ''
    ck_path = Path(cfg.CKPT) / f'pooled_genetoken1024_g{cfg.N_GENE_LAYERS}_s{args.seed}{loo_tag}.pt'
    ck = torch.load(ck_path, map_location='cpu')
    model = GeneTokenCells1024(cfg).to(dev).eval()
    model.load_state_dict(ck['model'])
    print(f"loaded {ck_path}")
    print(f"genetoken1024 g{cfg.N_GENE_LAYERS}  own_cos@save={ck['own_cos']:.4f}  ep={ck['ep']}")

    from diffusers import DiffusionPipeline, AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                         torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-1024', vae=vae,
                                             custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                             torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    if args.pixcell_lora:
        from peft import PeftModel
        pix.transformer = PeftModel.from_pretrained(pix.transformer, args.pixcell_lora).to(dev)
        # if a fp16/fp32 mismatch surfaces during generation, append .half() to the line above
        print(f'loaded PixCell-1024 LoRA adapter: {args.pixcell_lora}')
    pix.set_progress_bar_config(disable=True)

    def _uncond(n_tok):
        u = pix.get_unconditional_embedding(1)
        return u if u.shape[1] == n_tok else u[:, :1].expand(-1, n_tok, -1).contiguous()

    @torch.no_grad()
    def gen1024(uni_16):
        cond = uni_16.to(pix.device).to(pix.dtype).unsqueeze(0)
        unc = _uncond(NSUB)
        with torch.amp.autocast('cuda'):
            return pix(uni_embeds=cond, negative_uni_embeds=unc, num_inference_steps=args.steps,
                        guidance_scale=args.guidance, output_type='pt').images[0].float().cpu().clamp(0, 1)

    A = ad.read_h5ad(f'{cfg.OUT}/{args.sample}_adata.h5ad')
    layer = A.layers[cfg.GENE_LAYER] if cfg.GENE_LAYER in A.layers else A.X
    Xs = layer.tocsr()
    cols = np.array([G2I[g] for g in A.var_names])
    meta = json.load(open(f'{cfg.OUT}/{args.sample}_meta.json'))
    meta_by_gt = {(d['gy'], d['gx']): d for d in meta}
    with h5py.File(f'{cfg.OUT}/{args.sample}_tiles256_p5_uni.h5', 'r') as g:
        UNI_s = g['uni'][:]
    with h5py.File(f'{cfg.OUT}/{args.sample}_tiles256_p5.h5', 'r') as g:
        coords = g['coords'][:]
        nat = int(g.attrs['native'])
    grid2row_s = {(int(y0 // nat), int(x0 // nat)): i for i, (y0, x0, _) in enumerate(coords)}

    fields = {}
    for (gy, gx) in meta_by_gt:
        fields.setdefault((gy // GRID, gx // GRID), []).append((gy, gx))
    eval_holdout = max(cfg.CV_HOLDOUT_FRAC, 0.40)
    FXs = np.array([FX for (_, FX) in fields])
    thr = np.quantile(FXs, 1 - eval_holdout)
    full_va = []
    for (FY, FX), members in fields.items():
        if FX < thr:
            continue
        present = [(gy, gx) for (gy, gx) in members if grid2row_s.get((gy, gx)) is not None]
        if len(present) == NSUB:
            full_va.append((FY, FX, present))
    nshow = min(args.n_show, len(full_va))
    sel = np.random.default_rng(0).choice(len(full_va), nshow, replace=False)
    print(f'full 16/16 val fields: {len(full_va)}  showing {nshow}')

    def make_field_batch(FY, FX, present):
        per_cell = []
        xy_l = []
        for (gy, gx) in present:
            dy, dx = gy - GRID * FY, gx - GRID * FX
            d = meta_by_gt[(gy, gx)]
            ci = np.asarray(d['cell_idx'], np.int64)
            fxy = np.asarray(d['local_xy'], np.float32) + offset(dy, dx)
            for j, c in enumerate(ci):
                r = Xs.getrow(c)
                per_cell.append((cols[r.indices], r.data.astype(np.float32)))
                xy_l.append(fxy[j])
        N = cfg.N_MAX_CELLS_1024
        na = len(per_cell)
        if na > N:
            s = np.random.default_rng(0).choice(na, N, replace=False)
            per_cell = [per_cell[j] for j in s]
            xy_l = [xy_l[j] for j in s]
        xy = (np.stack(xy_l) / (GRID * TILE)) * 2 - 1
        n = min(len(per_cell), N)
        GID = np.full((N, cfg.K_MAX_GENES), PAD, np.int64)
        VAL = np.zeros((N, cfg.K_MAX_GENES), np.float32)
        GM = np.zeros((N, cfg.K_MAX_GENES), bool)
        XY = np.zeros((N, 2), np.float32)
        CM = np.zeros(N, bool)
        for c in range(n):
            gids, val = per_cell[c]
            k = min(len(gids), cfg.K_MAX_GENES)
            srt = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
            GID[c, :k] = gids[srt]
            VAL[c, :k] = (val[srt] - gene_mu[gids[srt]]) / gene_sd[gids[srt]]
            GM[c, :k] = True
        XY[:n] = xy[:n]
        CM[:n] = True
        return {'gene_ids': torch.from_numpy(GID)[None].to(dev), 'gene_vals': torch.from_numpy(VAL)[None].to(dev),
                'gmask': torch.from_numpy(GM)[None].to(dev), 'xy': torch.from_numpy(XY)[None].to(dev),
                'cmask': torch.from_numpy(CM)[None].to(dev)}

    fig, ax = plt.subplots(nshow, 3, figsize=(13, 4.3 * nshow))
    if nshow == 1:
        ax = ax[None]
    with h5py.File(f'{cfg.OUT}/{args.sample}_tiles256_p5.h5', 'r') as h5:
        for r, idx in enumerate(sel):
            FY, FX, present = full_va[idx]
            gt = np.zeros((GRID * TILE, GRID * TILE, 3), np.uint8)
            oracle_uni = np.zeros((NSUB, 1536), np.float32)
            for (gy, gx) in present:
                dy, dx = gy - GRID * FY, gx - GRID * FX
                k = dy * GRID + dx
                row = grid2row_s[(gy, gx)]
                gt[dy * TILE:(dy + 1) * TILE, dx * TILE:(dx + 1) * TILE] = h5['images'][row]
                oracle_uni[k] = UNI_s[row]
            with torch.no_grad():
                pv = model(make_field_batch(FY, FX, present))[0].cpu()
            orc = to_np(gen1024(torch.tensor(oracle_uni)))
            prd = to_np(gen1024(pv))
            own = F.cosine_similarity(pv, torch.tensor(oracle_uni), dim=-1).mean().item()
            dg, do, dp = nuc_dia(gt), nuc_dia(orc), nuc_dia(prd)
            for c, (img, ti, dia) in enumerate([(gt, 'GT', dg), (orc, 'Oracle', do), (prd, 'Pred', dp)]):
                ax[r, c].imshow(img)
                ax[r, c].axis('off')
                ax[r, c].set_title(f'{ti} \u2300{dia:.0f}px' + (f'  cos{own:.2f}' if c == 2 else ''))
            print(f"field({FY},{FX})  GT\u2300{dg:.1f}  oracle\u2300{do:.1f} (ratio {do/dg:.2f})  "
                  f"pred\u2300{dp:.1f}  mean_cos{own:.3f}")
    plt.tight_layout()
    lora_tag = '_lora' if args.pixcell_lora else ''
    out_path = Path(cfg.CKPT) / f'1024_scalecheck_{args.sample}{loo_tag}{lora_tag}.png'
    plt.savefig(out_path, dpi=110)
    print(f'\nsaved {out_path}')
    print('oracle ratio ~1.0 -> scale FIXED. seam alignment between sub-tiles = the 1024 coherence story.')


if __name__ == '__main__':
    main()

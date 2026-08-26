# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Qualitative ground-truth / no-LoRA / LoRA comparison grids.

One row per sample, one column per cell-density percentile.
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image

PERCENTILE_TARGETS = [5, 30, 70, 95]
dev = 'cuda'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--samples', default='C1,C2,P1', help='comma-separated sample names')
    p.add_argument('--out', default='./data/tiles', help='tile/adata directory')
    p.add_argument('--backbone', required=True,
                   help='mapper checkpoint path, with {sample} substituted per sample')
    p.add_argument('--lora', required=True,
                   help='LoRA adapter path, with {sample} substituted per sample')
    p.add_argument('--save-dir', required=True, help='directory for the generated PNGs')
    return p.parse_args()


def pick_tiles(OUT, sample):
    meta = json.load(open(f'{OUT}/{sample}_meta.json'))
    with h5py.File(f'{OUT}/{sample}_tiles256_p5.h5', 'r') as f:
        coords = f['coords'][:]
        nat = int(f.attrs['native'])
    grid2row = {(int(y0 // nat), int(x0 // nat)): i for i, (y0, x0, _) in enumerate(coords)}
    ns = np.array([len(d['cell_idx']) for d in meta])
    picked = []
    for pct in PERCENTILE_TARGETS:
        target = np.percentile(ns, pct)
        idx = int(np.argmin(np.abs(ns - target)))
        d = meta[idx]
        row = grid2row.get((d['gy'], d['gx']))
        if row is not None:
            picked.append((pct, d, row, ns[idx]))
    return picked


def main():
    a = parse_args()
    OUT, SAVE_DIR = a.out, a.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    DATASETS = {s: dict(backbone=a.backbone.format(sample=s), lora=a.lora.format(sample=s))
                for s in a.samples.split(',')}

    from ..core.config import make_cfg_256
    from ..models.mapper import GeneTokenCells256

    from diffusers import DiffusionPipeline, AutoencoderKL
    from peft import PeftModel

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                        torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-256', vae=vae,
                                            custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                            torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    pix.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def decode(uni_vec, use_lora_adapter):
        if use_lora_adapter:
            pix.transformer.set_adapter('default')
            pix.transformer.enable_adapter_layers()
        else:
            pix.transformer.disable_adapter_layers()
        cond = uni_vec.to(pix.device).to(pix.dtype).view(1, 1, -1)
        unc = pix.get_unconditional_embedding(1)
        with torch.amp.autocast('cuda'):
            img = pix(uni_embeds=cond, negative_uni_embeds=unc, num_inference_steps=30,
                      guidance_scale=2.0, output_type='pt').images[0]
        return (img.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')

    manifest = {}
    for sample, paths in DATASETS.items():
        print(f'=== {sample} ===', flush=True)
        cfg = make_cfg_256(seed=0, fm='uni2h')
        cfg.OUT = OUT
        cfg.SAMPLES = [sample]
        ck_d = torch.load(paths['backbone'], map_location='cpu', weights_only=False)
        GENES = ck_d['genes']
        G2I = {g: i for i, g in enumerate(GENES)}
        cfg.GENE_VOCAB = len(GENES)
        cfg.GENES = GENES
        model = GeneTokenCells256(cfg).to(dev).eval()
        model.load_state_dict(ck_d['model'])

        pix.transformer = PeftModel.from_pretrained(
            pix.transformer.get_base_model() if hasattr(pix.transformer, 'get_base_model') else pix.transformer,
            paths['lora']).to(dev)
        pix.transformer = pix.transformer.to(dev).to(torch.float16)

        from .dataset import sample_sparse, load_or_build_256
        import anndata as ad
        A = ad.read_h5ad(f'{OUT}/{sample}_adata.h5ad')
        Xs, cols = sample_sparse(A, cfg, G2I)

        train_ds, _, _, _, _ = load_or_build_256(cfg, G2I, 0, False, lambda i: None)
        gene_mu, gene_sd = train_ds.gene_mu, train_ds.gene_sd
        PAD = cfg.GENE_VOCAB

        with h5py.File(f'{OUT}/{sample}_tiles256_p5.h5', 'r') as f:
            GT_IMGS = f['images'][:]

        picked = pick_tiles(OUT, sample)
        manifest[sample] = []
        for pct, d, row, ncells in picked:
            gt = GT_IMGS[row]
            Lc, Lg = cfg.N_MAX_CELLS, cfg.K_MAX_GENES
            ci = np.asarray(d['cell_idx'])
            xy = (np.asarray(d['local_xy'], np.float32) / 256.0) * 2 - 1
            n = min(len(ci), Lc)
            GID = np.full((1, Lc, Lg), PAD, np.int64)
            VAL = np.zeros((1, Lc, Lg), np.float32)
            GM = np.zeros((1, Lc, Lg), bool)
            XY = np.zeros((1, Lc, 2), np.float32)
            CM = np.zeros((1, Lc), bool)
            for c in range(n):
                r = Xs.getrow(ci[c]); gids, val = cols[r.indices], r.data.astype(np.float32)
                k = min(len(gids), Lg)
                sel = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
                GID[0, c, :k] = gids[sel]
                VAL[0, c, :k] = (val[sel] - gene_mu[gids[sel]]) / gene_sd[gids[sel]]
                GM[0, c, :k] = True
            XY[0, :n] = xy[:n]; CM[0, :n] = True
            batch = {'gene_ids': torch.from_numpy(GID).to(dev), 'gene_vals': torch.from_numpy(VAL).to(dev),
                     'gmask': torch.from_numpy(GM).to(dev), 'xy': torch.from_numpy(XY).to(dev),
                     'cmask': torch.from_numpy(CM).to(dev)}
            with torch.no_grad():
                pv = model(batch)[0].cpu()
            pred_nolora = decode(pv, use_lora_adapter=False)
            pred_lora = decode(pv, use_lora_adapter=True)

            tag = f'{sample}_pct{pct:02d}_n{ncells}'
            Image.fromarray(gt).save(f'{SAVE_DIR}/{tag}_gt.png')
            Image.fromarray(pred_nolora).save(f'{SAVE_DIR}/{tag}_nolora.png')
            Image.fromarray(pred_lora).save(f'{SAVE_DIR}/{tag}_lora.png')
            print(f'  pct={pct} n_cells={ncells} -> saved {tag}_{{gt,nolora,lora}}.png', flush=True)
            manifest[sample].append(dict(pct=pct, n_cells=int(ncells), tag=tag))

    json.dump(manifest, open(f'{SAVE_DIR}/manifest.json', 'w'), indent=2)
    print('done, manifest at', f'{SAVE_DIR}/manifest.json')


if __name__ == '__main__':
    main()

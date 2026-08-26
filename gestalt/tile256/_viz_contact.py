# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Contact sheets of generated tiles.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

OUT = os.environ.get('GESTALT_OUT', './data/tiles')
MAPPER_CKPT = os.environ.get('GESTALT_CKPT', './checkpoints')
LORA_CKPT = os.environ.get('GESTALT_OUT', './data/tiles')
MANIFEST_PATH = '/tmp/rank32_retrain/gt_contact_sheets/manifest.json'
SAVE_DIR = os.environ.get('GESTALT_FIGS', './figures')
RENDER_CACHE = '/tmp/rank32_retrain/gt_contact_sheets/lora_renders'
os.makedirs(RENDER_CACHE, exist_ok=True)

DATASETS_CKPT = {
    'C1': dict(backbone=f'{MAPPER_CKPT}/pooled_genetoken_g1_s0_random_C1_k8_out-xenium_sc_overlap_out_40x.pt',
               lora=f'{LORA_CKPT}/pixcell_lora_r32_random_C1/step_8000'),
    'C2': dict(backbone=f'{MAPPER_CKPT}/pooled_genetoken_g1_s0_random_C2_k8_out-xenium_sc_overlap_out_40x.pt',
               lora=f'{LORA_CKPT}/pixcell_lora_r32_random_C2/step_8000'),
    'P1': dict(backbone=f'{MAPPER_CKPT}/pooled_genetoken_g1_s0_random_P1_k8_out-xenium_sc_overlap_out_40x.pt',
               lora=f'{LORA_CKPT}/pixcell_lora_r32_random_P1/step_8000'),
}
THUMB = 160
COLS = 10
ROWS = 5
LABEL_H = 22
dev = 'cuda'


def _font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_dataset(sample, picked):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ..core.config import make_cfg_256
    from ..models.mapper import GeneTokenCells256
    from .dataset import sample_sparse, load_or_build_256
    from diffusers import DiffusionPipeline, AutoencoderKL
    from peft import PeftModel
    import anndata as ad

    ckpt = DATASETS_CKPT[sample]
    cfg = make_cfg_256(seed=0, fm='uni2h')
    cfg.OUT = OUT
    cfg.SAMPLES = [sample]
    ck_d = torch.load(ckpt['backbone'], map_location='cpu', weights_only=False)
    GENES = ck_d['genes']
    G2I = {g: i for i, g in enumerate(GENES)}
    cfg.GENE_VOCAB = len(GENES)
    model = GeneTokenCells256(cfg).to(dev).eval()
    model.load_state_dict(ck_d['model'])

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                        torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-256', vae=vae,
                                            custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                            torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    pix.set_progress_bar_config(disable=True)
    pix.transformer = PeftModel.from_pretrained(
        pix.transformer.get_base_model() if hasattr(pix.transformer, 'get_base_model') else pix.transformer,
        ckpt['lora']).to(dev).to(torch.float16)
    pix.transformer.set_adapter('default')
    pix.transformer.enable_adapter_layers()

    A = ad.read_h5ad(f'{OUT}/{sample}_adata.h5ad')
    Xs, cols = sample_sparse(A, cfg, G2I)
    train_ds, _, _, _, _ = load_or_build_256(cfg, G2I, 0, False, lambda i: None)
    gene_mu, gene_sd = train_ds.gene_mu, train_ds.gene_sd
    PAD = cfg.GENE_VOCAB
    Lc, Lg = cfg.N_MAX_CELLS, cfg.K_MAX_GENES
    meta = json.load(open(f'{OUT}/{sample}_meta.json'))

    renders = {}
    for rec in picked:
        d = meta[rec['meta_idx']]
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
        cond = pv.to(dev).to(pix.dtype).view(1, 1, -1)
        unc = pix.get_unconditional_embedding(1)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            img = pix(uni_embeds=cond, negative_uni_embeds=unc, num_inference_steps=30,
                      guidance_scale=2.0, output_type='pt').images[0]
        arr = (img.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')
        renders[rec['tile_id']] = arr
        Image.fromarray(arr).save(f'{RENDER_CACHE}/{sample}_{rec["tile_id"]}_lora.png')
        print(f'  {rec["tile_id"]} rendered', flush=True)
    return renders


def build_sheet(picked, renders, out_path):
    canvas = Image.new('RGB', (COLS * THUMB, ROWS * (THUMB + LABEL_H)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    for i, rec in enumerate(picked):
        r, c = divmod(i, COLS)
        im = Image.fromarray(renders[rec['tile_id']]).resize((THUMB, THUMB))
        x0, y0 = c * THUMB, r * (THUMB + LABEL_H)
        canvas.paste(im, (x0, y0))
        draw.rectangle([x0, y0 + THUMB, x0 + THUMB, y0 + THUMB + LABEL_H], fill='black')
        draw.text((x0 + THUMB / 2, y0 + THUMB + LABEL_H / 2), rec['tile_id'],
                   font=font, fill='white', anchor='mm')
    canvas.save(out_path)
    print('saved', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='C1,C2,P1')
    args = ap.parse_args()
    manifest = json.load(open(MANIFEST_PATH))
    for sample in args.samples.split(','):
        print(f'=== {sample} ===', flush=True)
        picked = manifest[sample]
        renders = render_dataset(sample, picked)
        build_sheet(picked, renders, f'{SAVE_DIR}/{sample}_LoRA_contact_sheet.png')


if __name__ == '__main__':
    main()

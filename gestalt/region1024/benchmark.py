#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark for full 1024px region synthesis.
"""
import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ..core.config import make_cfg_1024
from ..models.mapper import GeneTokenCells1024
from .dataset import load_or_build_1024
from ..tile256.dataset import load_or_build_256, load_vocab
from ..core.ddp_utils import barrier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True, help='full path to a genetoken1024 checkpoint (.pt)')
    ap.add_argument('--sample', required=True, help='HEST sample this checkpoint was trained on, e.g. C2')
    ap.add_argument('--out', required=True, help='directory holding the non-overlap lattice')
    ap.add_argument('--pixcell-lora', required=True, help='PixCell-1024 LoRA adapter dir (shared decoder)')
    ap.add_argument('--train-overlap-stride', type=int, required=True,
                     help='must match the checkpoint; selects which field cache is used')
    ap.add_argument('--test-overlap-stride', type=int, default=1)
    ap.add_argument('--n', type=int, default=None, help='cap on fully-covered val fields (default: all)')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=1.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--results-dir', default=None)
    a = ap.parse_args()
    dev = 'cuda'

    cfg = make_cfg_1024(seed=a.seed)
    cfg.SAMPLES = [a.sample]
    cfg.OUT = a.out

    GENES, G2I = load_vocab(cfg)
    cfg.K_MAX_GENES = 87
    train256, _, _, _, _ = load_or_build_256(cfg, G2I, 0, False, barrier)
    gene_mu, gene_sd = train256.gene_mu, train256.gene_sd
    del train256

    _, _, va1024 = load_or_build_1024(cfg, G2I, gene_mu, gene_sd, 0, False, barrier,
                                       train_overlap_stride=a.train_overlap_stride,
                                       test_overlap_stride=a.test_overlap_stride)
    n_full = sum(1 for f in va1024 if f[3].sum() == 16)
    n_real_tiles = sum(int(f[3].sum()) for f in va1024)
    print(f'val: {len(va1024)} total fields ({n_full} fully covered 16/16), '
          f'{n_real_tiles} real subtiles total')
    fields = va1024
    if a.n and len(fields) > a.n:
        rng = np.random.RandomState(a.seed)
        idx = sorted(rng.choice(len(fields), a.n, replace=False).tolist())
        fields = [fields[i] for i in idx]
    print(f'evaluating n={len(fields)} fields '
          f'({sum(int(f[3].sum()) for f in fields)} real subtiles)')

    d = torch.load(a.ckpt_path, map_location='cpu', weights_only=False)
    for k, v in d['cfg'].items():
        if v is not None:
            setattr(cfg, k, v)
    model = GeneTokenCells1024(cfg).to(dev).eval()
    model.load_state_dict(d['model'])
    print(f'loaded {a.ckpt_path}: training-time own_cos={d["own_cos"]:.4f} ep={d["ep"]}')
    PAD = cfg.GENE_VOCAB

    with h5py.File(f'{a.out}/{a.sample}_tiles256_p5_uni.h5', 'r') as f:
        real_emb = torch.from_numpy(f['uni'][:]).float()
    real_emb_n = F.normalize(real_emb, dim=-1)
    with h5py.File(f'{a.out}/{a.sample}_tiles256_p5.h5', 'r') as f:
        real_images = f['images'][:]

    from diffusers import AutoencoderKL, DiffusionPipeline
    from peft import PeftModel
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                         torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-1024', vae=vae,
                                             custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                             torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    pix.transformer = PeftModel.from_pretrained(pix.transformer, a.pixcell_lora).to(dev).to(torch.float16)
    pix.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def decode_1024(uni16):
        cond = uni16.to(pix.device).to(pix.dtype).unsqueeze(0)
        unc = pix.get_unconditional_embedding(1)
        with torch.amp.autocast('cuda'):
            return (pix(uni_embeds=cond, negative_uni_embeds=unc, num_inference_steps=a.steps,
                         guidance_scale=a.guidance, height=1024, width=1024, output_type='pt').images[0]
                    .float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')

    def build_batch(per_cell, xy):
        N = cfg.N_MAX_CELLS_1024
        n = min(len(per_cell), N)
        GID = np.full((1, N, cfg.K_MAX_GENES), PAD, np.int64)
        VAL = np.zeros((1, N, cfg.K_MAX_GENES), np.float32)
        GM = np.zeros((1, N, cfg.K_MAX_GENES), bool)
        XY = np.zeros((1, N, 2), np.float32)
        CM = np.zeros((1, N), bool)
        for c in range(n):
            gids, val = per_cell[c]
            k = min(len(gids), cfg.K_MAX_GENES)
            srt = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
            GID[0, c, :k] = gids[srt]
            VAL[0, c, :k] = (val[srt] - gene_mu[gids[srt]]) / gene_sd[gids[srt]]
            GM[0, c, :k] = True
        XY[0, :n] = xy[:n]
        CM[0, :n] = True
        return {'gene_ids': torch.from_numpy(GID).to(dev), 'gene_vals': torch.from_numpy(VAL).to(dev),
                'gmask': torch.from_numpy(GM).to(dev), 'xy': torch.from_numpy(XY).to(dev),
                'cmask': torch.from_numpy(CM).to(dev)}

    def split_grid(g):
        """1024x1024 -> 16 256x256 crops, row-major (matches _construct_field's own
        pos = row*4+col packing in src/data_1024.py)."""
        return [g[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                for r in range(4) for c in range(4)]

    def grid(tiles):
        rows4 = [np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(4)]
        return np.concatenate(rows4, axis=0)

    # d_gt/d_pr: per-subtile crops for Crop FID (every real subtile, any field). d_gt_full/
    # d_pr_full: full mosaics for CLIP FID, from fully-covered fields only.
    d_gt, d_pr = tempfile.mkdtemp(), tempfile.mkdtemp()
    d_gt_full, d_pr_full = tempfile.mkdtemp(), tempfile.mkdtemp()
    ssims, psnrs, own_coss = [], [], []
    n_tiles, n_full = 0, 0
    for fi, (per_cell, xy, tgt, tmask) in enumerate(fields):
        with torch.no_grad():
            pred = model(build_batch(per_cell, xy))[0].cpu().numpy()  # (16,1536)
        real_pos = np.where(tmask)[0]
        oc = F.cosine_similarity(torch.from_numpy(pred[real_pos]), torch.from_numpy(tgt[real_pos]),
                                  dim=-1).mean().item()
        own_coss.append(oc)

        pred_grid = decode_1024(torch.from_numpy(pred))
        pred_tiles = split_grid(pred_grid)
        gt_tiles_this_field = {}
        for k in real_pos:
            tgt_k = F.normalize(torch.from_numpy(tgt[k]).float().unsqueeze(0), dim=-1)
            sims = (tgt_k @ real_emb_n.T).squeeze(0)
            gt_tile = real_images[sims.argmax().item()]
            gt_tiles_this_field[k] = gt_tile
            ssims.append(ssim_fn(gt_tile, pred_tiles[k], channel_axis=2))
            psnrs.append(psnr_fn(gt_tile, pred_tiles[k]))
            Image.fromarray(gt_tile).save(f'{d_gt}/{n_tiles:06d}.png')
            Image.fromarray(pred_tiles[k]).save(f'{d_pr}/{n_tiles:06d}.png')
            n_tiles += 1
        if len(real_pos) == 16:
            gt_grid = grid([gt_tiles_this_field[k] for k in range(16)])
            Image.fromarray(gt_grid).save(f'{d_gt_full}/{n_full:06d}.png')
            Image.fromarray(pred_grid).save(f'{d_pr_full}/{n_full:06d}.png')
            n_full += 1
        print(f'  field {fi}/{len(fields)} ({len(real_pos)}/16 real): own_cos={oc:.4f}')
    print(f'  {n_full}/{len(fields)} fields fully covered (usable for CLIP FID on the full mosaic)')

    from cleanfid import fid
    import numpy as _np
    from scipy import linalg as _linalg

    def regularized_frechet(mu1, sigma1, mu2, sigma2, eps):
        diff = mu1 - mu2
        offset = _np.eye(sigma1.shape[0]) * eps
        covmean, _ = _linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        if _np.iscomplexobj(covmean):
            covmean = covmean.real  # regularised above, so the imaginary part is numerical noise
        return float(diff.dot(diff) + _np.trace(sigma1) + _np.trace(sigma2) - 2 * _np.trace(covmean))

    def robust_fid(dir1, dir2, n_samples, model_name='inception_v3'):
        """FID between two image folders, with both covariances regularised before sqrtm.

        An eps*I offset is added to sigma1 and sigma2 and escalated until sqrtm returns a usable
        matrix, which keeps the Frechet term well defined when the sample count is small
        relative to the feature dimension (2048 for Inception, 512 for CLIP ViT-B/32).
        """
        if model_name == 'clip_vit_b_32':
            from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
            feat_model = CLIP_fx('ViT-B/32', device='cuda')
            resize_fn = img_preprocess_clip
        else:
            feat_model = fid.build_feature_extractor('clean', 'cuda')
            resize_fn = None
        feats1 = fid.get_folder_features(dir1, feat_model, num_workers=4, batch_size=64,
                                          device='cuda', mode='clean', verbose=False,
                                          custom_fn_resize=resize_fn)
        feats2 = fid.get_folder_features(dir2, feat_model, num_workers=4, batch_size=64,
                                          device='cuda', mode='clean', verbose=False,
                                          custom_fn_resize=resize_fn)
        mu1, sigma1 = _np.mean(feats1, axis=0), _np.cov(feats1, rowvar=False)
        mu2, sigma2 = _np.mean(feats2, axis=0), _np.cov(feats2, rowvar=False)
        for eps in (1e-6, 1e-3, 1e-1, 1.0, 10.0):
            try:
                return regularized_frechet(mu1, sigma1, mu2, sigma2, eps), eps
            except Exception as e:
                print(f'    {model_name} FID failed at eps={eps}: {e}')
        return None, None

    # Crop FID: 256px crops against real 256px patches with vanilla Inception-v3, the protocol
# PixCell reports for 1024px generations.
    crop_fid, crop_fid_eps = robust_fid(d_gt, d_pr, n_tiles, model_name='inception_v3')
    if crop_fid is not None:
        print(f'  Crop FID (regularized, eps={crop_fid_eps}) = {crop_fid:.2f} -- n={n_tiles}, '
              f'directional not Table-1-quality (n=2800+ there)')
    kid_score = fid.compute_kid(d_gt, d_pr, verbose=False)

    # CLIP FID: full 1024px mosaic, CLIP ViT-B/32 features (PixCell's stated reason: better
    # suited to evaluating images >224px than Inception, which was built for ~299px). Only
    # fully-covered fields have a full mosaic to compare.
    clip_fid, clip_fid_eps = (None, None)
    if n_full >= 2:
        clip_fid, clip_fid_eps = robust_fid(d_gt_full, d_pr_full, n_full, model_name='clip_vit_b_32')
        if clip_fid is not None:
            print(f'  CLIP FID (regularized, eps={clip_fid_eps}) = {clip_fid:.2f} -- n={n_full} '
                  f'full mosaics')
    else:
        print(f'  CLIP FID skipped: only {n_full} fully-covered field(s), need >=2')

    print(f'\n=== {a.sample} train_ov={a.train_overlap_stride} test_ov={a.test_overlap_stride} '
          f'n_fields={len(fields)} n_tiles={n_tiles} n_full={n_full} ===')
    fid_str = f'{crop_fid:.2f}' if crop_fid is not None else 'FAILED'
    clip_str = f'{clip_fid:.2f}' if clip_fid is not None else 'FAILED/SKIPPED'
    print(f'CropFID={fid_str}  CLIPFID={clip_str}  KIDx1000={kid_score * 1000:.3f}  '
          f'SSIM={np.mean(ssims):.3f}+/-{np.std(ssims):.3f}  '
          f'PSNR={np.mean(psnrs):.2f}+/-{np.std(psnrs):.2f}  '
          f'own_cos={np.mean(own_coss):.4f}+/-{np.std(own_coss):.4f}')

    result = {
        'timestamp': datetime.datetime.now().isoformat(), 'sample': a.sample,
        'ckpt_path': a.ckpt_path, 'ckpt_own_cos_train': d.get('own_cos'), 'ckpt_ep': d.get('ep'),
        'pixcell_lora': a.pixcell_lora,
        'train_overlap_stride': a.train_overlap_stride, 'test_overlap_stride': a.test_overlap_stride,
        'n_fields': len(fields), 'n_tiles': n_tiles, 'n_full': n_full, 'steps': a.steps,
        'guidance': a.guidance,
        'crop_fid': float(crop_fid) if crop_fid is not None else None,
        'crop_fid_eps': crop_fid_eps,
        'clip_fid': float(clip_fid) if clip_fid is not None else None,
        'clip_fid_eps': clip_fid_eps,
        'kid': float(kid_score),
        'ssim': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'psnr': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'own_cos_mean': float(np.mean(own_coss)), 'own_cos_std': float(np.std(own_coss)),
    }
    results_dir = Path(a.results_dir) if a.results_dir else Path(a.out) / 'benchmarks'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'benchmark_1024_log.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(f'logged -> {results_dir / "benchmark_1024_log.jsonl"}')


if __name__ == '__main__':
    main()

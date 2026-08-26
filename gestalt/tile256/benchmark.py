#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark: FID / SSIM / FeatDist / uniCOS for any conditioning signal.

Use --full-sample when the held-out slide sits outside the random split, which is the case for
every leave-one-out holdout.
"""
import argparse, sys, os, json, tempfile, shutil, datetime, hashlib
from pathlib import Path
import numpy as np, torch, h5py, anndata as ad
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from torchvision.models import inception_v3
from ..core.vocab import build_vocab

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ..core.config import make_cfg_256, samples_tag, fm_tag
from .pixcell import swap_caption_projection
from .dataset import load_or_build_256, fm_h5
from ..models.mapper import GeneTokenCells256


def _nb(is_ddp):
    pass


def _jsonable(v):
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _patch_frechet_distance(fid_module):
    """Regularise both covariances with eps*I before sqrtm, which keeps the Frechet term well
    defined when the sample count is small relative to the 2048-dim feature space."""
    from scipy import linalg as _linalg

    def _robust_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
        diff = mu1 - mu2
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = _linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    fid_module.frechet_distance = _robust_frechet_distance


def write_results(result, results_dir, out_json, sample, guidance):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_json is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # pid disambiguates concurrent runs on one sample: the timestamp is second-resolution,
        # so parallel arms finishing in the same second would overwrite each other's results
        out_json = results_dir / f'bench_{sample}_random_g{guidance:g}_{ts}_p{os.getpid()}.json'
    else:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    log_path = results_dir / 'benchmark_log.jsonl'
    with open(log_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
    return out_json, log_path


class _Inception2048(torch.nn.Module):
    """inception_v3 (IMAGENET1K_V1, transform_input=False) with fc replaced by Identity, giving
    2048-d features. Input in [0,1], any HxW, 1/3/more channels."""
    def __init__(self, device):
        super().__init__()
        m = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        m.fc = torch.nn.Identity()
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        self.m = m.to(device)

    @torch.no_grad()
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = x * 2 - 1
        return self.m(x)


def _feat_in(img_uint8, dev):
    """uint8 HxWxC in 0..255 -> [1,C,H,W] float in [0,1] on dev."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(dev)


def _tile_noise_seed(sampling_seed, sample, row):
    """Stable per-tile seed, independent of Python's process-randomized ``hash()``."""
    sample_id = int.from_bytes(
        hashlib.blake2b(sample.encode('utf-8'), digest_size=8).digest(), 'little')
    return int((int(sampling_seed) * 1_000_003 + sample_id + int(row)) % (2**63 - 1))


_random_holdout_cache = {}


def _random_holdout_rows_by_sample(cfg, G2I, eval_samples):
    """Return ordered image rows held out of training for every requested sample.

    The rows come from the cached train/val split that training itself used. cfg.SAMPLES must
    be the full pool the checkpoint was trained on for the split to resolve.
    """
    key = (tuple(cfg.SAMPLES), getattr(cfg, 'TEST_SAMPLE', None), cfg.OUT, cfg.SPLIT_SEED)
    # The cache key excludes eval_samples: the train/val split depends only on the remaining
    # fields and is shared across calls, while the returned rows are per sample.
    if key not in _random_holdout_cache:
        from .dataset import load_or_build_256
        _, val_ds, va_s, _, _ = load_or_build_256(cfg, G2I, 0, False, lambda is_ddp: None)
        by_idx = {}
        for rec, sid in zip(va_s, val_ds.slide_ids):
            by_idx.setdefault(int(sid), []).append(rec)
        _random_holdout_cache[key] = {'by_idx': by_idx, 'result': {}}
    by_idx = _random_holdout_cache[key]['by_idx']
    result = _random_holdout_cache[key]['result']
    for i, s in enumerate(cfg.SAMPLES):
        if s not in eval_samples or s in result:
            continue
        recs = by_idx.get(i, [])
        fm_path, fm_key = fm_h5(cfg, s)
        with h5py.File(fm_path, 'r') as g:
            UNI_s = g[fm_key][:]
        tgt2row = {UNI_s[r].astype(np.float32).tobytes(): r for r in range(UNI_s.shape[0])}
        rows = []
        unmatched = 0
        for per_cell, xy, tgt in recs:
            row = tgt2row.get(np.asarray(tgt, np.float32).tobytes())
            if row is None:
                unmatched += 1
            else:
                rows.append(row)
        if unmatched:
            raise RuntimeError(f'{unmatched}/{len(recs)} held-out records for {s} did not '
                               f'exact-match a row in {fm_key}; refusing to benchmark.')
        if len(set(rows)) != len(rows):
            raise RuntimeError(f'{s}: held-out records do not map one-to-one to image rows; '
                               'refusing to benchmark an ambiguous split.')
        result[s] = rows
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0,
                    help='mapper training/checkpoint seed; must match the trained mapper')
    ap.add_argument('--split-seed', type=int, default=0,
                    help='train/validation split seed; must match mapper and LoRA training')
    ap.add_argument('--sampling-seed', type=int, default=0,
                    help='diffusion-noise seed; changes renders without changing the mapper or '
                         'held-out tiles')
    ap.add_argument('--seed-check', action='store_true',
                    help='on the first tile, assert that identical sampling seeds reproduce '
                         'exactly and a different seed changes the decoded image')
    ap.add_argument('--samples', default=None, help='training slide set (checkpoint tag)')
    ap.add_argument('--sample', required=True,
                    help='slide(s) to evaluate on; comma-separated pools them into one FID')
    ap.add_argument('--test-sample', default=None)
    ap.add_argument('--full-sample', action='store_true',
                    help="evaluate every tile of --sample, for a slide held out in full")
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=1.5)
    ap.add_argument('--n', type=int, default=None, help='cap eval tiles (default: all holdout)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pixcell-lora', default=None, help='LoRA adapter dir to load onto PixCell')
    ap.add_argument('--fm', default='uni2h',
                    help='conditioning signal: a foundation model in FM_REGISTRY, or the tag of a '
                         'must match how --pixcell-lora was trained')
    ap.add_argument('--no-pred', action='store_true',
                    help='report only the row conditioned on the real embedding, skipping the '
                         'mapper')
    ap.add_argument('--no-oracle', action='store_true',
                    help='skip decoding the real embedding when no oracle row is reported; '
                         'roughly halves runtime for prediction-only benchmarks')
    ap.add_argument('--out', default=None, help='override cfg.OUT')
    ap.add_argument('--results-dir', default=None,
                    help='where to write JSON + benchmark_log.jsonl (default: <cfg.OUT>/benchmarks)')
    ap.add_argument('--out-json', default=None, help='explicit per-run JSON path')
    ap.add_argument('--ckpt', default=None, help='override cfg.CKPT (checkpoint dir)')
    ap.add_argument('--mapper-ckpt', default=None,
                    help='mapper checkpoint path, bypassing the tag-based name construction')
    ap.add_argument('--pairbias-k', type=int, default=None,
                    help='k for the neighbour graph; must match the checkpoint, which a mismatch '
                         'will not crash on')
    ap.add_argument('--keep-renders', default=None,
                    help='keep decoded PNGs under DIR/{gt,oracle,pred}; otherwise use temporary '
                         'directories and delete them after scoring')
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    cfg = make_cfg_256(seed=a.seed, fm=a.fm, split_seed=a.split_seed)
    cfg.TEST_SAMPLE = a.test_sample
    if a.samples:
        cfg.SAMPLES = a.samples.split(',')
    if a.out:
        cfg.OUT = a.out
    if a.ckpt:
        cfg.CKPT = a.ckpt
    if a.pairbias_k is not None:
        cfg.PAIRBIAS_K = a.pairbias_k
    loo_tag = f'_loo_{a.test_sample}' if a.test_sample else ''
    split_tag = '' if a.split_seed == 0 else f'_split{a.split_seed}'
    # these tags must match the ones the trainer writes, so the checkpoint resolves.
    k_tag = f'_k{a.pairbias_k}' if a.pairbias_k is not None else ''

    model = None
    if a.no_pred:
        print('[--no-pred] skipping genetoken checkpoint + gene-expression loading entirely; '
              'only the oracle row (decoder-only quality) will be computed')
    else:
        ck = (Path(a.mapper_ckpt) if a.mapper_ckpt else
              Path(cfg.CKPT) /
              f'mapper_s{a.seed}{split_tag}{loo_tag}{samples_tag(cfg)}{fm_tag(cfg)}{k_tag}.pt')
        if not ck.exists():
            raise FileNotFoundError(f'mapper checkpoint not found: {ck}')
        print(f'[mapper ckpt] {ck.name}'
              + ('  [explicit --mapper-ckpt]' if a.mapper_ckpt else '  [resolved from tags]'))
        # weights_only=False is required because some checkpoint configs embed numpy arrays.
        ck_d = torch.load(ck, map_location='cpu', weights_only=False)
        ck_cfg = ck_d.get('cfg', {})
        ck_args = ck_d.get('args', {})
        ck_train_seed = ck_d.get('seed', ck_cfg.get('SEED'))
        ck_split_seed = ck_args.get('split_seed', ck_cfg.get('SPLIT_SEED'))
        if ck_train_seed is not None and int(ck_train_seed) != a.seed:
            raise ValueError(f'checkpoint training seed {ck_train_seed} does not match '
                             f'--seed {a.seed}')
        if ck_split_seed is not None and int(ck_split_seed) != a.split_seed:
            raise ValueError(f'checkpoint split seed {ck_split_seed} does not match '
                             f'--split-seed {a.split_seed}')

        GENES = ck_d.get('genes')
        if GENES is None:
            GENES, _ = build_vocab(cfg)
            print(f'ckpt has no stamped vocab; rebuilt {len(GENES)} genes from train slides')
        G2I = {g: i for i, g in enumerate(GENES)}
        cfg.GENE_VOCAB = len(GENES)          # sizes enc.gene_emb; MUST precede build_registry
        cfg.GENES = GENES                    # exposes gene identity; needed by external-prior models
        PAD = cfg.GENE_VOCAB

        train_ds, val_ds, _, umu, usd = load_or_build_256(cfg, G2I, 0, False, _nb)
        gene_mu, gene_sd = train_ds.gene_mu, train_ds.gene_sd

        model = GeneTokenCells256(cfg).to(dev).eval()
        model.load_state_dict(ck_d['model'])
        print(f'loaded {ck} | vocab {len(GENES)}')

    eval_samples = a.sample.split(',')
    print(f'pooling {len(eval_samples)} eval slide(s): {eval_samples}')

    from diffusers import DiffusionPipeline, AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                        torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-256', vae=vae,
                                            custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                            torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    if a.fm != 'uni2h':
        swap_caption_projection(pix.transformer, a.fm)
    if a.pixcell_lora:
        from peft import PeftModel
        pix.transformer = PeftModel.from_pretrained(pix.transformer, a.pixcell_lora).to(dev)
        print(f'loaded LoRA {a.pixcell_lora}')
    pix.transformer = pix.transformer.to(dev).to(torch.float16)  # swap_caption_projection's new
    # nn.Linear (and, pre-LoRA, no adapter at all) default to fp32; keep dtypes consistent.
    pix.set_progress_bar_config(disable=True)

    incep = _Inception2048(dev)

    @torch.no_grad()
    def decode(uni, noise_seed):
        # Seed the sampler per tile so a given adapter and tile set reproduce exactly.
        cond = uni.to(pix.device).to(pix.dtype).view(1, 1, -1)
        unc = pix.get_unconditional_embedding(1)
        g = torch.Generator(device=pix.device).manual_seed(int(noise_seed))
        with torch.amp.autocast('cuda'):
            return (pix(uni_embeds=cond, negative_uni_embeds=unc, num_inference_steps=a.steps,
                        guidance_scale=a.guidance, generator=g, output_type='pt').images[0]
                    .float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')

    def make_batch(d, Xs, cols):
        xy = (np.asarray(d['local_xy'], np.float32) / 256.0) * 2 - 1
        ci = np.asarray(d['cell_idx'])
        n = min(len(ci), cfg.N_MAX_CELLS)
        GID = np.full((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), PAD, np.int64)
        VAL = np.zeros((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), np.float32)
        GM = np.zeros((cfg.N_MAX_CELLS, cfg.K_MAX_GENES), bool)
        XY = np.zeros((cfg.N_MAX_CELLS, 2), np.float32)
        CM = np.zeros(cfg.N_MAX_CELLS, bool)
        for c in range(n):
            r = Xs.getrow(ci[c]); gids = cols[r.indices]; val = r.data.astype(np.float32)
            k = min(len(gids), cfg.K_MAX_GENES)
            srt = np.argsort(-val)[:k] if len(gids) > k else np.arange(k)
            GID[c, :k] = gids[srt]
            VAL[c, :k] = (val[srt] - gene_mu[gids[srt]]) / gene_sd[gids[srt]]
            GM[c, :k] = True
        XY[:n] = xy[:n]; CM[:n] = True
        return {'gene_ids': torch.from_numpy(GID)[None].to(dev), 'gene_vals': torch.from_numpy(VAL)[None].to(dev),
                'gmask': torch.from_numpy(GM)[None].to(dev), 'xy': torch.from_numpy(XY)[None].to(dev),
                'cmask': torch.from_numpy(CM)[None].to(dev)}

    if a.no_pred:
        # The tile-level train/val split depends on tile count, not gene content, so a
        # vocabulary built from the current pool is sufficient here.
        GENES, G2I = build_vocab(cfg)
        cfg.GENE_VOCAB = len(GENES)  # _build_256_cache_random needs this sized correctly;
        # GENE_VOCAB must be set before allocating the per-gene accumulator.

    if a.keep_renders:
        d_gt = os.path.join(a.keep_renders, 'gt')
        d_or = os.path.join(a.keep_renders, 'oracle') if not a.no_oracle else None
        d_pr = os.path.join(a.keep_renders, 'pred') if not a.no_pred else None
        for directory in [d_gt] + ([d_or] if d_or else []) + ([d_pr] if d_pr else []):
            os.makedirs(directory, exist_ok=True)
    else:
        d_gt = tempfile.mkdtemp()
        d_or = tempfile.mkdtemp() if not a.no_oracle else None
        d_pr = tempfile.mkdtemp() if not a.no_pred else None
    ssim_o, psnr_o, featd_o = [], [], []
    ssim_p, psnr_p, coss, featd_p = [], [], [], []   # stay empty (unused) when --no-pred
    dropped_per_sample = {}
    n_eval_per_sample = {}
    k = 0  # global tile counter across the whole pool -- unique filenames, one shared FID at the end
    seed_checked = False
    for sample in eval_samples:
        if not a.no_pred:
            A = ad.read_h5ad(f'{cfg.OUT}/{sample}_adata.h5ad')
            keep = [i for i, g in enumerate(A.var_names) if str(g) in G2I]
            dropped = A.n_vars - len(keep)
            dropped_per_sample[sample] = dropped
            if dropped:
                print(f'{dropped}/{A.n_vars} genes on {sample} unseen by checkpoint -> dropped '
                      f'(cross-slide eval on the shared panel)')
            A = A[:, keep]
            layer = A.layers[cfg.GENE_LAYER] if cfg.GENE_LAYER in A.layers else A.X
            Xs = layer.tocsr()
            cols = np.array([G2I[str(g)] for g in A.var_names])

        meta = json.load(open(f'{cfg.OUT}/{sample}_meta.json'))
        fm_path, fm_key = fm_h5(cfg, sample)
        with h5py.File(fm_path, 'r') as g:
            UNI_s = g[fm_key][:]
        fh = h5py.File(f'{cfg.OUT}/{sample}_tiles256_p5.h5', 'r')
        coords = fh['coords'][:]
        nat = int(fh.attrs['native'])
        grid2row = {(int(y // nat), int(x // nat)): i for i, (y, x, _) in enumerate(coords)}

        full_eval = a.full_sample or a.test_sample == sample
        if full_eval:
            # A leave-one-out fold (or --full-sample) excluded the whole sample from training,
            # so every tile is held out and a further slice would only reduce eval power.
            va = meta
        else:
            held_rows = _random_holdout_rows_by_sample(cfg, G2I, set(eval_samples))[sample]
            row_to_meta = {grid2row.get((d['gy'], d['gx'])): d for d in meta
                           if grid2row.get((d['gy'], d['gx'])) is not None}
            missing_rows = [row for row in held_rows if row not in row_to_meta]
            if missing_rows:
                raise RuntimeError(f'{sample}: {len(missing_rows)} held-out image rows have no '
                                   'metadata record; refusing to benchmark.')
            va = [row_to_meta[row] for row in held_rows]
        if a.n:
            # `meta` is in tile raster order, so a plain va[:n] would be a spatial band of the
            # slide; shuffle first so the cap draws an unbiased sample.
            _rs = np.random.RandomState(a.seed)
            va = [va[i] for i in sorted(_rs.permutation(len(va))[:a.n])]
        n_eval_per_sample[sample] = len(va)
        print(f'eval tiles (holdout, split=random): {len(va)} on {sample}'
              + (' [full LOO sample]' if full_eval else ''))

        for d in va:
            row = grid2row.get((d['gy'], d['gx']))
            if row is None:
                continue
            gt = fh['images'][row]
            rv = torch.tensor(UNI_s[row])
            noise_seed = _tile_noise_seed(a.sampling_seed, sample, row)
            orc = decode(rv, noise_seed) if not a.no_oracle else None
            if a.seed_check and not seed_checked:
                reference = orc if orc is not None else decode(rv, noise_seed)
                same = decode(rv, noise_seed)
                changed = decode(rv, _tile_noise_seed(a.sampling_seed + 1, sample, row))
                if not np.array_equal(reference, same):
                    raise RuntimeError('seed check failed: identical sampling seeds produced '
                                       'different images')
                mean_abs_change = float(np.abs(reference.astype(np.float32) -
                                               changed.astype(np.float32)).mean())
                if mean_abs_change == 0.0:
                    raise RuntimeError('seed check failed: changing the sampling seed did not '
                                       'change the decoded image')
                print(f'[seed check] PASS: same seed is pixel-identical; seed+1 changes the '
                      f'image (mean absolute uint8 difference {mean_abs_change:.3f})')
                seed_checked = True
            Image.fromarray(gt).save(f'{d_gt}/{k:05d}.png')
            with torch.no_grad():
                f_gt = incep(_feat_in(gt, dev)).cpu().numpy()[0]
            if not a.no_oracle:
                Image.fromarray(orc).save(f'{d_or}/{k:05d}.png')
                ssim_o.append(ssim_fn(gt, orc, channel_axis=2)); psnr_o.append(psnr_fn(gt, orc))
                with torch.no_grad():
                    f_or = incep(_feat_in(orc, dev)).cpu().numpy()[0]
                featd_o.append(float(np.linalg.norm(f_gt - f_or)))

            if not a.no_pred:
                with torch.no_grad():
                    pv = model(make_batch(d, Xs, cols))[0].cpu()
                # Share the tile's noise with the oracle so their gap isolates conditioning.
                pr = decode(pv, noise_seed)
                Image.fromarray(pr).save(f'{d_pr}/{k:05d}.png')
                ssim_p.append(ssim_fn(gt, pr, channel_axis=2)); psnr_p.append(psnr_fn(gt, pr))
                coss.append(F.cosine_similarity(pv, rv, dim=-1).item())
                with torch.no_grad():
                    f_pr = incep(_feat_in(pr, dev)).cpu().numpy()[0]
                featd_p.append(float(np.linalg.norm(f_gt - f_pr)))

            k += 1
            if k % 25 == 0:
                print(f'  {k}/pool')
        fh.close()

    from cleanfid import fid
    _patch_frechet_distance(fid)
    fid_or = fid.compute_fid(d_gt, d_or, verbose=False) if not a.no_oracle else None
    fid_pr = fid.compute_fid(d_gt, d_pr, verbose=False) if not a.no_pred else None

    if not a.no_oracle:
        fo_m, fo_s = np.mean(featd_o), np.std(featd_o)
        ssim_o_m, ssim_o_s = np.mean(ssim_o), np.std(ssim_o)
        psnr_o_m, psnr_o_s = np.mean(psnr_o), np.std(psnr_o)
    if not a.no_pred:
        fp_m, fp_s = np.mean(featd_p), np.std(featd_p)
        ssim_p_m, ssim_p_s = np.mean(ssim_p), np.std(ssim_p)
        psnr_p_m, psnr_p_s = np.mean(psnr_p), np.std(psnr_p)

    print(f'\n=== {a.sample}  n={k}  guidance={a.guidance}  split=random  '
          f'fm={a.fm}  (H&E region target) ===')
    print(f'{"row":8s} {"FID dn":>8s} {"SSIM up":>8s} {"PSNR up":>8s} {"UNIcos up":>10s} '
          f'{"FeatDist dn":>16s}')
    if not a.no_pred:
        print(f'{"pred":8s} {fid_pr:8.2f} {ssim_p_m:8.3f} {psnr_p_m:8.2f} '
              f'{np.mean(coss):10.3f} {f"{fp_m:.1f}+/-{fp_s:.1f}":>16s}')
    if not a.no_oracle:
        print(f'{"oracle":8s} {fid_or:8.2f} {ssim_o_m:8.3f} {psnr_o_m:8.2f} '
              f'{"1.000":>10s} {f"{fo_m:.1f}+/-{fo_s:.1f}":>16s}')
    if not a.no_oracle:
        print('\noracle = decoder ceiling (PixCell of real UNI). pred-oracle gap = your head; '
              'oracle-GT gap = frozen PixCell.')
    print('FID unstable below ~500 tiles -- treat small-n FID as directional. '
          'FeatDist = paired Inception-2048 L2.')

    result = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'sample': a.sample,
        'train_samples': list(cfg.SAMPLES),
        'test_sample': a.test_sample,
        'fm': a.fm,
        'seed': a.seed,
        'split_seed': a.split_seed,
        'sampling_seed': a.sampling_seed,
        'seed_check_passed': bool(seed_checked) if a.seed_check else None,
        # k is a runtime attribute that changes the attention pattern without changing any
        # tensor shape, so a train/eval mismatch is invisible unless it is written down here
        'pairbias_k': cfg.PAIRBIAS_K,
        'guidance': _jsonable(a.guidance),
        'steps': a.steps,
        'n_eval': k,
        'n_cap': a.n,
        'checkpoint': None if a.no_pred else str(ck),
        'ckpt_own_cos': None if a.no_pred else _jsonable(ck_d.get('own_cos')),
        'ckpt_ep': None if a.no_pred else ck_d.get('ep'),
        'gene_vocab': None if a.no_pred else len(GENES),
        'genes_dropped_on_eval': None if a.no_pred else dropped_per_sample,
        'n_eval_per_sample': n_eval_per_sample,
        'pixcell_lora': a.pixcell_lora,
        'native_grid_unit': nat,
        'holdout_frac': _jsonable(cfg.CV_HOLDOUT_FRAC),
        'oracle': None if a.no_oracle else {
            'fid': _jsonable(fid_or),
            'ssim': _jsonable(ssim_o_m), 'ssim_std': _jsonable(ssim_o_s),
            'psnr': _jsonable(psnr_o_m), 'psnr_std': _jsonable(psnr_o_s),
            'unicos': 1.0,
            'featdist_mean': _jsonable(fo_m), 'featdist_std': _jsonable(fo_s),
        },
    }
    if not a.no_pred:
        result['pred'] = {
            'fid': _jsonable(fid_pr),
            'ssim': _jsonable(ssim_p_m), 'ssim_std': _jsonable(ssim_p_s),
            'psnr': _jsonable(psnr_p_m), 'psnr_std': _jsonable(psnr_p_s),
            'unicos': _jsonable(np.mean(coss)),
            'featdist_mean': _jsonable(fp_m), 'featdist_std': _jsonable(fp_s),
        }
    result_tag = f'{len(eval_samples)}pool_{eval_samples[0]}' if len(eval_samples) > 1 else a.sample
    results_dir = a.results_dir or os.path.join(cfg.OUT, 'benchmarks')
    json_path, log_path = write_results(result, results_dir, a.out_json,
                                        result_tag, a.guidance)
    print(f'\nwrote {json_path}\nappended -> {log_path}')

    if a.keep_renders:
        print(f'kept renders -> {a.keep_renders}/{{gt,oracle,pred}}')
    else:
        for dd in [d_gt] + ([d_or] if d_or else []) + ([d_pr] if d_pr else []):
            shutil.rmtree(dd)


if __name__ == '__main__':
    main()

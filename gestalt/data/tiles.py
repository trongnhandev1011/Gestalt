#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tile extraction.

Three layouts, selected with --layout:

    grid        non-overlapping lattice (step = crop). Used for the HEST lung and breast
                cohorts, and to assemble 1024px fields.
    overlap     50% overlap, empty windows skipped.
    singlecell  one window per cell, centred on that cell and conditioned on that cell's
                expression alone.

Xenium Prime (P1) takes its cell centroids from the raw 10x cells.parquet, supplied with
--coords-parquet. Pass it whenever the slide is P1.

Crops are normalised by physical size: crop = round(field_um / native_um_per_px), resized to
256, so every slide contributes the same physical field of view.
"""
import os, json, glob, argparse, time

import numpy as np
import pandas as pd
import h5py, cv2, tifffile
import torch
from PIL import Image

RAW_ROOT = os.environ.get('GESTALT_RAW', './data/raw')
OUT = os.path.expanduser(os.environ.get(
    'SPATIALGENE_OUT', os.environ.get('GESTALT_OUT', './data/tiles')))
FOLDER = {'C1': 'Xenium_V1_hSkin_Melanoma_Base_FFPE',
          'C2': 'Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE',
          'P1': 'Xenium_Prime_Human_Skin_FFPE'}

CELL_CROP = 256                 # window size for the single-cell layout
OUT_PX = 256                    # embedding tile size, fixed by the loader
TARGET_UMPP = 0.5               # PixCell / UNI2-h / HEST scale
OVERLAP_FRAC = 0.5              # default overlap fraction for the overlap layout
DSC = 8
MIN_CELLS, MIN_TISSUE, WHITE = 1, 0.15, 215   # MIN_CELLS=1: their skip_empty is "0 cells"
CALIB_N, CALIB_NCC, MAX_RESID = 60, 0.98, 3.0
HEST_DARK = 15.0
os.makedirs(OUT, exist_ok=True)


def load_uni2h(dev):
    import timm
    from timm.data import resolve_data_config, create_transform
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
              reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, **kw).eval().to(dev)
    return m, create_transform(**resolve_data_config(m.pretrained_cfg, model=m))


def read_obs(f, key):
    if f'obs/{key}' not in f:
        return None
    a = f[f'obs/{key}']
    return None if isinstance(a, h5py.Group) else a[:]


def read_coords(f):
    if 'obsm/spatial_microns' in f:
        return f['obsm/spatial_microns'][:].astype(np.float64), 'spatial_microns'
    print('  !! spatial_microns missing, falling back to obsm/spatial (may be unscaled)')
    return f['obsm/spatial'][:].astype(np.float64), 'spatial'


def _fit_affine(S, P):
    mu, sd = S.mean(0), S.std(0); sd[sd == 0] = 1.0
    Z = (S - mu) / sd
    B, *_ = np.linalg.lstsq(np.hstack([Z, np.ones((len(Z), 1))]), P, rcond=None)
    A = np.empty((3, 2)); A[:2] = B[:2] / sd[:, None]; A[2] = B[2] - (mu / sd) @ B[:2]
    return A


def _decompose(A):
    L = A[:2]; sx, sy = np.linalg.norm(L[0]), np.linalg.norm(L[1])
    scale = np.sqrt(abs(np.linalg.det(L)))
    theta = np.degrees(np.arctan2(L[0, 1], L[0, 0]))
    aniso = max(sx, sy) / max(min(sx, sy), 1e-12)
    return scale, theta, aniso


def match_crops(slide, d, sp, cid, he, n_want):
    heg = cv2.cvtColor(he, cv2.COLOR_RGB2GRAY)
    heg_ds = cv2.resize(heg, (heg.shape[1] // DSC, heg.shape[0] // DSC), interpolation=cv2.INTER_AREA)
    idx = {c: i for i, c in enumerate(cid)}
    tifs = sorted(glob.glob(os.path.join(d, 'cell_patch_256_aux/input/cell_images/*_original.tif')))
    if not tifs:
        raise RuntimeError(f'{slide}: no cell crops, cannot calibrate')
    S, P, order = [], [], np.random.RandomState(0).permutation(len(tifs))
    for p in (tifs[i] for i in order):
        stem = os.path.basename(p).split('_original')[0]
        i = idx.get(stem, idx.get(stem.rsplit('-', 1)[0]))
        if i is None:
            continue
        tg = cv2.cvtColor(np.ascontiguousarray(tifffile.imread(p)[:, :, :3]), cv2.COLOR_RGB2GRAY)
        _, _, _, l = cv2.minMaxLoc(cv2.matchTemplate(
            heg_ds, cv2.resize(tg, (CELL_CROP // DSC, CELL_CROP // DSC), interpolation=cv2.INTER_AREA),
            cv2.TM_CCOEFF_NORMED))
        x0, y0 = max(0, l[0] * DSC - 300), max(0, l[1] * DSC - 300)
        sub = heg[y0:y0 + 600 + CELL_CROP, x0:x0 + 600 + CELL_CROP]
        if sub.shape[0] < CELL_CROP or sub.shape[1] < CELL_CROP:
            continue
        _, ncc, _, l2 = cv2.minMaxLoc(cv2.matchTemplate(sub, tg, cv2.TM_CCOEFF_NORMED))
        if ncc < CALIB_NCC:
            continue
        S.append(sp[i]); P.append([x0 + l2[0] + CELL_CROP / 2, y0 + l2[1] + CELL_CROP / 2])
        if len(S) >= n_want:
            break
    return np.array(S), np.array(P)


def calibrate(slide, d, sp, cid, he):
    S, P = match_crops(slide, d, sp, cid, he, CALIB_N)
    if len(S) < 16:
        raise RuntimeError(f'{slide}: only {len(S)} crops matched at NCC>={CALIB_NCC}')
    rs = np.random.RandomState(1).permutation(len(S)); ntr = len(S) // 2
    tr, ho = rs[:ntr], rs[ntr:]
    A = _fit_affine(S[tr], P[tr])
    res = np.linalg.norm(np.hstack([S[ho], np.ones((len(ho), 1))]) @ A - P[ho], axis=1)
    scale, theta, aniso = _decompose(A)
    print(f'  affine: {len(S)} crops ({ntr} fit/{len(ho)} hold)  scale={scale:.6f} he-px/um  '
          f'rot={theta:+.2f}  aniso={aniso:.4f}')
    print(f'    HOLDOUT residual median={np.median(res):.2f}px max={res.max():.2f}px')
    if aniso > 1.05:
        print(f'    !! anisotropy {aniso:.3f}')
    if np.median(res) > MAX_RESID:
        raise RuntimeError(f'{slide}: holdout residual {np.median(res):.1f}px > {MAX_RESID}.')
    return _fit_affine(S, P)


def cell_filter(f, min_counts):
    n = f['obsm/spatial_microns'].shape[0] if 'obsm/spatial_microns' in f else f['obsm/spatial'].shape[0]
    good = np.ones(n, bool)
    tc = read_obs(f, 'transcript_counts')
    if tc is None:
        tc = read_obs(f, 'total_counts')
    if tc is not None and min_counts > 0:
        good &= (tc >= min_counts)
    print(f'  keeping {good.sum()}/{n} cells')
    return good


@torch.no_grad()
def embed_batch(crops, model, tf, dev):
    x = torch.stack([tf(Image.fromarray(c)) for c in crops]).to(dev)
    with torch.autocast('cuda', enabled=(dev.type == 'cuda')):
        return model(x).float().cpu().numpy().astype(np.float32)


def load_clean_coords(parquet_path, cid):
    """Load P1 cell centroids in microns from the raw 10x cells.parquet.

    Centroids are joined on cell_id and reindexed onto obs order. Raises if any cell fails to
    join or if the values are not on a micron scale.
    """
    if not os.path.exists(parquet_path):
        raise SystemExit(f'--coords-parquet not found: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    low = {c.lower(): c for c in df.columns}
    xcol = low.get('x_centroid') or low.get('x')
    ycol = low.get('y_centroid') or low.get('y')
    idcol = low.get('cell_id') or low.get('cell') or low.get('id')
    if not (xcol and ycol and idcol):
        raise RuntimeError(f'cells.parquet missing x/y_centroid or cell_id; has {list(df.columns)}')
    ids = df[idcol].astype(str).to_numpy()
    xy = df[[xcol, ycol]].to_numpy(np.float64)
    if not np.isfinite(xy).all():
        raise RuntimeError('cells.parquet centroids contain nan/inf')
    amax = float(np.abs(xy).max())
    if not (1.0 < amax < 1e6):
        raise RuntimeError(f'cells.parquet centroid absmax={amax:.4g} -- not microns as expected')
    lut = {c: i for i, c in enumerate(ids)}
    rows = np.fromiter((lut.get(c, -1) for c in cid), dtype=np.int64, count=len(cid))
    miss = int((rows < 0).sum())
    if miss:
        ex = [c for c, r in zip(cid, rows) if r < 0][:5]
        raise RuntimeError(f'{miss}/{len(cid)} obs cells absent from cells.parquet (e.g. {ex}); '
                           f'cell_id format mismatch -- run inspect_p1.py --raw to compare, do NOT build')
    print(f'  clean coords: {len(cid)}/{len(cid)} obs cells joined to {os.path.basename(parquet_path)} '
          f'({xcol},{ycol}) absmax={amax:.1f}um  parquet_rows={len(ids)}')
    return xy[rows], 'cells_parquet_microns'


def build(slide, model, tf, dev, args):
    """Dispatch on --layout. The three bodies are the original, unmodified builders."""
    if slide == 'P1' and not args.coords_parquet:
        # P1's own embedded coordinates (obsm/spatial_microns) are overflow-corrupt
        # (+/-1e21..22): confirmed directly, this produces a degenerate affine fit
        # (scale=0.0) and a confusing "crop=0px absurd" failure several steps later if not
        # caught here. --coords-parquet is optional for C1/C2 but required for P1.
        raise SystemExit('P1 requires --coords-parquet (its own embedded coordinates are '
                          'corrupt). Get it with: python -m gestalt.data.download --cohort '
                          'xenium --slides P1')
    if args.layout == 'singlecell':
        return _build_singlecell(slide, model, tf, dev, args)
    return _build_lattice(slide, model, tf, dev, args)


def _build_lattice(slide, model, tf, dev, args):
    out = args.out or OUT
    os.makedirs(out, exist_ok=True)
    d = os.path.join(args.root, FOLDER[slide]); src = os.path.join(d, 'adata.h5ad')
    print(f'\n=== {slide} (overlap {args.overlap:.0%}, skip_empty={not args.keep_empty}) -> {out} ===')
    with h5py.File(src, 'r') as f:
        cid = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f['obs/cell_id'][:]])
        if args.coords_parquet:
            sp, coord_key = load_clean_coords(args.coords_parquet, cid)
        else:
            sp, coord_key = read_coords(f)
        he = f['uns/he_image'][:, :, :3]
        good = cell_filter(f, args.min_counts)
    H, W = he.shape[:2]

    A = calibrate(slide, d, sp, cid, he)
    px = np.hstack([sp, np.ones((len(sp), 1))]) @ A
    scale, _, _ = _decompose(A); umpp = 1.0 / scale
    crop = int(round(OUT_PX * args.target_umpp / umpp))      # native window = OUT_PX*target_umpp
    step = max(1, int(round(crop * (1 - args.overlap))))    # overlap=0 -> step=crop (lattice)
    if crop < 16:
        raise RuntimeError(f'{slide}: crop={crop}px absurd')
    print(f'  native {umpp:.4f} um/px -> crop {crop}px step {step}px -> resize {OUT_PX} '
          f'(={OUT_PX*args.target_umpp:.0f}um @ {args.target_umpp}um/px)')

    keep_ids = np.where(good)[0]
    kpx = px[keep_ids]
    min_cells = 1 if not args.keep_empty else 0

    # sliding windows on the native H&E grid.
    ys = list(range(0, H - crop + 1, step)); xs = list(range(0, W - crop + 1, step))
    if args.overlap > 0:            # ragged boundary window ONLY when overlapping; at 0 it would
        if ys and ys[-1] != H - crop:   # alias the previous window's gy=y0//step key -> collision
            ys.append(H - crop)
        if xs and xs[-1] != W - crop:
            xs.append(W - crop)

    kept, meta, t0 = [], [], time.time()
    for wi, y0 in enumerate(ys):
        for xj, x0 in enumerate(xs):
            inwin = ((kpx[:, 0] >= x0) & (kpx[:, 0] < x0 + crop) &
                     (kpx[:, 1] >= y0) & (kpx[:, 1] < y0 + crop))
            rows = keep_ids[inwin]
            if len(rows) < min_cells:
                continue                                  # skip-empty (their skip_empty_patch)
            patch = he[y0:y0 + crop, x0:x0 + crop]
            if (patch > WHITE).all(-1).mean() > (1 - MIN_TISSUE):
                continue                                  # tissue guard (documented deviation)
            gy, gx = y0 // step, x0 // step                # unique per window (native=step)
            local = ((px[rows] - [x0, y0]) * (OUT_PX / crop)).astype(np.float32)
            kept.append((y0, x0))
            meta.append({'gy': int(gy), 'gx': int(gx), 'x0': int(x0),
                         'cell_idx': rows.tolist(), 'local_xy': local.tolist()})
    if args.max_tiles and len(kept) > args.max_tiles:
        sel = np.random.RandomState(0).choice(len(kept), args.max_tiles, replace=False)
        sel.sort()
        kept = [kept[i] for i in sel]
        meta = [meta[i] for i in sel]
        print(f'  --max-tiles: kept {len(kept)} windows at random')
    N = len(kept)
    if N == 0:
        raise RuntimeError(f'{slide}: 0 windows kept')
    nc = np.array([len(m['cell_idx']) for m in meta])
    # (y0//step, x0//step) is unique because y0,x0 are multiples of step by construction -> safe.
    assert len({(m['gy'], m['gx']) for m in meta}) == N, 'grid key collision -- check step'
    print(f'  {N} windows | cells/window p50={int(np.median(nc))} p95={int(np.quantile(nc,.95))} '
          f'max={nc.max()} | mean windows/cell~{nc.sum()/max(good.sum(),1):.2f}x (overlap inflation)')

    # QC gate (unchanged)
    gray = he.mean(-1)
    ii = np.clip(kpx.astype(int), [0, 0], [W - 1, H - 1])
    on = ~(he[ii[:, 1], ii[:, 0]] > WHITE).all(-1)
    yy, xx = np.where(~(he[::16, ::16] > WHITE).all(-1))
    k = np.random.RandomState(0).choice(len(yy), min(50000, len(yy)), replace=False)
    dark = gray[yy[k]*16, xx[k]*16].mean() - gray[ii[on, 1], ii[on, 0]].mean()
    print(f'  QC: centroid darker by {dark:+.1f} (HEST band +21.9..+33.1)')
    if dark < HEST_DARK:
        raise RuntimeError(f'{slide}: darker-by {dark:+.1f} < {HEST_DARK}.')
    if args.dry_run:
        print('  --dry-run: nothing written.'); return

    suf = f'tiles{OUT_PX}_p5'
    with h5py.File(f'{out}/{slide}_{suf}.h5', 'w') as gt, \
         h5py.File(f'{out}/{slide}_{suf}_uni.h5', 'w') as gu:
        gt.create_dataset('coords', data=np.array([(y, x, 0) for y, x in kept], np.int64))
        gt.attrs['native'] = step               # loader grid unit = step (overlap-safe)
        gt.attrs['crop_px'] = crop
        gt.attrs['out_px'] = OUT_PX
        gt.attrs['native_umpp'] = float(umpp)
        gt.attrs['target_umpp'] = float(args.target_umpp)
        gt.attrs['overlap_frac'] = args.overlap
        gt.attrs['skip_empty'] = int(not args.keep_empty)
        gt.attrs['affine'] = A.astype(np.float64).ravel()
        gt.attrs['coord_key'] = coord_key
        gt.attrs['darker_by'] = float(dark)
        imds = gt.create_dataset('images', shape=(N, OUT_PX, OUT_PX, 3), dtype='uint8',
                                 chunks=(1, OUT_PX, OUT_PX, 3), compression='gzip',
                                 compression_opts=4) if not args.no_images else None
        uds = gu.create_dataset('uni', shape=(N, 1536), dtype='float32')

        buf, bi, norms = [], [], []
        for i, (y0, x0) in enumerate(kept):
            patch = cv2.resize(np.ascontiguousarray(he[y0:y0 + crop, x0:x0 + crop]),
                               (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
            if imds is not None:
                imds[i] = patch
            buf.append(patch); bi.append(i)
            if len(buf) == args.bs or i == N - 1:
                emb = embed_batch(buf, model, tf, dev)
                for j, kk in enumerate(bi):
                    uds[kk] = emb[j]
                norms.append(np.linalg.norm(emb, axis=1)); buf, bi = [], []
                if (i // args.bs) % 20 == 0:
                    print(f'    {i+1}/{N} ({(i+1)/max(time.time()-t0,1e-9):.0f} win/s)')

    json.dump(meta, open(f'{out}/{slide}_meta.json', 'w'))
    link = f'{out}/{slide}_adata.h5ad'
    if not os.path.exists(link):
        os.symlink(src, link)
    print(f'  done. crop={crop} step={step} |uni| mean={float(np.concatenate(norms).mean()):.2f}')


def _build_singlecell(slide, model, tf, dev, args):
    out = args.out or OUT
    os.makedirs(out, exist_ok=True)
    d = os.path.join(args.root, FOLDER[slide]); src = os.path.join(d, 'adata.h5ad')
    print(f'\n=== {slide} single-cell (one 256px window per cell, centred on that cell, '
          f"that cell's own expression only) -> {out} ===")
    with h5py.File(src, 'r') as f:
        cid = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f['obs/cell_id'][:]])
        if args.coords_parquet:
            sp, coord_key = load_clean_coords(args.coords_parquet, cid)
        else:
            sp, coord_key = read_coords(f)
        he = f['uns/he_image'][:, :, :3]
        good = cell_filter(f, args.min_counts)
    H, W = he.shape[:2]

    A = calibrate(slide, d, sp, cid, he)
    px = np.hstack([sp, np.ones((len(sp), 1))]) @ A
    scale, _, _ = _decompose(A); umpp = 1.0 / scale
    crop = int(round(OUT_PX * args.target_umpp / umpp))
    if crop < 16:
        raise RuntimeError(f'{slide}: crop={crop}px absurd')
    print(f'  native {umpp:.4f} um/px -> crop {crop}px (centered per cell) -> resize {OUT_PX} '
          f'(={OUT_PX*args.target_umpp:.0f}um @ {args.target_umpp}um/px)')

    keep_ids = np.where(good)[0]
    kpx = px[keep_ids]
    if args.n_cap and len(keep_ids) > args.n_cap:
        sel = np.random.RandomState(0).choice(len(keep_ids), args.n_cap, replace=False)
        keep_ids, kpx = keep_ids[sel], kpx[sel]
        print(f'  --n-cap {args.n_cap}: subsampled from {good.sum()} cells')

    half = crop / 2.0
    kept, meta, t0, n_edge, n_bg = [], [], time.time(), 0, 0
    for c, (cx, cy) in zip(keep_ids, kpx):
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        if x0 < 0 or y0 < 0 or x0 + crop > W or y0 + crop > H:
            n_edge += 1; continue                    # window would run off the slide -- skip
        patch = he[y0:y0 + crop, x0:x0 + crop]
        if (patch > WHITE).all(-1).mean() > (1 - MIN_TISSUE):
            n_bg += 1; continue                       # same tissue guard as the multi-cell script
        local = ((px[[c]] - [x0, y0]) * (OUT_PX / crop)).astype(np.float32)
        kept.append((y0, x0))
        # native=1 (raw rounded pixel origin as the grid key) -- windows aren't on a fixed step,
        # so there's no shared "native" unit; a per-cell-unique pixel origin serves the same role
        # the dataset's grid2row lookup needs (see module docstring).
        meta.append({'gy': int(y0), 'gx': int(x0), 'x0': int(x0),
                     'cell_idx': [int(c)], 'local_xy': local.tolist()})
    N = len(kept)
    if N == 0:
        raise RuntimeError(f'{slide}: 0 cell-centered windows kept')
    assert len({(m['gy'], m['gx']) for m in meta}) == N, 'grid key collision -- two cells rounded to the same window origin'
    print(f'  {N} cell-centered windows kept ({n_edge} dropped: window off-slide, '
          f'{n_bg} dropped: mostly background)')

    # QC gate, same as build_tiles_overlap.py: is the cell's own pixel actually darker than a
    # random tissue background sample (sanity check the calibration landed on real cells, not glass).
    gray = he.mean(-1)
    ii = np.clip(kpx.astype(int), [0, 0], [W - 1, H - 1])
    on = ~(he[ii[:, 1], ii[:, 0]] > WHITE).all(-1)
    yy, xx = np.where(~(he[::16, ::16] > WHITE).all(-1))
    k = np.random.RandomState(0).choice(len(yy), min(50000, len(yy)), replace=False)
    dark = gray[yy[k]*16, xx[k]*16].mean() - gray[ii[on, 1], ii[on, 0]].mean()
    print(f'  QC: centroid darker by {dark:+.1f} (HEST band +21.9..+33.1)')
    if dark < HEST_DARK:
        raise RuntimeError(f'{slide}: darker-by {dark:+.1f} < {HEST_DARK}.')
    if args.dry_run:
        print('  --dry-run: nothing written.'); return

    suf = f'tiles{OUT_PX}_p5'
    with h5py.File(f'{out}/{slide}_{suf}.h5', 'w') as gt, \
         h5py.File(f'{out}/{slide}_{suf}_uni.h5', 'w') as gu:
        gt.create_dataset('coords', data=np.array([(y, x, 0) for y, x in kept], np.int64))
        gt.attrs['native'] = 1                   # see meta['gy']/['gx'] comment above
        gt.attrs['crop_px'] = crop
        gt.attrs['out_px'] = OUT_PX
        gt.attrs['native_umpp'] = float(umpp)
        gt.attrs['target_umpp'] = float(args.target_umpp)
        gt.attrs['protocol'] = 'single_cell'
        gt.attrs['affine'] = A.astype(np.float64).ravel()
        gt.attrs['coord_key'] = coord_key
        gt.attrs['darker_by'] = float(dark)
        imds = gt.create_dataset('images', shape=(N, OUT_PX, OUT_PX, 3), dtype='uint8',
                                 chunks=(1, OUT_PX, OUT_PX, 3), compression='gzip',
                                 compression_opts=4) if not args.no_images else None
        uds = gu.create_dataset('uni', shape=(N, 1536), dtype='float32')

        buf, bi, norms = [], [], []
        for i, (y0, x0) in enumerate(kept):
            patch = cv2.resize(np.ascontiguousarray(he[y0:y0 + crop, x0:x0 + crop]),
                               (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
            if imds is not None:
                imds[i] = patch
            buf.append(patch); bi.append(i)
            if len(buf) == args.bs or i == N - 1:
                emb = embed_batch(buf, model, tf, dev)
                for j, kk in enumerate(bi):
                    uds[kk] = emb[j]
                norms.append(np.linalg.norm(emb, axis=1)); buf, bi = [], []
                if (i // args.bs) % 50 == 0:
                    print(f'    {i+1}/{N} ({(i+1)/max(time.time()-t0,1e-9):.0f} win/s)')

    json.dump(meta, open(f'{out}/{slide}_meta.json', 'w'))
    link = f'{out}/{slide}_adata.h5ad'
    if not os.path.exists(link):
        os.symlink(src, link)
    print(f'  done. N={N} crop={crop} |uni| mean={float(np.concatenate(norms).mean()):.2f}')


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--slides', nargs='+', required=True)
    ap.add_argument('--layout', choices=['grid', 'overlap', 'singlecell'], default='grid',
                    help='grid = non-overlapping lattice; overlap = 50%% overlap, skip-empty '
                         '; singlecell = one window per cell')
    ap.add_argument('--out', default=None,
                    help='output dir (default $SPATIALGENE_OUT, then $GESTALT_OUT)')
    ap.add_argument('--root', default=RAW_ROOT,
                    help='directory holding the raw 10x bundles (default $GESTALT_RAW)')
    ap.add_argument('--coords-parquet', default=None,
                    help="REQUIRED for Xenium Prime (P1): clean micron centroids from the raw "
                         "10x cells.parquet. Supply this for P1.")
    ap.add_argument('--target-umpp', type=float, default=0.5,
                    help='resolution to normalise crops to. Use 0.25 for the native-40x '
                         'experiments (C1/C2/P1); 0.5 is the default 20x pipeline.')
    ap.add_argument('--field-um', type=float, default=128.0, help='physical field per tile')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--min-counts', type=int, default=0)
    ap.add_argument('--keep-empty', action='store_true')
    ap.add_argument('--max-tiles', type=int, default=None,
                    help='keep at most this many windows, drawn at random across the slide')
    ap.add_argument('--no-images', action='store_true',
                    help='write embeddings only, no image dataset')
    ap.add_argument('--dry-run', action='store_true',
                    help='scan and report, write nothing')
    a = ap.parse_args()
    a.overlap = 0.5 if a.layout == 'overlap' else 0.0
    if a.out is None:
        a.out = OUT  # module-level default: checks SPATIALGENE_OUT before GESTALT_OUT
    return a


def main():
    a = _parse_args()
    if 'P1' in a.slides and not a.coords_parquet:
        # fail before the expensive UNI2-h load, not just before the calibration step -- see
        # the identical check in build() for why this is required, not optional, for P1.
        raise SystemExit('P1 requires --coords-parquet (its own embedded coordinates are '
                          'corrupt). Get it with: python -m gestalt.data.download --cohort '
                          'xenium --slides P1')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tf = load_uni2h(dev)
    for slide in a.slides:
        build(slide, model, tf, dev, a)


if __name__ == '__main__':
    main()

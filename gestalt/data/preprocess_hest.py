#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch and normalise HEST-1k Xenium samples (lung and breast cohorts).
"""
import argparse
import glob
import json
import logging
import os

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

logging.getLogger('tifffile').setLevel(logging.ERROR)  # mute benign "series shape" warning

PX_T = 0.5     # PixCell training resolution: 20x == 0.5 um/px (verified vs PixCell paper)
TILE = 256     # PixCell-256 patch size; UNI resizes 256->224 itself via timm transform
BATCH = 64     # UNI encode batch size
QV_MIN = 20    # Xenium transcript quality-value threshold
# transcripts whose feature_name starts with any of these are control probes, not genes
CTRL_PREFIXES = ('NegControl', 'BLANK', 'antisense', 'UnassignedCodeword', 'DeprecatedCodeword', 'Intergenic')
# Sentinel used by the transcript table for a transcript not assigned to any cell.
UNASSIGNED_CELL_ID = '-1'


# ============================ DOWNLOAD ============================
def download_hest_samples(sample_ids, root):
    """Pull each sample's raw WSI/transcripts/segmentation/metadata from the HEST HF dataset repo."""
    import zipfile
    from huggingface_hub import snapshot_download
    patterns = [f'*{sid}[_.]**' for sid in sample_ids]
    snapshot_download(repo_id='MahmoodLab/hest', repo_type='dataset',
                       allow_patterns=patterns, local_dir=root)
    seg_dir = os.path.join(root, 'cellvit_seg')
    if os.path.exists(seg_dir):
        # Extract only the archives belonging to the requested samples.
        for sid in sample_ids:
            f = os.path.join(seg_dir, f'{sid}_cellvit_seg.geojson.zip')
            if os.path.exists(f):
                with zipfile.ZipFile(f) as z:
                    z.extractall(seg_dir)


# ============================ STAGE A: TILE EXTRACTION ============================
def _full_res(path):
    """Largest (full-res) level of a possibly-pyramidal TIFF, lazily if possible."""
    import tifffile
    try:
        s = tifffile.TiffFile(path).series[0]
        lvl = (max(s.levels, key=lambda L: int(np.prod(L.shape[:2])))
               if hasattr(s, 'levels') and len(s.levels) > 1 else s)
    except RuntimeError:
        # some pyramids cannot be grouped into a series ('incompatible keyframe'). Their pages
        # still read when indexed from a fresh handle -- the failed series attempt leaves a
        # keyframe behind that breaks any further access through the same one.
        pages = tifffile.TiffFile(path).pages
        lvl = max((pages[i] for i in range(len(pages))),
                  key=lambda p: int(np.prod(p.shape[:2])))
    try:
        return lvl.asarray(out='memmap')
    except Exception:
        return lvl.asarray()


def _tissue_mask(root, sample, gh, gw, native):
    """Tile-grid bool mask from the tissue_seg contour, or all-True (whole WSI) if none exists."""
    import cv2
    gj = glob.glob(f'{root}/tissue_seg/{sample}_contours.geojson')
    if not gj:
        return np.ones((gh, gw), bool), False
    m = np.zeros((gh, gw), np.uint8)
    for feat in json.load(open(gj[0])).get('features', []):
        for poly in feat['geometry']['coordinates']:
            ring = poly[0] if isinstance(poly[0][0], (list, tuple)) else poly
            pts = (np.array(ring) / native).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
    return m.astype(bool), True


def extract_tiles(sample, root, out, overlap=0.0):
    """Stage A. Returns (sample, status, n_tiles)."""
    import cv2
    out_h5 = f'{out}/{sample}_tiles256_p5.h5'
    if os.path.exists(out_h5):                        # resume: already done
        with h5py.File(out_h5, 'r') as f:
            return sample, 'cached', f['images'].shape[0]
    meta = json.load(open(f'{root}/metadata/{sample}.json'))
    px = meta['pixel_size_um_embedded']
    if px is None or px >= 1.0:
        # the embedded resolution is a 1.0 placeholder on some slides; HEST's own estimate is
        # the real value there
        px = meta.get('pixel_size_um_estimated') or meta.get('pixel_size')
    if px is None or px >= 1.0:
        return sample, 'skip:px', 0
    tif = f'{root}/wsis/{sample}.tif'
    if not os.path.exists(tif):
        return sample, 'skip:no_wsi', 0
    native = round(TILE * PX_T / px)                  # native px per tile before resize to 256
    arr = _full_res(tif)
    H, W = arr.shape[:2]
    gh, gw = H // native, W // native
    mask, masked = _tissue_mask(root, sample, gh, gw, native)
    # --overlap fraction f gives step = native*(1-f); f=0 is a disjoint lattice. The tissue
    # mask is evaluated on the disjoint grid, so a window is kept when the disjoint cell
    # containing its top-left corner is tissue.
    step = max(1, int(round(native * (1.0 - overlap))))
    ys = list(range(0, max(H - native + 1, 1), step))
    xs = list(range(0, max(W - native + 1, 1), step))
    tiles, coords = [], []
    for y in ys:
        for x in xs:
            i, j = min(y // native, gh - 1), min(x // native, gw - 1)
            if not mask[i, j]:
                continue
            crop = np.asarray(arr[y:y + native, x:x + native])
            if crop.shape[:2] != (native, native) or crop.ndim != 3:
                continue
            tiles.append(cv2.resize(crop[..., :3], (TILE, TILE), interpolation=cv2.INTER_AREA))
            coords.append((y, x, native))
    if not tiles:
        return sample, 'empty', 0
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('images', data=np.stack(tiles), compression='gzip')
        f.create_dataset('coords', data=np.array(coords))    # native-px, for stage C's cell join
        f.attrs.update(px=float(px), native=int(native), tissue_masked=masked, sample=sample)
    return sample, ('ok' if masked else 'ok_NOMASK'), len(tiles)


# ============================ STAGE B: UNI2-h ENCODING ============================
def _load_uni(device):
    import torch, timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
              reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, **kw).eval().to(device)
    tf = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, tf


def encode_uni(out, device=None, batch=BATCH):
    """Stage B. Encodes every {out}/*_tiles256_p5.h5 not already encoded."""
    import torch
    from PIL import Image
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    uni, tf = _load_uni(dev)

    def _encode(imgs_uint8):
        x = torch.stack([tf(Image.fromarray(im)) for im in imgs_uint8]).to(dev)
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
            return uni(x).float().cpu().numpy()

    for h5path in sorted(glob.glob(f'{out}/*_tiles256_p5.h5')):
        sample = os.path.basename(h5path).split('_tiles')[0]
        uni_path = h5path.replace('.h5', '_uni.h5')
        if os.path.exists(uni_path):                  # resume
            print(f'{sample}: cached')
            continue
        with h5py.File(h5path, 'r') as f:
            imgs = f['images'][:]
        embs = np.concatenate([_encode(imgs[i:i + batch]) for i in range(0, len(imgs), batch)])
        with h5py.File(uni_path, 'w') as f:
            f.create_dataset('uni', data=embs.astype(np.float32))
        n = np.linalg.norm(embs, axis=1)
        print(f'{sample}: {embs.shape} | norm {n.mean():.2f} (want ~15.3-15.5) range {n.min():.1f}-{n.max():.1f}')


# ============================ STAGE C: EXPRESSION + META ============================
def build_expression(sample, root, out):
    """Stage C. transcripts parquet -> {sample}_adata.h5ad + {sample}_meta.json.
    Requires stage A's {sample}_tiles256_p5.h5 to already exist (reads its coords/native)."""
    tx = pd.read_parquet(f'{root}/transcripts/{sample}_transcripts.parquet',
                         columns=['cell_id', 'feature_name', 'he_x', 'he_y', 'qv'])
    # anndata's h5ad writer serialises plain Python str, not pandas' nullable 'string'
    # extension dtype, which some HEST source parquets use for these two columns.
    tx['cell_id'] = tx['cell_id'].astype(str)
    tx['feature_name'] = tx['feature_name'].astype(str)
    tx = tx[(tx.qv >= QV_MIN) & (tx.cell_id != UNASSIGNED_CELL_ID) & tx.he_x.notna()]
    tx = tx[~tx.feature_name.str.startswith(CTRL_PREFIXES)]

    with h5py.File(f'{out}/{sample}_tiles256_p5.h5', 'r') as f:
        coords = f['coords'][:]

    genes = sorted(tx.feature_name.unique())
    gidx = {g: i for i, g in enumerate(genes)}
    cell_ids = tx.cell_id.unique()
    cidx = {c: i for i, c in enumerate(cell_ids)}
    raw = csr_matrix((np.ones(len(tx), np.float32),
                      (tx.cell_id.map(cidx).values, tx.feature_name.map(gidx).values)),
                     shape=(len(cell_ids), len(genes))).tocsr()
    cellxy = tx.groupby('cell_id')[['he_x', 'he_y']].mean().loc[cell_ids].values.astype(np.float32)

    # norm = log1p(median-scaled total), i.e. scanpy normalize_total + log1p, done in numpy.
    tot = np.asarray(raw.sum(1)).ravel()
    tot[tot == 0] = 1
    scale = np.median(tot)
    norm = raw.multiply((scale / tot)[:, None]).tocsr()
    norm.data = np.log1p(norm.data)

    A = ad.AnnData(X=norm)
    A.var_names = genes
    A.obs['cell_id'] = cell_ids
    A.obsm['he_xy'] = cellxy
    A.layers['norm'] = norm
    A.layers['raw'] = raw
    A.write(f'{out}/{sample}_adata.h5ad')

    cx, cy = cellxy[:, 0], cellxy[:, 1]
    meta = []
    for y0, x0, nat in coords:
        m = (cx >= x0) & (cx < x0 + nat) & (cy >= y0) & (cy < y0 + nat)
        ci = np.where(m)[0]
        if len(ci) == 0:
            continue
        lx = (cx[ci] - x0) / nat * 256.0
        ly = (cy[ci] - y0) / nat * 256.0
        meta.append(dict(gy=int(y0 // nat), gx=int(x0 // nat), y0=int(y0), x0=int(x0),
                         n=len(ci), cell_idx=ci.tolist(),
                         local_xy=np.stack([lx, ly], 1).tolist()))
    json.dump(meta, open(f'{out}/{sample}_meta.json', 'w'))
    mean_cells = float(np.mean([m['n'] for m in meta])) if meta else 0.0
    return sample, A.n_obs, len(genes), len(meta), mean_cells


# ================================ main ================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', required=True, help='comma-separated HEST sample IDs')
    ap.add_argument('--root', required=True, help='raw HEST download root (metadata/wsis/transcripts/tissue_seg live here)')
    ap.add_argument('--out', required=True, help="output dir for {sample}_adata.h5ad etc -- this project's cfg.OUT convention")
    ap.add_argument('--download', action='store_true', help='pull raw data for --samples from the HEST HF dataset repo first')
    ap.add_argument('--stages', default='A,B,C', help='comma list of stages to run (A=tiles, B=uni encode, C=expression+meta)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--overlap', type=float, default=0.0,
                    help='window overlap fraction for stage A; 0.0 (default) = the original '
                         'disjoint lattice, 0.5 = 50%% overlap (step = native/2). '
                         'ALWAYS write overlap tiles to a SEPARATE --out; they must not overwrite '
                         'the non-overlap tiles every existing lung number was computed on.')
    args = ap.parse_args()

    samples = args.samples.split(',')
    os.makedirs(args.out, exist_ok=True)
    stages = set(args.stages.split(','))

    if args.download:
        print(f'=== downloading {len(samples)} samples to {args.root} ===')
        download_hest_samples(samples, args.root)

    if 'A' in stages:
        print('=== STAGE A: extract tiles ===')
        for s in samples:
            try:
                r = extract_tiles(s, args.root, args.out, args.overlap)
            except Exception as e:
                r = (s, f'FAIL:{type(e).__name__}:{e}', 0)
            print(f'{r[0]:12s} {r[1]:20s} {r[2]:>6}')

    if 'B' in stages:
        print('\n=== STAGE B: encode with UNI2-h ===')
        encode_uni(args.out, device=args.device)

    if 'C' in stages:
        print('\n=== STAGE C: build expression + meta ===')
        print(f"{'sample':12s} {'cells':>8s} {'genes':>6s} {'tiles':>6s} {'cells/tile':>10s}")
        for s in samples:
            try:
                r = build_expression(s, args.root, args.out)
                print(f'{r[0]:12s} {r[1]:>8d} {r[2]:>6d} {r[3]:>6d} {r[4]:>10.1f}')
            except Exception as e:
                print(f'{s:12s} FAILED {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()

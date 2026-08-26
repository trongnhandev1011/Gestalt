#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Re-encode cached 256px crops through a foundation model listed in FM_REGISTRY.
"""
import os
import argparse
import time


import h5py
import numpy as np
import torch
from PIL import Image
from huggingface_hub import login, get_token

from ..core.config import FM_REGISTRY

OUT = os.path.expanduser(os.environ.get(
    'SPATIALGENE_OUT', os.environ.get('GESTALT_OUT', './data/tiles')))


def load_uni2h(dev):
    import timm
    from timm.data import resolve_data_config, create_transform
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
              reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, **kw).eval().to(dev)
    tf = create_transform(**resolve_data_config(m.pretrained_cfg, model=m))
    return (lambda x: m(x)), tf


def load_uni_v1(dev):
    """MahmoodLab/UNI, ViT-L/16, 1024-d CLS embedding."""
    import timm
    from timm.data import resolve_data_config, create_transform
    m = timm.create_model('hf-hub:MahmoodLab/UNI', pretrained=True, init_values=1e-5,
                          dynamic_img_size=True, num_classes=0).eval().to(dev)
    tf = create_transform(**resolve_data_config(m.pretrained_cfg, model=m))
    return (lambda x: m(x)), tf


def load_conch(dev):
    """MahmoodLab/CONCH, CLIP-style vision-language model. Visual tower only, 512-d
    projected image embedding (proj_contrast=False, normalize=False: the raw projected
    vector, not the contrastive-space normalized one)."""
    from conch.open_clip_custom import create_model_from_pretrained
    m, preprocess = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/CONCH')
    m = m.eval().to(dev)
    fwd = lambda x: m.encode_image(x, proj_contrast=False, normalize=False)
    return fwd, preprocess


def load_virchow2(dev):
    """paige-ai/Virchow2. 2560-d = 1280-d CLS token concatenated with the mean of the 256
    patch tokens (tokens 5: -- the first 4 are register tokens, not patches)."""
    import timm
    from timm.data import resolve_data_config, create_transform
    from timm.layers import SwiGLUPacked
    m = timm.create_model('hf-hub:paige-ai/Virchow2', pretrained=True,
                          mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().to(dev)
    tf = create_transform(**resolve_data_config(m.pretrained_cfg, model=m))

    def fwd(x):
        out = m(x)                                        # (B, 261, 1280)
        cls, patches = out[:, 0], out[:, 5:]
        return torch.cat([cls, patches.mean(1)], dim=-1)  # (B, 2560)
    return fwd, tf


LOADERS = {'uni2h': load_uni2h, 'uni_v1': load_uni_v1, 'conch': load_conch, 'virchow2': load_virchow2}


@torch.no_grad()
def embed_batch(crops, fwd, tf, dev):
    x = torch.stack([tf(Image.fromarray(c)) for c in crops]).to(dev)
    with torch.autocast('cuda', enabled=(dev.type == 'cuda')):
        return fwd(x).float().cpu().numpy().astype(np.float32)


def extract(slide, fm, fwd, tf, dev, bs):
    reg = FM_REGISTRY[fm]
    src_path = f'{OUT}/{slide}_tiles256_p5.h5'
    out_path = f'{OUT}/{slide}_tiles256_p5_{reg["suffix"]}.h5'
    if os.path.exists(out_path):
        print(f'  {out_path} exists, skipping (delete it to re-extract).')
        return
    with h5py.File(src_path, 'r') as gt:
        if 'images' not in gt:
            raise RuntimeError(f'{slide}: {src_path} has no `images` dataset (built with '
                                f'--no-images?) -- need the raw crops to re-embed.')
        N = gt['images'].shape[0]
        print(f'\n=== {slide} -> {fm} ({N} tiles) ===')
        with h5py.File(out_path, 'w') as gu:
            uds = gu.create_dataset(reg['h5_key'], shape=(N, reg['dim']), dtype='float32')
            buf, bi, t0 = [], [], time.time()
            for i in range(N):
                buf.append(gt['images'][i])
                bi.append(i)
                if len(buf) == bs or i == N - 1:
                    emb = embed_batch(buf, fwd, tf, dev)
                    for j, kk in enumerate(bi):
                        uds[kk] = emb[j]
                    buf, bi = [], []
                    if (i // bs) % 20 == 0:
                        print(f'    {i + 1}/{N} ({(i + 1) / max(time.time() - t0, 1e-9):.0f} tiles/s)')
    print(f'  done. wrote {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fm', required=True, choices=list(LOADERS))
    ap.add_argument('--slides', nargs='+', default=['C2'])
    ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()

    tok = os.environ.get('HF_TOKEN') or get_token()
    if not tok:
        raise SystemExit('set HF_TOKEN in the environment')
    login(token=tok, add_to_git_credential=False)

    print(f'[out] SPATIALGENE_OUT resolved to: {OUT}')
    for s in a.slides:
        src_path = f'{OUT}/{s}_tiles256_p5.h5'
        with h5py.File(src_path, 'r') as g:
            print(f'  {s}: {g["images"].shape[0]} tiles in {src_path} '
                  f'(native={g.attrs.get("native")}, overlap_frac={g.attrs.get("overlap_frac", 0)})'
                  ' -- confirm this is the tile set you meant to re-embed.')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fwd, tf = LOADERS[a.fm](dev)
    for s in a.slides:
        extract(s, a.fm, fwd, tf, dev, a.bs)
    print('\ndone.')

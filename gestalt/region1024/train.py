#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train the 1024px region-synthesis mapper (16 sub-region embeddings in one pass).
"""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ..core.config import make_cfg_1024, samples_tag
from ..models.mapper import GeneTokenCells1024
from ..tile256.dataset import load_vocab, load_or_build_256
from .dataset import load_or_build_1024
from ..core.ddp_utils import ddp_setup, ddp_cleanup, barrier, reduce_sum_count


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None, help='per-GPU batch size (fields, not cells)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--test-sample', default=None,
                     help='slide to hold out entirely')
    ap.add_argument('--samples', default=None,
                     help='comma-separated slides to train on; tags the checkpoint and the '
                          'field cache')
    ap.add_argument('--out', default=None,
                     help='override cfg.OUT: root holding the *_adata.h5ad and tile caches.')
    ap.add_argument('--ckpt', default=None, help='override cfg.CKPT (checkpoint dir)')
    ap.add_argument('--train-overlap-stride', type=int, default=2,
                     help='stride, in 256px windows, of the train-side sliding 4x4 field '
                          'window: 4 = disjoint fields, 2 = 50%% field overlap, 1 = every position')
    ap.add_argument('--test-overlap-stride', type=int, default=1,
                     help='same stride for the val-side window, over val-assigned windows '
                          'only; independent of --train-overlap-stride')
    ap.add_argument('--readout-decay', type=float, default=0.0,
                     help='per-head -gamma*||offset||^2 penalty on the GridReadout attention '
                          'logits, initial gamma. 0 = the original unconstrained readout, which '
                          'measures as near-global. gamma is learnable, so this sets the prior, '
                          'not a hard limit. The reported results use 50.')
    args = ap.parse_args()

    rank, local_rank, world_size, device, is_ddp = ddp_setup()
    torch.backends.cudnn.benchmark = True

    cfg = make_cfg_1024(seed=args.seed)
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch:
        cfg.BATCH = args.batch
    if args.samples:
        cfg.SAMPLES = args.samples.split(',')
    if args.out:
        cfg.OUT = args.out
    if args.ckpt:
        cfg.CKPT = args.ckpt
    cfg.TEST_SAMPLE = args.test_sample
    cfg.READOUT_DECAY = args.readout_decay
    set_seed(cfg.SEED)

    GENES, G2I = load_vocab(cfg)
    cfg.K_MAX_GENES = 87

    train256, _, _, umu, usd = load_or_build_256(cfg, G2I, rank, is_ddp, barrier)
    gene_mu, gene_sd = train256.gene_mu, train256.gene_sd
    del train256

    train_ds, val_ds, va1024 = load_or_build_1024(cfg, G2I, gene_mu, gene_sd, rank, is_ddp, barrier,
                                                   train_overlap_stride=args.train_overlap_stride,
                                                   test_overlap_stride=args.test_overlap_stride)
    if rank == 0:
        held_out = f', held out: {cfg.TEST_SAMPLE}' if cfg.TEST_SAMPLE else ''
        print(f'cfg ready{held_out} | samples={cfg.SAMPLES} | split=random | '
              f'train_overlap_stride={args.train_overlap_stride} | '
              f'test_overlap_stride={args.test_overlap_stride} | world_size={world_size} | '
              f'train {len(train_ds)} val {len(val_ds)}')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                        seed=cfg.SEED, drop_last=True) if is_ddp else None
    tdl = DataLoader(train_ds, cfg.BATCH, shuffle=(train_sampler is None), sampler=train_sampler,
                      num_workers=args.workers, pin_memory=True, drop_last=True,
                      persistent_workers=args.workers > 0)
    vdl = DataLoader(val_ds, cfg.BATCH, shuffle=False, num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0)

    Model = GeneTokenCells1024
    model = Model(cfg).to(device)
    mu, sd = umu.to(device), usd.to(device)
    model.dec.uni_mean.copy_(mu)
    model.dec.uni_std.copy_(sd)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if is_ddp else model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD, betas=(0.9, 0.95))
    tot = cfg.EPOCHS * len(tdl)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / cfg.WARMUP if s < cfg.WARMUP
        else 0.5 * (1 + math.cos(math.pi * (s - cfg.WARMUP) / max(1, tot - cfg.WARMUP))))
    scaler = torch.amp.GradScaler('cuda')
    NORM_W = 0.1
    best = -9.0
    loo_tag = f'_loo_{cfg.TEST_SAMPLE}' if cfg.TEST_SAMPLE else ''
    samp_tag = samples_tag(cfg)
    ov_tag = '' if args.train_overlap_stride == 2 else f'_ov{args.train_overlap_stride}'
    test_tag = '' if args.test_overlap_stride == 4 else f'_test{args.test_overlap_stride}'
    ck_path = Path(cfg.CKPT) / f'pooled_{cfg.MODEL}_g{cfg.N_GENE_LAYERS}_s{cfg.SEED}{loo_tag}{samp_tag}{ov_tag}{test_tag}.pt'

    Rv = F.normalize(torch.tensor(np.concatenate([s[2][s[3]] for s in va1024], 0)), dim=-1).to(device) \
        if rank == 0 else None

    for ep in range(cfg.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        loss_sum, loss_n = 0.0, 0
        it = tqdm(tdl, desc=f'gt1024 s{cfg.SEED} ep{ep:03d}', disable=(rank != 0))
        for b in it:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            m = b['tmask']  # (B,16)
            with torch.amp.autocast('cuda'):
                p = model(b)
                t = b['tgt']
                w = m.float().unsqueeze(-1)
                denom = m.sum().clamp(min=1)
                mse = ((((p - mu) / sd - (t - mu) / sd) ** 2) * w).sum() / (denom * p.shape[-1])
                cos = ((1 - F.cosine_similarity(p, t, dim=-1)) * m).sum() / denom
                nrm = (((p.norm(dim=-1) - t.norm(dim=-1)).abs()) * m).sum() / denom
                loss = mse + 0.25 * cos + NORM_W * nrm
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            scaler.step(opt)
            scaler.update()
            sch.step()
            loss_sum += loss.item()
            loss_n += 1
        train_loss = reduce_sum_count(loss_sum, loss_n, device, is_ddp)

        if rank == 0:
            model.eval()
            cs, P, pn, rn = [], [], [], []
            with torch.no_grad():
                for b in vdl:
                    b = {k: v.to(device) for k, v in b.items()}
                    p = raw_model(b)
                    t = b['tgt']
                    m = b['tmask']
                    cs.append(F.cosine_similarity(p, t, dim=-1)[m].cpu())
                    P.append(F.normalize(p[m], dim=-1))
                    pn.append(p.norm(dim=-1)[m].cpu())
                    rn.append(t.norm(dim=-1)[m].cpu())
            own = torch.cat(cs).mean().item()
            nn_ = (torch.cat(P) @ Rv.T).max(1).values.median().item()
            print(f'  own_cos={own:.4f}  pred_nn={nn_:.3f}  |p|={torch.cat(pn).mean():.2f} '
                  f'|r|={torch.cat(rn).mean():.2f}  loss={train_loss:.4f}')
            if own > best:
                best = own
                torch.save({'model': raw_model.state_dict(),
                            'cfg': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                            'own_cos': own, 'ep': ep, 'seed': cfg.SEED}, ck_path)
                print('   -> saved')
        barrier(is_ddp)

    if rank == 0:
        print(f'[gt1024 s{cfg.SEED}] best own_cos={best:.4f}  ckpt={ck_path}')
    ddp_cleanup(is_ddp)


if __name__ == '__main__':
    main()

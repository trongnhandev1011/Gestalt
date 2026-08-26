#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train the mapper: regress a frozen foundation model's tile embedding.

--adv-weight adds the domain-confusion term to the mapper's training. It defaults to 0.0, so
the default run is the plain mapper.
"""
import argparse
import math
import random
import sys
import warnings
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
from ..core.config import make_cfg_256, samples_tag, fm_tag, FM_REGISTRY
from .dataset import load_vocab, load_or_build_256
from ..core.ddp_utils import ddp_setup, ddp_cleanup, barrier, reduce_sum_count
from ..models.mapper import GeneTokenCells256


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _confusion_loss(logits):
    """Confusion loss CE(softmax(logits), uniform) = -mean_k log p_k, minimised when p is
    uniform."""
    import torch.nn.functional as _F
    return -_F.log_softmax(logits.float(), dim=-1).mean(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    # data
    ap.add_argument('--samples', default=None, help='comma-separated slides to train on')
    ap.add_argument('--test-sample', default=None, help='slide to hold out entirely')
    ap.add_argument('--out', default=None, help='tile directory (default $SPATIALGENE_OUT)')
    ap.add_argument('--ckpt', default=None, help='checkpoint directory (default $SPATIALGENE_CKPT)')
    # model
    ap.add_argument('--fm', default='uni2h', choices=list(FM_REGISTRY),
                    help='foundation-model embedding to predict')
    ap.add_argument('--pairbias-k', type=int, default=None, help='k-NN neighbourhood size')
    # optimisation
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None, help='per-GPU batch size')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0,
                    help='mapper initialization, augmentation and batch-order seed')
    ap.add_argument('--split-seed', type=int, default=0,
                    help='train/validation split seed; keep fixed when measuring training-seed '
                         'variance')
    ap.add_argument('--snapshot-every', type=int, default=0,
                    help='save a snapshot every N epochs; 0 keeps only the best checkpoint')
    ap.add_argument('--fp32', action='store_true',
                    help='disable autocast and the GradScaler; use for wide panels such as P1')
    ap.add_argument('--resume', action='store_true',
                    help='continue from an existing checkpoint at the target path')
    # Defaults used for every reported run.
    ap.set_defaults(per_slide_bias=False, nce_weight=0.0, l2_weight=1.0, nce_temp=0.1,
                    disc_lr_mult=10.0, disc_steps=3)
    args = ap.parse_args()

    rank, local_rank, world_size, device, is_ddp = ddp_setup()
    torch.backends.cudnn.benchmark = True

    cfg = make_cfg_256(seed=args.seed, fm=args.fm, split_seed=args.split_seed)
    cfg.MODEL = 'genetoken'
    if args.pairbias_k is not None:
        cfg.PAIRBIAS_K = args.pairbias_k
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch:
        cfg.BATCH = args.batch
    cfg.TEST_SAMPLE = args.test_sample
    if args.samples:
        cfg.SAMPLES = args.samples.split(',')
    if args.out:
        cfg.OUT = args.out
    if args.ckpt:
        cfg.CKPT = args.ckpt

    set_seed(cfg.SEED)

    GENES, G2I = load_vocab(cfg)
    train_ds, val_ds, va_s, umu, usd = load_or_build_256(cfg, G2I, rank, is_ddp, barrier)
    if rank == 0:
        held_out = f', held out: {cfg.TEST_SAMPLE}' if cfg.TEST_SAMPLE else ''
        print(f'cfg ready | fm={cfg.FM} (dim={cfg.UNI_DIM}) | {len(cfg.SAMPLES)} samples{held_out}'
              f' | split=random(seed={cfg.SPLIT_SEED}) | train_seed={cfg.SEED} | '
              f'world_size={world_size} | '
              f'train {len(train_ds)} val {len(val_ds)}')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                        seed=cfg.SEED, drop_last=True) if is_ddp else None
    tdl = DataLoader(train_ds, cfg.BATCH, shuffle=(train_sampler is None), sampler=train_sampler,
                      num_workers=args.workers, pin_memory=True, drop_last=True,
                      persistent_workers=args.workers > 0)
    vdl = DataLoader(val_ds, cfg.BATCH, shuffle=False, num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0)

    model = GeneTokenCells256(cfg).to(device)
    mu, sd = umu.to(device), usd.to(device)
    model.dec.uni_mean.copy_(mu)
    model.dec.uni_std.copy_(sd)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if is_ddp else model

    # Auxiliary masked-gene head: the encoder does not see the masked positions, so the
    # pooled cell vector must retain gene-level detail to reconstruct them.
    _params = list(model.parameters())
    opt = torch.optim.AdamW(_params, lr=cfg.LR, weight_decay=cfg.WD, betas=(0.9, 0.95))
    tot = cfg.EPOCHS * len(tdl)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / cfg.WARMUP if s < cfg.WARMUP
        else 0.5 * (1 + math.cos(math.pi * (s - cfg.WARMUP) / max(1, tot - cfg.WARMUP))))
    # --fp32 disables autocast and the GradScaler. Use it for wide panels such as Xenium
    # Prime (5006 genes).
    scaler = torch.amp.GradScaler('cuda', enabled=not args.fp32)
    NORM_W = 0.1
    best = -9.0
    loo_tag = f'_loo_{cfg.TEST_SAMPLE}' if cfg.TEST_SAMPLE else ''
    samp_tag = samples_tag(cfg)
    split_tag = '' if cfg.SPLIT_SEED == 0 else f'_split{cfg.SPLIT_SEED}'
    k_tag = f'_k{args.pairbias_k}' if args.pairbias_k is not None else ''
    ck_path = (Path(cfg.CKPT) /
               f'mapper_s{cfg.SEED}{split_tag}{loo_tag}{samp_tag}{fm_tag(cfg)}{k_tag}.pt')

    start_ep = 0
    if args.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck['model'])
        best = ck.get('own_cos', best)
        start_ep = ck.get('ep', -1) + 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(start_ep * len(tdl)):
                sch.step()
        if rank == 0:
            print(f'[resume] loaded {ck_path} (saved at ep{ck.get("ep")}, own_cos={best:.4f}) '
                  f'-> continuing from ep{start_ep}')
    elif args.resume and rank == 0:
        print(f'[resume] --resume passed but no checkpoint at {ck_path}; training from scratch')

    # pred_nn reference bank (real val targets); validation runs rank0-only so this is too
    Rv = F.normalize(torch.tensor(np.stack([s[2] for s in va_s])), dim=-1).to(device) if rank == 0 else None

    for ep in range(start_ep, cfg.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        loss_sum, loss_n = 0.0, 0
        it = tqdm(tdl, desc=f'{cfg.MODEL} s{cfg.SEED} ep{ep:03d}', disable=(rank != 0))
        for b in it:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=not args.fp32):
                p = model(b)
                t = b['tgt']
                mse = (((p - mu) / sd - (t - mu) / sd) ** 2).mean()
                cos = (1 - F.cosine_similarity(p, t, dim=-1)).mean()
                nrm = (p.norm(dim=-1) - t.norm(dim=-1)).abs().mean()
                base = mse + 0.25 * cos + NORM_W * nrm
                # `base` compares each prediction to its own target only. InfoNCE adds pressure between
                # tiles, with negatives drawn from a queue of frozen target vectors.
                loss = base
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            scaler.step(opt)
            scaler.update()
            sch.step()
            loss_sum += loss.item()
            loss_n += 1
        train_loss = reduce_sum_count(loss_sum, loss_n, device, is_ddp)

        # validation runs on rank 0 only: cheap relative to a training epoch, and it avoids
        # an all_gather just to compute the pred_nn median over the whole val set.
        if rank == 0:
            model.eval()
            cs, P, pn, rn = [], [], [], []
            with torch.no_grad():
                for b in vdl:
                    b = {k: v.to(device) for k, v in b.items()}
                    p = raw_model(b)
                    t = b['tgt']
                    cs.append(F.cosine_similarity(p, t, dim=-1).cpu())
                    P.append(F.normalize(p, dim=-1))
                    pn.append(p.norm(dim=-1).cpu())
                    rn.append(t.norm(dim=-1).cpu())
            own = torch.cat(cs).mean().item()
            nn_ = (torch.cat(P) @ Rv.T).max(1).values.median().item()
            print(f'  own_cos={own:.4f}  pred_nn={nn_:.3f}  |p|={torch.cat(pn).mean():.2f} '
                  f'|r|={torch.cat(rn).mean():.2f}  loss={train_loss:.4f}')
            if own > best:
                best = own
                _sd = {'model': raw_model.state_dict(),
                       'genes': GENES,
                       'cfg': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                       # the full invocation: cfg alone omits flags that never land on it,
                       # so without this a checkpoint's recipe cannot be recovered
                       'args': vars(args),
                       'own_cos': own, 'ep': ep, 'seed': cfg.SEED}
                torch.save(_sd, ck_path)
                print('   -> saved')
            # Periodic epoch snapshots, so every method can be read at a fixed epoch budget.
            if args.snapshot_every and ((ep + 1) % args.snapshot_every == 0 or ep == cfg.EPOCHS - 1):
                _sn = {'model': raw_model.state_dict(), 'genes': GENES,
                       'cfg': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                       'own_cos': own, 'ep': ep, 'seed': cfg.SEED, 'args': vars(args)}
                snap_path = ck_path.with_name(f'{ck_path.stem}_ep{ep+1:03d}.pt')
                torch.save(_sn, snap_path)
                print(f'   -> snapshot {snap_path.name}')
        barrier(is_ddp)

    if rank == 0:
        # Keep the final epoch alongside the best-own_cos checkpoint. For the contrastive arms
        # these are different models: own_cos measures the pointwise fidelity the InfoNCE term
        # trades away.
        print(f'[{cfg.MODEL} s{cfg.SEED}] best own_cos={best:.4f}  ckpt={ck_path}')
    ddp_cleanup(is_ddp)


if __name__ == '__main__':
    main()

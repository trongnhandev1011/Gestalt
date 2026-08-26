#!/usr/bin/env python
# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Decoder LoRA for a MAPPER conditioning signal (foundation-model embedding space).
"""
import argparse
import random
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch


def _no_barrier(*a, **k):
    pass

class GTTileResolver:
    """Maps a training record's target embedding vector back to its real (3,H,W) [0,1] image.

    sample_name=None indexes every sample in cfg.SAMPLES except cfg.TEST_SAMPLE; pass an
    explicit sample_name to restrict to one. The index is built on the first .get() call, so
    each DataLoader worker builds only the index it uses.
    """

    def __init__(self, cfg, sample_name):
        self.cfg = cfg
        self.samples = [sample_name] if sample_name else [
            s for s in cfg.SAMPLES if s != getattr(cfg, 'TEST_SAMPLE', None)
        ]
        self._built = False
        self._uni2loc = {}   # tgt.tobytes() -> (sample, row)
        self._img = {}       # (sample, row) -> cached (3,H,W) float32 [0,1] tensor
        self._h5 = {}        # sample -> open h5py.File for {sample}_tiles256_p5.h5 (images)

    def _build(self):
        fm = getattr(self.cfg, 'FM', 'uni2h')
        suffix, key = FM_REGISTRY[fm]['suffix'], FM_REGISTRY[fm]['h5_key']
        for s in self.samples:
            with h5py.File(f'{self.cfg.OUT}/{s}_tiles256_p5_{suffix}.h5', 'r') as f:
                emb = f[key][:]
            for row in range(emb.shape[0]):
                self._uni2loc[np.asarray(emb[row], np.float32).tobytes()] = (s, row)
            # held open for the resolver's lifetime: _load_img runs once per training step,
            # so one handle per sample replaces ~2800 file opens per epoch.
            self._h5[s] = h5py.File(f'{self.cfg.OUT}/{s}_tiles256_p5.h5', 'r')
        self._built = True

    def _load_img(self, sample, row):
        img = self._h5[sample]['images'][row]                          # (256,256,3) uint8
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t                                                       # (3,H,W) in [0,1]

    def get(self, tgt):
        if not self._built:
            self._build()
        loc = self._uni2loc.get(np.asarray(tgt, np.float32).tobytes())
        if loc is None:
            return None
        if loc not in self._img:
            self._img[loc] = self._load_img(*loc)
        return self._img[loc]


def _set_lora_seed(s):
    import random as _r
    import numpy as _np
    _r.seed(s); _np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from ..core.config import samples_tag

from ..core.config import make_cfg_256, fm_tag, FM_REGISTRY
from .pixcell import swap_caption_projection
from .dataset import load_vocab, load_or_build_256
from ..core.ddp_utils import ddp_setup, ddp_cleanup, barrier


def _worker_init(worker_id):
    res = torch.utils.data.get_worker_info().dataset.res
    res._img = {}
    res._built = False
    res._uni2loc = {}


def collect_train_records(cfg, G2I):
    train_ds, val_ds, va_s, umu, usd = load_or_build_256(
        cfg, G2I, rank=0, is_ddp=False, barrier_fn=_no_barrier)
    if cfg.TEST_SAMPLE:
        records = list(train_ds.s) + list(va_s)
    else:
        records = list(train_ds.s)
    return records


# --------------------------------------------------------------------------- train

class UniTileDS(Dataset):
    """(embedding, image) pairs, resolving each record's target embedding back to its tile."""

    def __init__(self, records, resolver):
        self.recs = records
        self.res = resolver
        print(f'[data] {len(records)} train-fold (embedding, image) pairs')

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, k):
        # A record can fail to resolve to a tile. A map-style Dataset cannot skip an index, so
        # retry with a fresh random one.
        for _ in range(10):
            r = self.recs[k]
            img = self.res.get(r[2])
            if img is not None:
                break
            k = random.randrange(len(self.recs))
        uni = torch.from_numpy(np.asarray(r[2], np.float32))
        img = F.interpolate(img[None], size=(256, 256), mode='bilinear', align_corners=False)[0]
        img = img * 2 - 1
        return uni, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-sample', default=None)
    ap.add_argument('--steps', type=int, default=8000, help='Total target steps (e.g., 12000 to train 4k more past step 8000)')
    ap.add_argument('--resume-from', type=str, default=None, help='Path to LoRA checkpoint to resume from (e.g., .../step_8000)')
    ap.add_argument('--batch', type=int, default=4, help='per-GPU batch')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=32, help='LoRA rank')
    ap.add_argument('--alpha', type=int, default=64, help='LoRA alpha')
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save-every', type=int, default=None,
                    help='checkpoint interval in steps; the default saves only the final step')
    ap.add_argument('--samples', default=None,
                    help='slide subset to fine-tune on, e.g. C1 or C1,C2')
    ap.add_argument('--out', default=None, help='override cfg.OUT')
    ap.add_argument('--train-seed', type=int, default=0,
                    help='seeds adapter initialization, batch order and CUDA')
    ap.add_argument('--split-seed', type=int, default=0,
                    help='train/validation split seed; must match mapper training')
    ap.add_argument('--fm', default='uni2h', choices=list(FM_REGISTRY),
                    help='foundation model to adapt PixCell to; its embedding cache must '
                         'already exist for every slide in --samples')
    args = ap.parse_args()
    _set_lora_seed(args.train_seed)
    if args.save_every is None:
        args.save_every = args.steps

    rank, local_rank, world_size, dev, is_ddp = ddp_setup()

    cfg = make_cfg_256(fm=args.fm, split_seed=args.split_seed)
    cfg.TEST_SAMPLE = args.test_sample
    if args.samples:
        cfg.SAMPLES = args.samples.split(',')
    if args.out:
        cfg.OUT = args.out

    GENES, G2I = load_vocab(cfg)

    records = collect_train_records(cfg, G2I)
    # sample_name=None indexes cfg.SAMPLES minus cfg.TEST_SAMPLE.
    resolver = GTTileResolver(cfg, None)
    ds = UniTileDS(records, resolver)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.train_seed, drop_last=True) if is_ddp else None
    dl = DataLoader(ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0,
                    worker_init_fn=_worker_init if args.workers > 0 else None)

    # ---- load PixCell-256 (base DiT) ----
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large',
                                        subfolder='vae', torch_dtype=torch.float32)
    pipe = DiffusionPipeline.from_pretrained(
        'StonyBrook-CVLab/PixCell-256', vae=vae,
        custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
        torch_dtype=torch.float32, trust_remote_code=True).to(dev)
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, 'shift_factor', 0.0)

    swap_caption_projection(transformer, args.fm)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # ---- LoRA Initialization / Resumption ----
    # modules_to_save keeps caption_projection fully trainable and stores it in the adapter
    # checkpoint, which is required when --fm is not uni2h.
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias='none',
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2'],
        modules_to_save=None if args.fm == 'uni2h' else ['caption_projection'],
    )
    transformer = get_peft_model(transformer, lora)

    if args.resume_from:
        if rank == 0:
            print(f'[lora] manually loading weights from {args.resume_from}')
        
        from safetensors.torch import load_file
        
        safe_path = Path(args.resume_from) / "adapter_model.safetensors"
        bin_path = Path(args.resume_from) / "adapter_model.bin"
        
        if safe_path.exists():
            state_dict = load_file(safe_path)
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Could not find adapter weights in {args.resume_from}")
            
        # strict=False is required because the state_dict only contains LoRA weights, 
        # not the frozen base model weights.
        transformer.load_state_dict(state_dict, strict=False)
        
    transformer.to(dev)
    transformer.train()
    
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if rank == 0:
        print(f'[lora] trainable params: {n_train/1e6:.2f}M  (rank {args.rank}, alpha {args.alpha})')

    if is_ddp:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)
    raw_model = transformer.module if is_ddp else transformer 

    opt = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)

    # ---- Optimizer / Scheduler State Resumption ----
    step = 0
    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu')
            opt.load_state_dict(state['opt'])
            sch.load_state_dict(state['sch'])
            step = state['step']
            if rank == 0:
                print(f"[resume] Fully restored optimizer & scheduler state from step {step}")
        else:
            # Fallback for a checkpoint that saved weights only
            try:
                step = int(Path(args.resume_from).name.split('_')[-1])
            except ValueError:
                step = 0
            if rank == 0:
                print("[resume] No training_state.pt found. Fresh optimizer initialized.")
                print(f"[resume] Fast-forwarding learning rate scheduler to step {step}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(step):
                    sch.step()

    if step >= args.steps:
        if rank == 0:
            print(f"Current step ({step}) is already >= target steps ({args.steps}). Exiting.")
        return
    loo_tag = f'_loo_{args.test_sample}' if args.test_sample else ''
    train_seed_tag = '' if args.train_seed == 0 else f'_s{args.train_seed}'
    split_tag = '' if args.split_seed == 0 else f'_split{args.split_seed}'
    out_dir = (Path(cfg.CKPT) /
               f'pixcell_lora_r{args.rank}{train_seed_tag}{split_tag}{loo_tag}'
               f'{samples_tag(cfg)}{fm_tag(cfg)}')
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier(is_ddp)

    ep = 0
    pbar = tqdm(total=args.steps, initial=step, desc='pixcell-lora', disable=(rank != 0))
    losses = []
    
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(ep)
        ep += 1
        for uni, img in dl:
            uni = uni.to(dev).float()
            img = img.to(dev).float()
            B = img.shape[0]
            if uni.dim() == 2:
                uni = uni.unsqueeze(1)                                

            with torch.no_grad():
                lat = vae.encode(img).latent_dist.mean               
                lat = (lat - vae_shift) * vae_scale                  

            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device='cpu')
            atbar = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(dev)
            eps = torch.randn_like(lat)
            noisy = atbar.sqrt() * lat + (1 - atbar).sqrt() * eps
            timesteps = t.to(dev)

            added = {'resolution': torch.tensor([[256, 256]], device=dev).repeat(B, 1),
                     'aspect_ratio': torch.tensor([[1.0]], device=dev).repeat(B, 1)}

            eps_pred = transformer(noisy, encoder_hidden_states=uni, timestep=timesteps,
                                   added_cond_kwargs=added, return_dict=False)[0]
            loss = F.mse_loss(eps_pred[:, :16], eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad], 1.0)
            opt.step()
            sch.step()

            losses.append(loss.item())
            step += 1
            if rank == 0:
                pbar.update(1)
                if step % 50 == 0:
                    pbar.set_postfix({'loss': f'{np.mean(losses[-50:]):.4f}'})
                if step % args.save_every == 0 or step == args.steps:
                    p = out_dir / f'step_{step}'
                    raw_model.save_pretrained(p)         # Save LoRA adapters
                    torch.save({                         # Save state for future resumes
                        'step': step,
                        'opt': opt.state_dict(),
                        'sch': sch.state_dict(),
                        'args': vars(args),
                        'train_seed': args.train_seed,
                        'split_seed': args.split_seed,
                    }, p / "training_state.pt")
                    print(f'\n[save] {p}  loss(ema50)={np.mean(losses[-50:]):.4f}')
            if step >= args.steps:
                break
    pbar.close()
    if rank == 0:
        print(f'\ndone. adapters in {out_dir}. Sweep step_* checkpoints on held-out '
              f'oracle-vs-gt (benchmark with the SAME {loo_tag}) '
              f'and keep the peak.')
    ddp_cleanup(is_ddp)

if __name__ == '__main__':
    main()

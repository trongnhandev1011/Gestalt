# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""torchrun-compatible DDP helpers.

Every training script falls back to a single process on one GPU, or on CPU, when it is not
launched under torchrun.
"""
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def ddp_setup():
    """Returns (rank, local_rank, world_size, device, is_ddp)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', device_id=torch.device('cuda', local_rank))
        device = torch.device(f'cuda:{local_rank}')
        is_ddp = world_size > 1
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_ddp = False
    return rank, local_rank, world_size, device, is_ddp


def ddp_cleanup(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def barrier(is_ddp):
    if is_ddp:
        dist.barrier()


def reduce_sum_count(sum_, count_, device, is_ddp):
    """All-reduce a (sum, count) pair across ranks and return the global mean."""
    if not is_ddp:
        return sum_ / max(count_, 1)
    t = torch.tensor([sum_, count_], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s, c = t.tolist()
    return s / max(c, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def collate(items):
    keys = items[0].keys()
    return {k: torch.stack([it[k] for it in items]) for k in keys}

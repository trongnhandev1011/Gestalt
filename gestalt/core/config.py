# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for Gestalt.

Holds the foundation-model registry, naming the image encoder that produced the embedding a
mapper is trained to predict, and the model and training hyperparameters.

The registry lists the four image encoders a mapper can be trained to predict.
"""
import os

# --------------------------------------------------------------------------------------
# Foundation models. 'suffix' names the cache file {sample}_tiles256_p5_{suffix}.h5 and
# 'h5_key' the dataset inside it.
FM_REGISTRY = {
    'uni2h':    dict(dim=1536, suffix='uni',      h5_key='uni', hf_id='hf-hub:MahmoodLab/UNI2-h'),
    'uni_v1':   dict(dim=1024, suffix='univ1',    h5_key='emb', hf_id='hf-hub:MahmoodLab/UNI'),
    'conch':    dict(dim=512,  suffix='conch',    h5_key='emb', hf_id='hf_hub:MahmoodLab/CONCH'),
    'virchow2': dict(dim=2560, suffix='virchow2', h5_key='emb', hf_id='hf-hub:paige-ai/Virchow2'),
}

_HEST13 = ['NCBI783', 'NCBI784', 'NCBI785', 'TENX105', 'TENX106', 'TENX115', 'TENX117',
           'TENX120', 'TENX138', 'TENX140', 'TENX141', 'TENX142', 'TENX148']

def fm_spec(name):
    """Spec for a foundation model in FM_REGISTRY."""
    if name not in FM_REGISTRY:
        raise ValueError(f'unknown foundation model {name!r}; choose from {sorted(FM_REGISTRY)}')
    return FM_REGISTRY[name]


def fm_tag(cfg):
    """Deterministic suffix identifying the target FM, for cache/checkpoint filenames.
    Empty for uni2h, else '_<name>'."""
    fm = getattr(cfg, 'FM', 'uni2h')
    return '' if fm == 'uni2h' else f'_{fm}'


class Cfg:
    HEST_ROOT = os.path.expanduser(os.environ.get('SPATIALGENE_HEST_ROOT', '~/SpatialGene/hest_data'))
    OUT = os.path.expanduser(os.environ.get('SPATIALGENE_OUT', '~/SpatialGene/tiles'))
    SAMPLES = os.environ.get('SPATIALGENE_SAMPLES',
        'NCBI783,NCBI784,NCBI785,TENX105,TENX106,TENX115,TENX117,'
        'TENX120,TENX138,TENX140,TENX141,TENX142,TENX148').split(',')

    GENE_LAYER = 'norm'
    ZSCORE_GENES = True
    CAP_TILES = None
    FM = 'uni2h'          # target foundation model; see FM_REGISTRY. Sets UNI_DIM below.
    SUB, UNI_DIM = 256, FM_REGISTRY['uni2h']['dim']
    N_MAX_CELLS = None
    GENE_DIM = None
    GENE_VOCAB = None

    MODEL = 'genetoken'
    HIDDEN, N_HEADS, N_BLOCKS = 384, 8, 4
    K_NN = 16
    # settled default: the k-NN neighbourhood the cross-cell attention reads, at TRAIN
    # and at INFERENCE. It changes no tensor shape, so a train/eval mismatch is silent.
    PAIRBIAS_K = 8

    # BATCH is PER-GPU. Under DDP, effective batch = BATCH * world_size.
    # 8 fits ~22GB for the 256 model. Raise on A100 40/80GB for more throughput per GPU.
    BATCH, LR, WD, EPOCHS, WARMUP, CLIP = 8, 2e-4, 1e-4, 120, 200, 1.0
    AUG_DROP, AUG_MINKEEP, AUG_JITTER = 0.2, 16, 0.01
    CV_HOLDOUT_FRAC = 0.2
    SEED = 0        # mapper initialization, augmentation and batch order
    SPLIT_SEED = 0  # train/validation membership; fixed across reported training seeds
    N_GENE_LAYERS = 1  # gene self-attention depth in GeneTokenEncoder; matches g1 checkpoints
    TEST_SAMPLE = None  # HEST sample name to hold out entirely (leave-one-slide-out); None = pool all 13

    def __init__(self):
        # checkpoints, benchmark JSONs, LoRA adapters and logs
        self.CKPT = os.path.expanduser(os.environ.get(
            'SPATIALGENE_CKPT', '~/SpatialGene/checkpoints'))
        os.makedirs(self.CKPT, exist_ok=True)


def make_cfg_256(seed=0, fm='uni2h', split_seed=0):
    cfg = Cfg()
    cfg.MODEL = 'genetoken'
    cfg.SEED = seed
    cfg.SPLIT_SEED = split_seed
    cfg.FM = fm
    cfg.UNI_DIM = fm_spec(fm)['dim']
    return cfg


def make_cfg_1024(seed=0, fm='uni2h'):
    cfg = Cfg()
    cfg.MODEL = 'genetoken1024'
    cfg.SEED = seed
    cfg.FM = fm
    cfg.UNI_DIM = fm_spec(fm)['dim']
    cfg.BATCH = 2  # per-GPU; ~16x cells/field vs 256, drop to 1 (or lower K_MAX_GENES) if OOM
    return cfg


def samples_tag(cfg):
    """Suffix identifying the training slide set, shared by the checkpoint and cache paths.

    Empty for the default 13-slide HEST pool, otherwise the sorted slide names joined by '-'.
    """
    if set(cfg.SAMPLES) == set(_HEST13):
        return ''
    return '_' + '-'.join(sorted(cfg.SAMPLES))

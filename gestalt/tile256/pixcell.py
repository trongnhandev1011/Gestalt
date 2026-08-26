# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Decoder-side helpers shared by the LoRA trainers and the benchmark."""
import torch

from ..core.config import fm_spec


def swap_caption_projection(transformer, fm):
    """Retarget PixCell's caption projection at a conditioning embedding of a different width.

    PixCell's cross-attention runs at a fixed internal width, and caption_projection
    (linear_1 -> GELU -> linear_2) maps the conditioning embedding into it. Only linear_1 is
    replaced, in place; act_1 and linear_2 keep their pretrained weights.

    Args:
        transformer: the PixCell transformer to modify in place.
        fm (str): the foundation model whose embedding space conditions the decoder.
    """
    if fm == 'uni2h':
        return  # already the pretrained caption_channels
    proj = transformer.caption_projection
    new_dim = fm_spec(fm)['dim']
    proj.linear_1 = torch.nn.Linear(new_dim, proj.linear_1.out_features, bias=True)
    proj.uncond_embedding = torch.nn.Parameter(
        torch.randn(proj.uncond_embedding.shape[0], new_dim) / new_dim ** 0.5)
    transformer.config.caption_channels = new_dim

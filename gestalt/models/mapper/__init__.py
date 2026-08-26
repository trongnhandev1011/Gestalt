# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Objective 1: predict a frozen foundation model's tile embedding."""
from .model import GeneTokenCells256, GeneTokenCells1024

__all__ = ['GeneTokenCells256', 'GeneTokenCells1024']

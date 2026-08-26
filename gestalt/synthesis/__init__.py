# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Controllable tissue synthesis: an explicit statistical model of gene expression, used to
build synthetic (position, expression) fields that the mapper and decoder then render.

See sec/5_synthesis.tex and sec/s17_synthesis.tex in the paper for the described method.
This is a from-scratch reconstruction of that method against its written description, not a
copy of the exact run that produced the reported figures (the original scripts were prototyped
in a session scratchpad that no longer exists) -- see README.md for what that implies.
"""
from .expression_model import TileTypeExpressionModel
from .placement import hardcore_positions

__all__ = ['TileTypeExpressionModel', 'hardcore_positions']

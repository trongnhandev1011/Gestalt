# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared backbone, readouts and configuration.

Every conditioning objective in `gestalt.models` builds on the components here.
"""
from .backbone import GATLayer, GeneSelfAttn, GeneTokenEncoder, RelPosBias, build_knn
from .config import (FM_REGISTRY, Cfg, fm_spec, fm_tag, make_cfg_256,
                     make_cfg_1024, samples_tag)
from .readout import GridReadout, QueryReadout

__all__ = ['GATLayer', 'GeneSelfAttn', 'GeneTokenEncoder', 'RelPosBias', 'build_knn',
           'FM_REGISTRY', 'Cfg', 'fm_spec', 'fm_tag', 'make_cfg_256', 'make_cfg_1024',
           'samples_tag', 'GridReadout', 'QueryReadout']

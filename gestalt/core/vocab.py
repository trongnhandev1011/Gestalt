# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gene vocabulary, computed from the training slides.

The vocabulary is the union of gene panels over training slides only, and is therefore
determined by (cfg.OUT, cfg.SAMPLES, cfg.TEST_SAMPLE): each leave-one-out fold has its own.
Panels differ across slides, so every checkpoint stores the gene list it was trained with and
the benchmark reads that list back from the checkpoint. Genes absent from every training panel
are dropped at evaluation time.
"""
import anndata as ad


def train_samples(cfg):
    test = getattr(cfg, 'TEST_SAMPLE', None)
    tr = [s for s in cfg.SAMPLES if s != test]
    if not tr:
        raise ValueError(f'TEST_SAMPLE={test} removes every slide from SAMPLES={cfg.SAMPLES}')
    return tr


def _panel(cfg, s):
    A = ad.read_h5ad(f'{cfg.OUT}/{s}_adata.h5ad', backed='r')
    p = {str(g) for g in A.var_names}
    A.file.close()
    return p


def build_vocab(cfg, verbose=True):
    tr = train_samples(cfg)
    panels = {s: _panel(cfg, s) for s in tr}
    GENES = sorted(set().union(*panels.values()))
    G2I = {g: i for i, g in enumerate(GENES)}

    if verbose:
        test = getattr(cfg, 'TEST_SAMPLE', None)
        print(f'vocab: {len(GENES)} genes | train {",".join(tr)}'
              + (f' | held out {test}' if test else ''))
        for s in tr:
            if len(panels[s]) < len(GENES):
                print(f'  {s}: {len(panels[s])}/{len(GENES)}')
        if test:
            tp = _panel(cfg, test)
            print(f'  {test}: {len(tp)} genes, {len(tp - set(GENES))} unseen -> dropped at eval')
    return GENES, G2I
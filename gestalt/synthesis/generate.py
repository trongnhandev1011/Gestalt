# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate synthetic tissue fields and render them through the (already-trained, frozen)
mapper and decoder. Three modes, matching the paper's three synthesis figures:

  celltype     one synthetic field per requested cell type (fig:synth-celltype)
  density      a real tile's own state, cell count ramped up from its real count
               (fig:synth-density, sec:supp-synthesis)
  composition  fixed positions and count, the type mix swept between two types
               (fig:synth-composition, sec:supp-synthesis)

This module fits everything at inference time: PCA + Gaussian mixture + covariogram +
gene-count regression are all classical statistical fits (seconds on CPU, not a training run),
over real cells already extracted by gestalt.data. The mapper and decoder are loaded frozen,
exactly as in gestalt.tile256.benchmark.

Cell types are supplied by the caller as marker-gene sets (--markers, a JSON file
{"type_name": ["GENE1", "GENE2", ...]}) -- this repo does not ship a fixed marker panel, since
the panel is dataset- and tissue-specific. A cell is assigned the type whose mean marker
expression (in standardized units) is highest, if that score clears --marker-min-z.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from ..core.config import make_cfg_256
from ..models.mapper import GeneTokenCells256
from ..tile256.dataset import load_vocab, sample_sparse
from ..tile256.pixcell import swap_caption_projection
from .expression_model import TileTypeExpressionModel


def _load_real_cells(cfg, G2I, samples):
    """Pool every cell of every requested sample into dense, standardized (n_cells, n_genes)
    expression, per-tile grouping, and positions -- the same three things
    TileTypeExpressionModel.fit expects, built directly from the *_adata.h5ad / *_meta.json
    extraction gestalt.data.tiles/preprocess_hest produce, independent of the 256px train/val
    cache (that cache pads/truncates per the mapper's own tensor budget, which this fit should
    not inherit).
    """
    cell_expr_by_tile, xy_by_tile, n_detected_by_tile = [], [], []
    for s in samples:
        A = ad.read_h5ad(f'{cfg.OUT}/{s}_adata.h5ad')
        Xs, cols = sample_sparse(A, cfg, G2I)
        meta = json.load(open(f'{cfg.OUT}/{s}_meta.json'))
        for d in meta:
            ci = np.asarray(d['cell_idx'], np.int64)
            if len(ci) < 2:
                continue
            xy = (np.asarray(d['local_xy'], np.float32) / 256.0) * 2 - 1
            dense = np.zeros((len(ci), cfg.GENE_VOCAB), np.float32)
            n_det = np.zeros(len(ci), np.int64)
            for row, c in enumerate(ci):
                r = Xs.getrow(c)
                gids, val = cols[r.indices], r.data.astype(np.float32)
                dense[row, gids] = val
                n_det[row] = len(gids)
            cell_expr_by_tile.append(dense)
            xy_by_tile.append(xy.astype(np.float32))
            n_detected_by_tile.append(n_det)
    return cell_expr_by_tile, xy_by_tile, n_detected_by_tile


def _standardize(cell_expr_by_tile, gene_mu, gene_sd):
    """(value - mu) / sd, only where a gene was actually detected (zeros elsewhere stay zero
    rather than becoming -mu/sd), matching how gestalt.tile256.dataset standardizes cells."""
    out = []
    for dense in cell_expr_by_tile:
        z = np.zeros_like(dense)
        mask = dense != 0
        z[mask] = (dense[mask] - gene_mu[np.where(mask)[1]]) / gene_sd[np.where(mask)[1]]
        out.append(z)
    return out


def score_cell_types(cell_expr_by_tile, G2I, markers, min_z=0.5):
    """Assign each real tile's cells a type by highest mean marker expression.

    Args:
        cell_expr_by_tile (list[np.ndarray]): dense, UNstandardized (raw value) expression,
            one (n_cells, n_genes) array per tile (marker scoring uses raw values, so a gene
            simply absent from the panel scores zero rather than needing standardization stats).
        G2I (dict): gene name -> column index.
        markers (dict[str, list[str]]): type name -> marker gene list.
        min_z (float): a cell below this mean-marker-expression floor for every type is left
            unassigned (type -1) rather than forced into the closest match.

    Returns:
        list[np.ndarray]: one (n_cells,) int array per tile; -1 means unassigned.
        list[str]: type names, index i corresponds to label value i.
    """
    names = list(markers)
    cols = [[G2I[g] for g in markers[t] if g in G2I] for t in names]
    labels_by_tile = []
    for dense in cell_expr_by_tile:
        scores = np.stack([dense[:, c].mean(axis=1) if c else np.zeros(len(dense))
                            for c in cols], axis=1)  # (n_cells, n_types)
        best = scores.argmax(axis=1)
        best[scores.max(axis=1) < min_z] = -1
        labels_by_tile.append(best)
    return labels_by_tile, names


def fit_models(cell_expr_by_tile, xy_by_tile, n_detected_by_tile, labels_by_tile, names,
               gene_mu, gene_sd, n_pca=100, bic_range=(1, 20), seed=0):
    """One TileTypeExpressionModel per type with enough cells to fit."""
    models = {}
    for i, name in enumerate(names):
        expr_t, xy_t, nd_t = [], [], []
        for dense, xy, nd, lab in zip(cell_expr_by_tile, xy_by_tile, n_detected_by_tile,
                                       labels_by_tile):
            m = lab == i
            if m.sum() < 3:
                continue
            expr_t.append(dense[m])
            xy_t.append(xy[m])
            nd_t.append(nd[m])
        if len(expr_t) < 2:
            print(f'[skip] type {name!r}: fewer than 2 usable tiles')
            continue
        std_t = _standardize(expr_t, gene_mu, gene_sd)
        models[name] = TileTypeExpressionModel(n_pca=n_pca, bic_range=bic_range).fit(
            std_t, xy_t, nd_t, seed=seed)
        print(f'[fit] type {name!r}: {len(expr_t)} tiles, {sum(len(e) for e in expr_t)} cells, '
              f'gmm k={models[name].gmm.n_components}, range={models[name].range_:.3f}')
    return models


def field_to_batch(xy, gene_expr_std, n_detected, cfg, dev):
    """A synthesized (xy, standardized dense expression, detected-gene-count) field into the
    mapper's sparse per-cell tensor format: for each cell, keep its top-n_detected genes by
    |standardized value| as the detected set. Matches gestalt.tile256.dataset.TileDS exactly.
    """
    n = len(xy)
    assert n <= cfg.N_MAX_CELLS, f'field has {n} cells, cfg.N_MAX_CELLS={cfg.N_MAX_CELLS}'
    PAD = cfg.GENE_VOCAB
    GID = np.full((1, cfg.N_MAX_CELLS, cfg.K_MAX_GENES), PAD, np.int64)
    VAL = np.zeros((1, cfg.N_MAX_CELLS, cfg.K_MAX_GENES), np.float32)
    GM = np.zeros((1, cfg.N_MAX_CELLS, cfg.K_MAX_GENES), bool)
    for c in range(n):
        k = min(int(n_detected[c]), cfg.K_MAX_GENES)
        top = np.argsort(-np.abs(gene_expr_std[c]))[:k]
        GID[0, c, :k] = top
        VAL[0, c, :k] = gene_expr_std[c, top]
        GM[0, c, :k] = True
    XY = np.zeros((1, cfg.N_MAX_CELLS, 2), np.float32)
    CM = np.zeros((1, cfg.N_MAX_CELLS), bool)
    XY[0, :n] = xy
    CM[0, :n] = True
    return {'gene_ids': torch.from_numpy(GID).to(dev),
            'gene_vals': torch.from_numpy(VAL).to(dev),
            'gmask': torch.from_numpy(GM).to(dev),
            'xy': torch.from_numpy(XY).to(dev),
            'cmask': torch.from_numpy(CM).to(dev)}


def load_mapper_and_decoder(cfg, mapper_ckpt, pixcell_lora, fm, dev):
    ck_d = torch.load(mapper_ckpt, map_location='cpu', weights_only=False)
    GENES = ck_d.get('genes')
    G2I = {g: i for i, g in enumerate(GENES)}
    cfg.GENE_VOCAB = len(GENES)
    cfg.GENES = GENES
    model = GeneTokenCells256(cfg).to(dev).eval()
    model.load_state_dict(ck_d['model'])

    from diffusers import AutoencoderKL, DiffusionPipeline
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='vae',
                                         torch_dtype=torch.float16)
    pix = DiffusionPipeline.from_pretrained('StonyBrook-CVLab/PixCell-256', vae=vae,
                                             custom_pipeline='StonyBrook-CVLab/PixCell-pipeline',
                                             torch_dtype=torch.float16, trust_remote_code=True).to(dev)
    if fm != 'uni2h':
        swap_caption_projection(pix.transformer, fm)
    if pixcell_lora:
        from peft import PeftModel
        pix.transformer = PeftModel.from_pretrained(pix.transformer, pixcell_lora).to(dev)
    pix.transformer = pix.transformer.to(dev).to(torch.float16)
    pix.set_progress_bar_config(disable=True)
    return model, G2I, pix


@torch.no_grad()
def render(model, pix, batch, steps=30, guidance=1.5, seed=0, dev='cuda'):
    emb = model(batch)[0]
    cond = emb.to(pix.device).to(pix.dtype).view(1, 1, -1)
    unc = pix.get_unconditional_embedding(1)
    gen = torch.Generator(device=pix.device).manual_seed(seed)
    with torch.amp.autocast('cuda'):
        img = pix(uni_embeds=cond, negative_uni_embeds=unc, generator=gen,
                   num_inference_steps=steps, guidance_scale=guidance,
                   output_type='pt').images[0]
    return (img.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='comma-separated real slides to fit from')
    ap.add_argument('--markers', required=True, help='JSON file: {type_name: [gene, ...]}')
    ap.add_argument('--marker-min-z', type=float, default=0.5)
    ap.add_argument('--mode', choices=['celltype', 'density', 'composition'], required=True)
    ap.add_argument('--mapper-ckpt', required=True)
    ap.add_argument('--pixcell-lora', required=True)
    ap.add_argument('--pairbias-k', type=int, default=None,
                     help='must match the checkpoint -- see README, "Defaults that matter"')
    ap.add_argument('--fm', default='uni2h')
    ap.add_argument('--out', default=None, help='override cfg.OUT (tile cache root)')
    ap.add_argument('--n-fields', type=int, default=1, help='fields per type (celltype mode)')
    ap.add_argument('--density-counts', default='19,28,38,48,66',
                     help='comma-separated cell counts to ramp to (density mode)')
    ap.add_argument('--composition-fracs', default='0.0,0.25,0.5,0.75,1.0',
                     help='comma-separated melanoma-vs-other mixing fractions (composition mode)')
    ap.add_argument('--composition-types', default=None,
                     help='typeA,typeB for composition mode (defaults to the first two markers '
                          'keys)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--results-dir', default='./synthesis_out')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=1.5)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    samples = a.samples.split(',')
    markers = json.load(open(a.markers))

    cfg = make_cfg_256(seed=a.seed)
    if a.out:
        cfg.OUT = a.out
    cfg.SAMPLES = samples
    cfg.FM = a.fm
    if a.pairbias_k is not None:
        cfg.PAIRBIAS_K = a.pairbias_k
    cfg.K_MAX_GENES = 87  # matches gestalt.tile256.dataset's cache builders
    max_requested = max(
        [int(x) for x in a.density_counts.split(',')] if a.mode == 'density' else [0],
        [36] if a.mode == 'composition' else [0])
    GENES, G2I = load_vocab(cfg)

    model, mapper_G2I, pix = load_mapper_and_decoder(cfg, a.mapper_ckpt, a.pixcell_lora, a.fm, dev)
    # the mapper's own vocab (from its checkpoint) is authoritative for gene ids fed to it;
    # if it differs from the fresh-build vocab used to fit the expression model, that's a
    # real mismatch the caller must resolve (same panel/gene set on both sides), not something
    # to silently paper over here.
    if mapper_G2I != G2I:
        raise ValueError('mapper checkpoint vocab does not match --samples vocab; use the same '
                          'panel the mapper was trained on')

    cell_expr_raw, xy_by_tile, n_detected = _load_real_cells(cfg, G2I, samples)
    cfg.N_MAX_CELLS = max(max(len(xy) for xy in xy_by_tile), max_requested) + 8
    labels_by_tile, names = score_cell_types(cell_expr_raw, G2I, markers, a.marker_min_z)

    # gene_mu/gene_sd for standardization: same convention as the 256px train cache (mean/std
    # over detected values only), computed here directly since this fit is independent of that
    # cache's train/val split.
    gene_mu = np.zeros(cfg.GENE_VOCAB, np.float32)
    gene_sd = np.ones(cfg.GENE_VOCAB, np.float32)
    for g in range(cfg.GENE_VOCAB):
        col = np.concatenate([d[:, g] for d in cell_expr_raw])
        col = col[col != 0]
        if len(col) > 1:
            gene_mu[g], gene_sd[g] = col.mean(), max(col.std(), 1e-6)

    models = fit_models(cell_expr_raw, xy_by_tile, n_detected, labels_by_tile, names,
                         gene_mu, gene_sd, seed=a.seed)

    rng = np.random.default_rng(a.seed)

    if a.mode == 'celltype':
        for name, m in models.items():
            for i in range(a.n_fields):
                xy, expr, nd = m.sample_field(rng=rng)
                batch = field_to_batch(xy, expr, nd, cfg, dev)
                img = render(model, pix, batch, a.steps, a.guidance, a.seed, dev)
                from PIL import Image
                Image.fromarray(img).save(f'{a.results_dir}/celltype_{name}_{i}.png')
                print(f'[celltype] {name} field {i}: n_cells={len(xy)} -> saved')

    elif a.mode == 'density':
        # "starting from one real tile ... we add cells drawn from that tile's own composition
        # and its own expression state" (sec:supp-synthesis): the state is the REAL tile's own
        # empirical mean, not a fresh GMM draw, so pick one real field per requested type and
        # hold its state fixed while ramping n_cells.
        counts = [int(x) for x in a.density_counts.split(',')]
        for name, m in models.items():
            expr_t = [d for d, lab in zip(cell_expr_raw, labels_by_tile) if (lab == names.index(name)).sum() >= 3]
            if not expr_t:
                continue
            base = _standardize([expr_t[0]], gene_mu, gene_sd)[0]
            base_pca = m.pca.transform(base)
            state = base_pca.mean(0)
            for n in counts:
                xy, expr, nd = m.sample_field(n_cells=n, rng=rng, state=state)
                batch = field_to_batch(xy, expr, nd, cfg, dev)
                img = render(model, pix, batch, a.steps, a.guidance, a.seed, dev)
                from PIL import Image
                Image.fromarray(img).save(f'{a.results_dir}/density_{name}_n{n}.png')
                print(f'[density] {name} n_cells={n} -> saved')

    elif a.mode == 'composition':
        type_names = (a.composition_types.split(',') if a.composition_types
                      else list(models)[:2])
        tA, tB = type_names
        fracs = [float(x) for x in a.composition_fracs.split(',')]
        n_fixed = 36
        xy_fixed = None
        for frac in fracs:
            n_a = int(round(n_fixed * frac))
            if xy_fixed is None:
                # fix positions/count across the whole sweep by generating them once, from
                # whichever of the two types has the tighter (larger-constant) packing.
                xy_fixed = hardcore_positions_wrapper(models[tA], models[tB], n_fixed, rng)
            state_a = models[tA].gmm.sample(1)[0][0]
            state_b = models[tB].gmm.sample(1)[0][0]
            xy = xy_fixed
            expr = np.zeros((n_fixed, models[tA].n_genes), np.float32)
            nd = np.zeros(n_fixed, np.int64)
            for j, (mdl, state, lo, hi) in enumerate([
                    (models[tA], state_a, 0, n_a), (models[tB], state_b, n_a, n_fixed)]):
                if hi <= lo:
                    continue
                sub_xy, sub_expr, sub_nd = mdl.sample_field(n_cells=hi - lo, rng=rng, state=state)
                expr[lo:hi], nd[lo:hi] = sub_expr, sub_nd
            batch = field_to_batch(xy, expr, nd, cfg, dev)
            img = render(model, pix, batch, a.steps, a.guidance, a.seed, dev)
            from PIL import Image
            Image.fromarray(img).save(f'{a.results_dir}/composition_frac{frac:.2f}.png')
            print(f'[composition] {tA}={frac:.2f} {tB}={1 - frac:.2f} -> saved')


def hardcore_positions_wrapper(model_a, model_b, n, rng):
    from .placement import hardcore_positions
    c = min(model_a.min_sep_const, model_b.min_sep_const)
    return hardcore_positions(n, c, rng=rng)


if __name__ == '__main__':
    main()

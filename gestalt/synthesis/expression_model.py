# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The statistical expression model: a two-stage random-effects draw per field (a shared
"state" for the field, then per-cell profiles correlated by an exponential covariogram around
it), following sec/5_synthesis.tex's method paragraph. One instance is fit per cell type.

This is a reconstruction from that paragraph's description, not a copy of the original
prototype (see gestalt/synthesis/__init__.py's module docstring for why). Specific modelling
choices the paper's prose leaves open are called out below.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from .placement import fit_min_separation, hardcore_positions


def _fit_covariogram(residuals_by_tile, xy_by_tile):
    """Fit a single exponential spatial kernel shared across PCA dimensions: correlation
    exp(-dist/range), range chosen by log-linear regression of mean per-dimension covariance
    against pairwise distance, over all within-tile cell pairs across every tile of this type.

    Per-dimension marginal variance is kept separate (see `sigma2`), since PCA components carry
    very different variance by construction; only the *spatial decay* is assumed shared.
    """
    dists, covs = [], []
    for res, xy in zip(residuals_by_tile, xy_by_tile):
        n = len(xy)
        if n < 2:
            continue
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        c = (res @ res.T) / res.shape[1]  # mean per-dim covariance between every cell pair
        iu = np.triu_indices(n, k=1)
        dists.append(d[iu])
        covs.append(c[iu])
    dists = np.concatenate(dists)
    covs = np.concatenate(covs)
    pos = covs > 0
    if pos.sum() < 10:
        return 1.0  # not enough same-sign pairs to fit a decay; caller falls back to no decay
    slope, intercept = np.polyfit(dists[pos], np.log(covs[pos]), 1)
    rng = -1.0 / slope if slope < 0 else np.inf
    return float(np.clip(rng, 1e-3, 10.0))


class TileTypeExpressionModel:
    """Fit and sample synthetic (position, expression) fields for one cell type.

    Args:
        n_pca (int): PCA dimensionality cells are reduced to before the mixture/covariogram/
            regression fits (100 in the paper).
        bic_range (tuple[int, int]): inclusive range of mixture-component counts the Gaussian
            mixture over field states is selected from, by BIC (1-20 in the paper).
    """

    def __init__(self, n_pca=100, bic_range=(1, 20)):
        self.n_pca = n_pca
        self.bic_range = bic_range

    def fit(self, cell_expr_by_tile, xy_by_tile, n_detected_by_tile, seed=0):
        """
        Args:
            cell_expr_by_tile (list[np.ndarray]): one (n_cells_t, n_genes) dense, standardized
                ((value-mu)/sd, zero-filled for undetected genes) expression matrix per real tile
                of this type.
            xy_by_tile (list[np.ndarray]): one (n_cells_t, 2) array of cell positions per tile,
                in the same normalized [-1, 1] frame the mapper consumes.
            n_detected_by_tile (list[np.ndarray]): one (n_cells_t,) int array per tile, the
                number of detected genes for each of that tile's real cells.
        """
        all_cells = np.concatenate(cell_expr_by_tile, axis=0)
        self.n_genes = all_cells.shape[1]
        self.pca = PCA(n_components=min(self.n_pca, all_cells.shape[0] - 1, all_cells.shape[1]))
        self.pca.fit(all_cells)

        pca_by_tile = [self.pca.transform(c) for c in cell_expr_by_tile]
        states = np.stack([p.mean(0) for p in pca_by_tile])  # (n_tiles, n_pca) -- one state/tile

        best = None
        lo, hi = self.bic_range
        for k in range(lo, min(hi, len(states)) + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='diag',
                                   random_state=seed, reg_covar=1e-4).fit(states)
            bic = gmm.bic(states)
            if best is None or bic < best[0]:
                best = (bic, gmm)
        self.gmm = best[1]

        residuals_by_tile = [p - s for p, s in zip(pca_by_tile, states)]
        self.sigma2 = np.concatenate(residuals_by_tile, axis=0).var(axis=0)  # (n_pca,) per-dim
        self.range_ = _fit_covariogram(residuals_by_tile, xy_by_tile)

        pca_all = np.concatenate(pca_by_tile, axis=0)
        n_det_all = np.concatenate(n_detected_by_tile)
        self.gene_count_reg = LinearRegression().fit(pca_all, n_det_all)
        self.gene_count_floor = max(1, int(n_det_all.min()))
        self.gene_count_ceil = int(n_det_all.max())

        self.min_sep_const = fit_min_separation(xy_by_tile)
        self.cell_counts = np.array([len(xy) for xy in xy_by_tile])
        return self

    def sample_field(self, n_cells=None, rng=None, state=None):
        """Draw one synthetic field.

        Args:
            n_cells (int, optional): requested cell count. Defaults to a draw from the fitted
                real per-tile count distribution (`self.cell_counts`).
            rng (np.random.Generator, optional): source of randomness.
            state (np.ndarray, optional): an explicit (n_pca,) field state, overriding the GMM
                draw -- used by sec:supp-synthesis's density/composition sweeps, which hold the
                state fixed and vary only count or a cell-type mixing fraction.

        Returns:
            xy (np.ndarray): (n_cells, 2) positions.
            gene_expr (np.ndarray): (n_cells, n_genes) dense reconstructed expression.
            n_detected (np.ndarray): (n_cells,) int, how many of each cell's genes to keep as
                "detected" -- callers select the top-n_detected entries of gene_expr by
                magnitude before handing cells to the mapper, matching its sparse per-cell input.
        """
        rng = rng or np.random.default_rng()
        if n_cells is None:
            n_cells = int(rng.choice(self.cell_counts))
        if state is None:
            state = self.gmm.sample(1)[0][0]

        xy = hardcore_positions(n_cells, self.min_sep_const, rng=rng)
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        corr = np.exp(-d / self.range_) if np.isfinite(self.range_) else np.eye(n_cells)
        L = np.linalg.cholesky(corr + 1e-6 * np.eye(n_cells))

        pca_cells = np.empty((n_cells, len(state)))
        for k in range(len(state)):
            z = rng.standard_normal(n_cells)
            pca_cells[:, k] = state[k] + np.sqrt(self.sigma2[k]) * (L @ z)

        gene_expr = self.pca.inverse_transform(pca_cells)
        n_detected = np.clip(np.round(self.gene_count_reg.predict(pca_cells)).astype(int),
                              self.gene_count_floor, self.gene_count_ceil)
        return xy.astype(np.float32), gene_expr.astype(np.float32), n_detected

"""Gene-set expression specificity in pseudobulk profiles (as in the SEA-AD polygenic tester).

Each label's pseudobulk is its cells' summed UMIs. Profiles are log2(CPM + 1), and every
gene is z-scored across the profiles of one annotation level, removing its average
expression. Within each profile, genes are ranked by z-score; the AUROC asks whether the
target genes rank above the other genes of the universe.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm, rankdata

MIN_CELLS = 20


def specificity_z(counts, n_cells, min_cells=MIN_CELLS):
    """z-scored log2(CPM + 1) for profiles with at least min_cells cells and some counts.

    Returns the kept-profile mask and a profiles x genes float32 array. Genes with no
    variation across the kept profiles get z = 0 everywhere (a tie in every profile).
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    keep = (np.asarray(n_cells) >= min_cells) & (totals > 0)
    x = np.log2(counts[keep]/totals[keep, None]*1e6 + 1)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    z = np.divide(x - mean, sd, out=np.zeros_like(x), where=sd > 0)
    return keep, z.astype(np.float32)


def rank_profiles(z):
    """Average ranks of genes within each profile, and each profile's tie term sum(t^3 - t)."""
    ranks = rankdata(z, axis=1).astype(np.float32)  # halves stay exact below 2**23
    ties = np.zeros(len(z))
    for row, values in enumerate(np.sort(z, axis=1)):
        starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
        t = np.diff(starts).astype(np.float64)
        ties[row] = (t**3 - t).sum()
    return ranks, ties


def specificity_table(ranks, ties, target, labels, n_cells, label_column, count_column='Cells'):
    """AUROC of the target genes (column positions in ranks) within every profile.

    Two-sided p-values use the Mann-Whitney normal approximation with tie and
    continuity correction, matching scipy.stats.mannwhitneyu(method='asymptotic').
    """
    n_genes = ranks.shape[1]
    target = np.unique(np.asarray(target, dtype=np.int64))
    k, m = len(target), n_genes - len(target)
    if k == 0 or m == 0:
        raise ValueError('The universe needs target and non-target genes')
    u = ranks[:, target].sum(axis=1, dtype=np.float64) - k*(k+1)/2
    sigma = np.sqrt(k*m/12*((n_genes+1) - ties/(n_genes*(n_genes-1))))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (np.maximum(u, k*m-u) - k*m/2 - 0.5)/sigma
        p = np.clip(2*norm.sf(z), 0, 1)
    auc = u/(k*m)
    constant = ~(sigma > 0)  # every gene tied: no evidence of specificity
    auc[constant], p[constant] = 0.5, 1.0
    table = pd.DataFrame({label_column: labels, count_column: np.asarray(n_cells, dtype=np.int64), 'AUROC': auc, 'p-value': p})
    corrected = false_discovery_control(p, method='bh') if len(p) else p.copy()
    # Apply the reporting floor AFTER BH, so it does not alter the correction.
    floor = np.nextafter(np.float64(0), np.float64(1))
    table['p-value'] = np.where(p == 0, floor, p)
    table['pFDR'] = np.where(corrected == 0, floor, corrected)  # Benjamini-Hochberg adjusted
    return table.sort_values(['AUROC', label_column], ascending=[False, True]).reset_index(drop=True)


def format_p_value(p):
    """Three significant digits: standard notation from 1e-4 up, scientific below (4.29e-6)."""
    if not np.isfinite(p):
        return ''
    if p >= 1e-4:
        return f'{p:.3g}'
    return re.sub(r'e([+-])0*(\d)', r'e\1\2', f'{p:.2e}')

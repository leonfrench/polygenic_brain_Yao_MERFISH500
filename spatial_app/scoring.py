"""Vectorized exact ROC AUC from sparse ranks, including subset-universe reranking."""
from dataclasses import dataclass
import re
import numpy as np


def parse_genes(text):
    """Whitespace, comma and semicolon separated approved symbols; no alias guesses."""
    return sorted(set(token.upper() for token in re.split(r'[\s,;]+', text.strip()) if token))


@dataclass(frozen=True)
class Query:
    target: tuple[int, ...]
    background: tuple[int, ...] | None
    target_symbols: tuple[str, ...]
    unknown_target: tuple[str, ...]
    unknown_background: tuple[str, ...]
    outside_background: tuple[str, ...]
    universe_size: int


def resolve_query(target_text, background_text, symbols):
    lookup = {str(s).upper(): i for i, s in enumerate(symbols)}
    requested = parse_genes(target_text)
    background_requested = parse_genes(background_text)
    unknown_target = tuple(s for s in requested if s not in lookup)
    known_target = set(requested) & lookup.keys()
    unknown_background = tuple(s for s in background_requested if s not in lookup)
    if background_text.strip():
        universe = set(background_requested) & lookup.keys()
        background = tuple(sorted(lookup[s] for s in universe))
    else:
        universe = set(lookup)
        background = None
    outside = tuple(sorted(known_target-universe))
    matched = sorted(known_target & universe)
    if not matched:
        raise ValueError('No target genes match the scoring universe. Use approved gene symbols and check your background.')
    if len(universe) <= len(matched):
        raise ValueError('The universe needs at least one non-target gene to calculate AUROC. Add background genes or leave background blank.')
    return Query(tuple(sorted(lookup[s] for s in matched)), background, tuple(matched),
                 unknown_target, unknown_background, outside, len(universe))


def genes_used(genes, query, columns=('symbol',)):
    """Every gene of the query's universe with its role, targets first, then by symbol.

    genes is the dataset's gene table in gene_index order; columns are copied from it.
    """
    universe = np.arange(len(genes)) if query.background is None else np.asarray(query.background)
    table = genes.iloc[universe][list(columns)].reset_index(drop=True)
    table.insert(0, 'role', np.where(np.isin(universe, query.target), 'target', 'background'))
    return table.sort_values(['role', 'symbol'], ascending=[False, True], ignore_index=True)


def _indices(values, n_genes):
    array = np.asarray(list(values))
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError('Gene indices must be integers')
    array = np.unique(array.astype(np.int64))
    if np.any((array < 0) | (array >= n_genes)):
        raise ValueError('Gene index outside dataset')
    return array


def rerank_background(shard, universe, n_genes):
    """Return row, gene, doubled average rank, zero rank inside the new universe.

    No cell loop or dense cells x genes allocation. Sort expressed background
    genes by (cell, original rank), then recompute within-cell tie groups.
    All omitted genes are zero-expression ties within that background.
    """
    n = len(shard['cell_index'])
    membership = np.zeros(n_genes, dtype=bool)
    membership[universe] = True
    all_rows = np.repeat(np.arange(n), np.diff(shard['indptr'].astype(np.int64)))
    keep = membership[shard['gene_index']]
    rows, genes, ranks = all_rows[keep], shard['gene_index'][keep], shard['rank2'][keep]
    counts = np.bincount(rows, minlength=n)
    zeros = len(universe)-counts
    zero_rank2 = np.where(zeros > 0, zeros+1, 0).astype(np.int64)
    if not len(rows):
        return rows, genes, np.empty(0, dtype=np.int64), zero_rank2
    order = np.lexsort((ranks, rows))
    rows, genes, ranks = rows[order], genes[order], ranks[order]
    group_start = np.r_[0, np.flatnonzero((np.diff(rows) != 0) | (np.diff(ranks.astype(np.int64)) != 0))+1]
    group_end = np.r_[group_start[1:], len(rows)]
    row_start = np.r_[0, np.cumsum(counts)[:-1]]
    # For a 0-based [start,end) tie, doubled average 1-based rank=start+end+1.
    group_rank2 = group_start+group_end+1-2*row_start[rows[group_start]]+2*zeros[rows[group_start]]
    rank2 = np.repeat(group_rank2, group_end-group_start)
    return rows, genes, rank2, zero_rank2


def auroc_vectorized(shard, target_indices, n_genes, background_indices=None):
    """One exact set-vs-complement score per cell, preserving half credit for ties.

    Background denotes the complete universe; targets outside it are excluded.
    Empty/full target sets return NaN. All reductions operate across entire
    shards using NumPy indexing, sorting and bincount, with no per-cell loop.
    """
    target = _indices(target_indices, n_genes)
    background = None if background_indices is None else _indices(background_indices, n_genes)
    if background is not None:
        target = np.intersect1d(target, background, assume_unique=True)
    g = n_genes if background is None else len(background)
    k = len(target)
    n = len(shard['cell_index'])
    if k == 0 or k == g:
        return np.full(n, np.nan)
    if background is None or len(background) == n_genes:
        rows = np.repeat(np.arange(n), np.diff(shard['indptr'].astype(np.int64)))
        genes, ranks, zero2 = shard['gene_index'], shard['rank2'], shard['zero_rank2'].astype(np.int64)
    else:
        rows, genes, ranks, zero2 = rerank_background(shard, background, n_genes)
    membership = np.zeros(n_genes, dtype=bool)
    membership[target] = True
    hit = membership[genes]
    correction = ranks[hit].astype(np.int64)-zero2[rows[hit]]
    rank_sum2 = k*zero2 + np.bincount(rows[hit], weights=correction, minlength=n)
    result = (rank_sum2-k*(k+1))/(2*k*(g-k))
    return np.clip(result, 0, 1)

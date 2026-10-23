"""Small file reader and exact per-observation AUROC helper; no application code."""
from pathlib import Path
import json
import numpy as np


def load_shard(path):
    """Materialize one small NPZ file; safe for caching in a future app."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def auroc(shard, gene_indices, n_genes):
    """Set-vs-complement ROC AUC. Unknown symbols must be handled before this call.

    Gene indices are deduplicated. Empty/full sets yield NaN; zeros and positive
    ties receive half credit. The background is the entire exported universe.
    Returns one score per shard row, aligned with shard['cell_index'].
    """
    raw = np.asarray(list(gene_indices))
    if raw.size and (not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('Gene indices must be integers')
    selected = np.unique(raw.astype(np.int64))
    if ((selected < 0) | (selected >= n_genes)).any():
        raise ValueError('Gene index outside the exported universe')
    k = len(selected)
    n = len(shard['cell_index'])
    if k == 0 or k == n_genes:
        return np.full(n, np.nan)
    membership = np.zeros(n_genes, dtype=bool)
    membership[selected] = True
    ptr = shard['indptr']
    # Sum sparse rank corrections on top of the shared zero rank, without
    # allocating a cells x genes array or a repeated row-index array.
    scores = np.empty(n, dtype=np.float64)
    hit = membership[shard['gene_index']]
    for row in range(n):
        start, stop = int(ptr[row]), int(ptr[row+1])
        base = float(shard['zero_rank2'][row])
        vals = shard['rank2'][start:stop][hit[start:stop]].astype(np.float64)
        rank_sum = (k*base + np.sum(vals-base))/2
        scores[row] = (rank_sum-k*(k+1)/2)/(k*(n_genes-k))
    return scores


def resolve_symbols(genes, symbols):
    """Resolve exact approved symbols; return indices and explicitly missing names."""
    lookup = dict(zip(genes.symbol, genes.gene_index))
    requested = sorted(set(symbols))
    return ([int(lookup[s]) for s in requested if s in lookup],
            [s for s in requested if s not in lookup])


def spatial_paths(manifest, bbox):
    """Files whose stored coordinate bounds intersect [xmin, ymin, xmax, ymax].

    Then filter cells to the actual viewport and deduplicate by cell_index when
    combining spatial files with the progressive sample already in memory.
    """
    xmin,ymin,xmax,ymax = bbox
    if xmin>xmax or ymin>ymax:
        raise ValueError('Invalid viewport bounds')
    return [f['path'] for f in manifest['files'] if f.get('layout')=='spatial'
            and f['bbox'][0]<=xmax and f['bbox'][2]>=xmin
            and f['bbox'][1]<=ymax and f['bbox'][3]>=ymin]


def load_manifest(directory):
    return json.loads((Path(directory)/'manifest.json').read_text())

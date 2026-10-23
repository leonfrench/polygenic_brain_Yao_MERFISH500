"""MERFISH whole mouse brain sections (Allen Brain Cell Atlas) -> exact sparse ranks in per-section batches.

Inputs in --raw (data/MERFISH), laid out as in the Allen Brain Cell Atlas bucket:
  MERFISH-C57BL6J-638850/20230630/C57BL6J-638850-raw.h5ad   all segmented cells x 550 barcodes, raw transcript counts (CSR)
  MERFISH-C57BL6J-638850/20230830/cell_metadata.csv          cells that passed QC and were mapped to the taxonomy:
      cell_label, brain_section_label, cluster_alias, average_correlation_score, x, y, z (section coordinates, mm)
  MERFISH-C57BL6J-638850/20230830/gene.csv                   the panel: 500 genes and 50 Blank control barcodes
  MERFISH-C57BL6J-638850-CCF/20231215/reconstructed_coordinates.csv  cell_label, aligned x, y, z (mm), parcellation_index
  WMB-taxonomy/20231215/views/cluster_to_cluster_annotation_membership_{pivoted,color}.csv  class to cluster names, colours
  Allen-CCF-2020/20230630/views/parcellation_to_parcellation_term_membership_acronym.csv  index -> division to substructure
  Allen-CCF-2020/20230630/parcellation_term.csv              acronym names and Allen colours
The annotation tables are downloaded from the atlas bucket when missing, and the MGI marker
lists into --reference (data/reference).

Only cells with a cluster and a CCF registration are kept. The universe is every panel gene
(Blank barcodes excluded), under its current MGI symbol where that is unambiguous.

  .conda-env/bin/python scripts/prepare_merfish_cells.py --force
"""
from pathlib import Path
import argparse, colorsys, datetime, hashlib, json, platform, shutil, time
import h5py
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
MGI_URLS = {'MRK_List2.rpt': 'https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt',
            'MRK_List1.rpt': 'https://www.informatics.jax.org/downloads/reports/MRK_List1.rpt'}
ATLAS_URL = 'https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/metadata/'
DATASET = 'C57BL6J-638850'
EXPRESSION = f'MERFISH-{DATASET}/20230630/{DATASET}-raw.h5ad'
CELL_METADATA = f'MERFISH-{DATASET}/20230830/cell_metadata.csv'
PANEL = f'MERFISH-{DATASET}/20230830/gene.csv'
TABLES = dict(coordinates=f'MERFISH-{DATASET}-CCF/20231215/reconstructed_coordinates.csv',
              taxonomy='WMB-taxonomy/20231215/views/cluster_to_cluster_annotation_membership_pivoted.csv',
              taxonomy_colors='WMB-taxonomy/20231215/views/cluster_to_cluster_annotation_membership_color.csv',
              parcellation='Allen-CCF-2020/20230630/views/parcellation_to_parcellation_term_membership_acronym.csv',
              parcellation_terms='Allen-CCF-2020/20230630/parcellation_term.csv')
# Sixteen CCF-registered sections from the olfactory bulb to the cerebellum, about 0.9 mm apart.
# Where the nearest section is torn or partial, an intact neighbour within 0.2 mm is used.
SECTIONS = ['67', '64', '62', '59', '56', '52', '47', '44', '40', '36', '32', '28', '24', '15', '09', '05']
CELL_LEVELS = {'cell_class': ('class', 'Cell class'), 'cell_subclass': ('subclass', 'Cell subclass'),
               'cell_supertype': ('supertype', 'Cell supertype'), 'cell_cluster': ('cluster', 'Cell cluster')}
REGION_LEVELS = {'region_division': ('division', 'Brain division'), 'region_structure': ('structure', 'Brain structure'),
                 'region_substructure': ('substructure', 'Brain substructure')}
LEVELS = [*CELL_LEVELS, *REGION_LEVELS]
UNASSIGNED = '#C9CED6'  # the atlas colour for unassigned tissue is black, the plot background


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path):
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


# ---------------------------------------------------------------- gene universe

def mgi_tables(reference):
    current = pd.read_csv(download(MGI_URLS['MRK_List2.rpt'], reference/'MRK_List2.rpt'), sep='\t', dtype=str, keep_default_na=False)
    current = current[current['Status'].eq('O') & current['Marker Type'].eq('Gene')]
    pc = current[current['Feature Type'].eq('protein coding gene')][['MGI Accession ID', 'Marker Symbol', 'Marker Name']]
    pc.columns = ['mgi_id', 'symbol', 'name']
    full = pd.read_csv(download(MGI_URLS['MRK_List1.rpt'], reference/'MRK_List1.rpt'), sep='\t', dtype=str, keep_default_na=False)
    withdrawn = full[full['Status'].eq('W')][['Marker Symbol', 'Current MGI Accession ID (if withdrawn)']]
    withdrawn.columns = ['old_symbol', 'mgi_id']
    return current, pc.reset_index(drop=True), withdrawn


def map_genes(names, current, pc, withdrawn):
    """Current MGI symbol first, then an unambiguous withdrawn symbol. No synonym guessing."""
    by_symbol = dict(zip(current['Marker Symbol'], current['MGI Accession ID']))
    pc_ids = dict(zip(pc.mgi_id, pc.symbol))
    old = withdrawn[withdrawn.mgi_id.ne('')].groupby('old_symbol').mgi_id.agg(lambda s: sorted(set(s)))
    rows = []
    for name in names:
        ident, method = by_symbol.get(name), 'current_symbol'
        if ident is None:
            candidates = old.get(name, [])
            method = 'withdrawn_symbol' if candidates else 'unmapped'
            ident = candidates[0] if len(candidates) == 1 else None
            if len(candidates) > 1:
                method = 'ambiguous_withdrawn_symbol'
        status = 'retained' if ident in pc_ids else ('not_protein_coding' if ident else method)
        rows.append((name, ident, pc_ids.get(ident), method, status))
    return pd.DataFrame(rows, columns=['source_gene', 'mgi_id', 'symbol', 'mapping_method', 'status'])


def panel_genes(panel, current, pc, withdrawn):
    """Every panel gene except the Blank control barcodes, under its current MGI symbol when unambiguous.

    Every panel gene was chosen as a marker, so non-coding genes stay in the universe (a
    whole-transcriptome universe would keep protein-coding genes only); unmapped genes keep
    the panel symbol.
    """
    mapping = map_genes(panel.gene_symbol.tolist(), current, pc, withdrawn)
    by_id = current.set_index('MGI Accession ID')
    mapping['symbol'] = mapping.mgi_id.map(by_id['Marker Symbol']).fillna(mapping.source_gene)
    mapping['name'] = mapping.mgi_id.map(by_id['Marker Name']).fillna(panel['name'])
    mapping['feature_type'] = mapping.mgi_id.map(by_id['Feature Type']).fillna('')
    mapping['ensembl_id'] = panel.gene_identifier.to_numpy()
    control = mapping.source_gene.str.startswith('Blank-').to_numpy()
    mapping['status'] = np.where(control, 'control_barcode', 'retained')
    if mapping.symbol[~control].duplicated().any():
        raise ValueError('Panel genes must keep distinct symbols')
    return mapping


# ---------------------------------------------------------------- ranks and pseudobulks

def rank_rows(rows, cols, values, n_rows, n_genes):
    """Doubled average ranks per row; positive values only, zeros share (Z+1)/2."""
    rows, cols, values = (np.asarray(a, dtype=np.int64) for a in (rows, cols, values))
    # One integer key per sort; np.lexsort is much slower on unordered rows.
    order = np.argsort(rows*(int(values.max(initial=0))+1) + values)
    rows, cols, values = rows[order], cols[order], values[order]
    counts = np.bincount(rows, minlength=n_rows)
    zeros = n_genes - counts
    start = np.r_[0, np.flatnonzero((np.diff(rows) != 0) | (np.diff(values) != 0)) + 1]
    end = np.r_[start[1:], len(rows)]
    row_start = np.r_[0, np.cumsum(counts)[:-1]]
    # A 0-based [start,end) tie inside the positive block has doubled 1-based rank start+end+1.
    rank2 = np.repeat(start + end + 1 - 2*row_start[rows[start]] + 2*zeros[rows[start]], end - start)
    order = np.argsort(rows*n_genes + cols)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return indptr, cols[order].astype(np.uint16), rank2[order].astype(np.uint16), np.where(zeros > 0, zeros+1, 0).astype(np.uint16)


def pseudobulk(rows, cols, values, codes, n_genes):
    """Summed counts per (label, gene) for the labels present; rows index cells, codes give each cell's label."""
    present, inverse = np.unique(codes, return_inverse=True)
    sums = np.bincount(inverse[rows]*n_genes + cols, weights=values, minlength=len(present)*n_genes)
    if sums.max(initial=0) >= 2**32:
        raise ValueError('Pseudobulk counts exceed uint32')
    return present, np.bincount(inverse, minlength=len(present)), sums.reshape(len(present), n_genes).astype(np.uint32)


def validate_rows(indptr, cols, rank2, zero2, dense_rows, n_genes, rng):
    checked = 0
    for row, dense in dense_rows:
        a, b = indptr[row], indptr[row+1]
        decoded = np.full(n_genes, zero2[row]/2); decoded[cols[a:b]] = rank2[a:b]/2
        np.testing.assert_array_equal(decoded, rankdata(dense, method='average'))
        selected = rng.choice(n_genes, 100, replace=False)
        other = np.setdiff1d(np.arange(n_genes), selected)
        truth = np.mean((dense[selected, None] > dense[other]) + 0.5*(dense[selected, None] == dense[other]))
        auc = (decoded[selected].sum() - 100*101/2) / (100*(n_genes-100))
        np.testing.assert_allclose(auc, truth, atol=1e-12)
        checked += 1
    return checked


# ---------------------------------------------------------------- per-section processing

def section_cells(h5, bounds, meta, coordinates, taxonomy, parcellation, label):
    """Kept cells of one section, in expression-matrix row order, with every annotation level."""
    start, stop = bounds[label]
    rows = pl.DataFrame({'cell_label': h5['obs/cell_label'].asstr()[start:stop], 'h5ad_row': np.arange(start, stop)})
    mapped = meta.filter(pl.col('brain_section_label') == label)
    names = taxonomy.select(c for c in taxonomy.columns if not c.endswith('_color'))
    cells = (rows.join(mapped, on='cell_label').join(coordinates, on='cell_label')
             .join(names, on='cluster_alias', how='left').join(parcellation, on='parcellation_index', how='left').sort('h5ad_row'))
    if cells.select(pl.col(c).null_count() for c in [s for s, _ in CELL_LEVELS.values()] + [s for s, _ in REGION_LEVELS.values()]).sum_horizontal().item():
        raise ValueError(f'{label}: clusters or parcellation indices missing from the annotation tables')
    if not (cells['z'] == cells['z_reconstructed']).all():
        raise ValueError(f'{label}: section and reconstructed z differ')
    report = dict(segmented_cells=stop-start, mapped_cells=mapped.height, kept_cells=cells.height,
                  # A few mapped cells (0.06% overall) have no row in the expression matrix.
                  mapped_cells_without_expression=mapped.join(rows, on='cell_label', how='anti').height)
    cells = cells.rename({source: column for column, (source, _) in {**CELL_LEVELS, **REGION_LEVELS}.items()})
    return cells.to_pandas(), report


def read_counts(x, start, stop, local, m, remap):
    """Kept-cell (row, gene_index, count) entries of matrix rows [start, stop) and each row's Blank counts."""
    ptr = x['indptr'][start:stop+1].astype(np.int64)
    data, genes = x['data'][ptr[0]:ptr[-1]], x['indices'][ptr[0]:ptr[-1]]
    if (data < 0).any() or not np.array_equal(data, np.round(data)):
        raise ValueError('Raw counts must be non-negative integers')
    rows = local[np.repeat(np.arange(stop-start), np.diff(ptr))]
    cols, values = remap[genes], data.astype(np.int64)
    keep = (rows >= 0) & (values > 0)
    blank = np.bincount(rows[keep & (cols < 0)], weights=values[keep & (cols < 0)], minlength=m).astype(np.int64)
    keep &= cols >= 0
    rows, cols, values = rows[keep], cols[keep], values[keep]
    if len(np.unique(rows*len(remap) + cols)) != len(rows):
        raise ValueError('Each cell must list a gene at most once')
    return rows, cols, values, blank


# ---------------------------------------------------------------- annotations

def hex_rgb(color):
    return tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))


def separate_colors(colors):
    """Spread lightness inside groups of (near-)identical colours, keeping each hue family."""
    labels = sorted(colors)
    groups, assigned = [], {}
    for label in labels:
        rgb = np.array(hex_rgb(colors[label]))
        for group in groups:
            if np.abs(rgb-group['rgb']).max() < 12/255:
                group['labels'].append(label); break
        else:
            groups.append(dict(rgb=rgb, labels=[label]))
    for group in groups:
        m = len(group['labels'])
        for i, label in enumerate(group['labels']):
            h, l, s = colorsys.rgb_to_hls(*hex_rgb(colors[label]))
            if m > 1:
                l = float(np.clip(l + np.linspace(-1, 1, m)[i]*min(0.22, 0.09*(m-1)), 0.12, 0.88))
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            assigned[label] = '#%02X%02X%02X' % tuple(round(v*255) for v in (r, g, b))
    return assigned


def annotation_levels(cells, taxonomy, terms):
    levels = {}
    for column, (source, title) in CELL_LEVELS.items():
        palette = taxonomy.select(source, f'{source}_color').unique()
        if palette[source].is_duplicated().any():
            raise ValueError(f'{source} labels must have one colour')
        palette = dict(palette.iter_rows())
        present = sorted(set(cells[column]))
        levels[column] = dict(kind='cell_type', title=title, labels={k: palette[k].upper() for k in present}, descriptions={})
    # parcellation_term lists an acronym once per level it appears in; the copies agree.
    terms = terms.drop_duplicates(['acronym', 'name', 'color_hex_triplet'])
    if terms.acronym.duplicated().any():
        raise ValueError('Parcellation acronyms must have one name and colour')
    terms = terms.set_index('acronym')
    for column, (_, title) in REGION_LEVELS.items():
        present = sorted(set(cells[column]))
        colors = {k: UNASSIGNED if k == 'unassigned' else terms.color_hex_triplet[k].upper() for k in present}
        descriptions = {k: 'Outside the CCF parcellation' if k == 'unassigned' else terms.name[k] for k in present}
        levels[column] = dict(kind='region', title=title, labels=separate_colors(colors), descriptions=descriptions)
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/MERFISH')
    parser.add_argument('--sections', nargs='*', default=SECTIONS, help='Section numbers, e.g. 67 40 05; shown anterior to posterior.')
    parser.add_argument('--out', type=Path, default=ROOT/'data/merfish-638850')
    parser.add_argument('--reference', type=Path, default=ROOT/'data/reference')
    parser.add_argument('--batch-cells', type=int, default=16384, help='Maximum cells per batch.')
    parser.add_argument('--batch-entries', type=int, default=12_000_000,
                        help='Maximum stored (cell, detected gene) entries per batch; keeps files well under 95 MB.')
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--force', action='store_true', help='Replace an existing output directory.')
    args = parser.parse_args()
    started = time.perf_counter()
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit(f'{args.out} is not empty; pass --force to replace it')
    staging = args.out.with_name(args.out.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    (staging/'progressive').mkdir(parents=True)
    (staging/'pseudobulk').mkdir()

    tables = {name: download(ATLAS_URL+path, args.raw/path) for name, path in TABLES.items()}
    expression, metadata_path, panel_path = (args.raw/p for p in (EXPRESSION, CELL_METADATA, PANEL))
    current, pc, withdrawn = mgi_tables(args.reference)
    panel = pd.read_csv(panel_path, dtype=str, keep_default_na=False)
    mapping = panel_genes(panel, current, pc, withdrawn)
    kept = mapping[mapping.status.eq('retained')]
    symbols = np.array(sorted(kept.symbol))
    g = len(symbols)
    if not 1 < g < 32768:
        raise ValueError('uint16 doubled ranks require 2 <= genes <= 32767')
    remap = np.full(len(mapping), -1, np.int64)
    remap[kept.index] = pd.Index(symbols).get_indexer(kept.symbol)

    meta = pl.read_csv(metadata_path, columns=['cell_label', 'brain_section_label', 'cluster_alias', 'average_correlation_score', 'x', 'y', 'z'],
                       schema_overrides={'cell_label': pl.Utf8})
    coordinates = pl.read_csv(tables['coordinates'], schema_overrides={'cell_label': pl.Utf8}).rename(
        {'x': 'rx', 'y': 'ry', 'z': 'z_reconstructed'})
    taxonomy = pl.read_csv(tables['taxonomy']).join(pl.read_csv(tables['taxonomy_colors']), on='cluster_alias', validate='1:1')
    parcellation = pl.read_csv(tables['parcellation']).drop('organ', 'category')
    terms = pd.read_csv(tables['parcellation_terms'], dtype=str, keep_default_na=False)

    with h5py.File(expression, 'r') as h5:
        if not np.array_equal(h5['var/gene_identifier'].asstr()[:], panel.gene_identifier.to_numpy()):
            raise ValueError('gene.csv must list the expression matrix genes in order')
        categories = h5['obs/brain_section_label/categories'].asstr()[:]
        codes = h5['obs/brain_section_label/codes'][:]
        if (np.diff(codes) < 0).any():
            raise ValueError('Expression rows must be grouped by section')
        bounds = {label: (int(np.searchsorted(codes, k)), int(np.searchsorted(codes, k, side='right'))) for k, label in enumerate(categories)}
        labels = [f'{DATASET}.{s}' for s in args.sections]
        missing = [s for s in labels if s not in bounds]
        if missing or len(set(labels)) != len(labels):
            raise SystemExit(f'Unknown or repeated sections: {missing or labels}')

        # Pass 1: kept cells and their annotations, sections anterior (high z) to posterior.
        metas, reports = [], {}
        for label in labels:
            cells, report = section_cells(h5, bounds, meta, coordinates, taxonomy, parcellation, label)
            if cells.empty:
                raise ValueError(f'{label} has no CCF-registered cells')
            metas.append(cells); reports[label] = report
        order = np.argsort([-m.z.iloc[0] for m in metas], kind='stable')
        metas, labels = [metas[i] for i in order], [labels[i] for i in order]
        slices = [label.rsplit('.', 1)[1] for label in labels]
        for slice_id, cells in zip(slices, metas):
            cells.insert(0, 'slice', slice_id)
        cells = pd.concat(metas, ignore_index=True)
        n = len(cells)
        cells.insert(0, 'cell_index', np.arange(n, dtype=np.uint32))
        offsets = dict(zip(slices, np.r_[0, np.cumsum([len(m) for m in metas])[:-1]]))
        for column in ['rx', 'ry', 'x', 'y']:  # micrometres, so display positions keep sub-cell precision
            cells[column] = cells[column]*1000
        cells = cells.rename(columns={'z': 'z_mm'}).drop(columns=['z_reconstructed'])
        for column in ['slice', *LEVELS]:
            cells[column] = cells[column].astype(str)
        cells['draw_order'] = np.random.default_rng(args.seed).permutation(n).astype(np.uint32)
        cells['batch'] = ''; cells['batch_row'] = np.zeros(n, np.int32)
        for column in ['n_counts', 'n_genes_detected', 'n_blank_counts']:
            cells[column] = np.zeros(n, np.int64)

        # Pass 2: exact ranks, randomly ordered per-section batches and per-section pseudobulk counts.
        label_index = {level: np.unique(cells[level].to_numpy()) for level in LEVELS}
        rng = np.random.default_rng(args.seed)
        detection = np.zeros(g, np.int64); totals = np.zeros(g, np.float64); per_slice = {}
        manifest_files, slice_entries, validated = [], [], 0
        def record(path, **extra):
            item = dict(path=str(path.relative_to(staging)), bytes=path.stat().st_size, sha256=sha256(path), **extra)
            if item['bytes'] >= 95_000_000:
                raise ValueError(f'File exceeds conservative GitHub limit: {path}')
            manifest_files.append(item); return item
        for slice_id, label in zip(slices, labels):
            t = time.perf_counter()
            part = cells.iloc[offsets[slice_id]:offsets[slice_id]+reports[label]['kept_cells']]
            m = len(part)
            start, stop = bounds[label]
            local = np.full(stop-start, -1, np.int64); local[part.h5ad_row.to_numpy()-start] = np.arange(m)
            rows, cols, values, blank = read_counts(h5['X'], start, stop, local, m, remap)
            index = part.index.to_numpy()
            cells.loc[index, 'n_counts'] = np.bincount(rows, weights=values, minlength=m).astype(np.int64)
            cells.loc[index, 'n_genes_detected'] = np.bincount(rows, minlength=m)
            cells.loc[index, 'n_blank_counts'] = blank
            per_slice[slice_id] = np.bincount(cols, minlength=g)
            detection += per_slice[slice_id]; totals += np.bincount(cols, weights=values, minlength=g)
            indptr, gene_index, rank2, zero2 = rank_rows(rows, cols, values, m, g)
            dense_rows = []
            for row in rng.choice(m, min(m, 32), replace=False):
                dense = np.zeros(g); hit = rows == row
                dense[cols[hit]] = values[hit]; dense_rows.append((row, dense))
            validated += validate_rows(indptr, gene_index, rank2, zero2, dense_rows, g, rng)
            shuffled = rng.permutation(m)
            # Equal-size batches, small enough for both the cell and stored-entry limits.
            size = max(1, min(args.batch_cells, int(args.batch_entries*m/max(1, len(gene_index)))))
            size = -(-m // -(-m // size))
            batches = []
            for b, first in enumerate(range(0, m, size)):
                chosen = shuffled[first:first+size]
                lengths = indptr[chosen+1] - indptr[chosen]
                ptr = np.r_[0, np.cumsum(lengths)]
                gather = np.repeat(indptr[chosen] - ptr[:-1], lengths) + np.arange(ptr[-1])
                path = staging/'progressive'/slice_id/f'{b:05d}.npz'; path.parent.mkdir(exist_ok=True)
                global_rows = (offsets[slice_id] + chosen).astype(np.uint32)
                np.savez_compressed(path, cell_index=global_rows, indptr=ptr.astype(np.uint32),
                                    gene_index=gene_index[gather], rank2=rank2[gather], zero_rank2=zero2[chosen])
                with np.load(path, allow_pickle=False) as z:  # round-trip check of the serialized batch
                    np.testing.assert_array_equal(z['gene_index'], gene_index[gather])
                    np.testing.assert_array_equal(z['rank2'], rank2[gather])
                cells.loc[global_rows, 'batch'] = str(path.relative_to(staging))
                cells.loc[global_rows, 'batch_row'] = np.arange(len(chosen), dtype=np.int32)
                batches.append(record(path, slice=slice_id, n_cells=len(chosen))['path'])
            arrays = {}
            for level in LEVELS:
                present, n_cells, sums = pseudobulk(rows, cols, values, np.searchsorted(label_index[level], part[level].to_numpy()), g)
                np.testing.assert_array_equal(sums.sum(axis=0, dtype=np.float64), np.bincount(cols, weights=values, minlength=g))
                arrays.update({f'{level}_index': present.astype(np.uint16), f'{level}_cells': n_cells.astype(np.uint32),
                               f'{level}_counts': sums})
            path = staging/'pseudobulk'/f'{slice_id}.npz'
            np.savez_compressed(path, **arrays)
            record(path, slice=slice_id, layout='pseudobulk')
            slice_entries.append(dict(id=slice_id, brain_section_label=label, z_mm=float(part.z_mm.iloc[0]), n_cells=m,
                                      cell_index_start=int(offsets[slice_id]), cell_index_stop=int(offsets[slice_id]+m),
                                      bbox_rxry=[float(part.rx.min()), float(part.ry.min()), float(part.rx.max()), float(part.ry.max())],
                                      batches=batches, pseudobulk=str(path.relative_to(staging)),
                                      rank_seconds=round(time.perf_counter()-t, 2), **reports[label]))
            print(f'{label}: {m:,} of {reports[label]["segmented_cells"]:,} segmented cells ranked into {len(batches)} batches '
                  f'in {time.perf_counter()-t:.1f}s', flush=True)

    genes = pd.DataFrame({'gene_index': np.arange(g, dtype=np.uint16), 'symbol': symbols})
    genes = genes.merge(kept[['symbol', 'mgi_id', 'name', 'feature_type', 'ensembl_id', 'source_gene']], on='symbol', validate='1:1')
    genes['n_cells_detected'] = detection; genes['total_counts'] = totals
    levels = annotation_levels(cells, taxonomy, terms)
    (staging/'annotations.json').write_text(json.dumps(levels, indent=1, ensure_ascii=False)+'\n'); record(staging/'annotations.json')
    cells.to_parquet(staging/'cells.parquet', index=False); record(staging/'cells.parquet')
    genes.to_parquet(staging/'genes.parquet', index=False); record(staging/'genes.parquet')
    mapping.to_parquet(staging/'gene_mapping.parquet', index=False); record(staging/'gene_mapping.parquet')
    detection_by_slice = pd.DataFrame({'gene_index': np.tile(np.arange(g, dtype=np.uint16), len(slices)),
                                       'slice': np.repeat(slices, g),
                                       'n_cells_detected': np.concatenate([per_slice[s] for s in slices])})
    detection_by_slice.to_parquet(staging/'gene_detection.parquet', index=False); record(staging/'gene_detection.parquet')
    manifest = dict(
        format='sparse-doubled-average-ranks-v1', layout='per-slice-random-batches-v1', dataset=DATASET,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        n_cells=n, n_genes=g, n_slices=len(slices), seed=args.seed, batch_cells=args.batch_cells, batch_entries=args.batch_entries,
        observations='MERFISH cells that passed QC, mapped to the whole mouse brain taxonomy and registered to the CCF; '
                     'counts are decoded transcripts per cell',
        coordinates=dict(columns='rx, ry', units='micrometres, reconstructed (section-aligned) coordinates', um_per_unit=1.0,
                         raw='x, y are the section coordinates from cell_metadata, also in micrometres; z_mm is the section position'),
        region_assignment='CCF parcellation of each cell\'s registered position; unassigned = outside the parcellation',
        sources={'expression': dict(path=EXPRESSION, sha256=sha256(expression)),
                 'cell_metadata': dict(path=CELL_METADATA, sha256=sha256(metadata_path)),
                 'panel': dict(path=PANEL, sha256=sha256(panel_path)),
                 **{name: dict(url=ATLAS_URL+path, sha256=sha256(tables[name])) for name, path in TABLES.items()},
                 'mgi': {k: dict(url=v, sha256=sha256(args.reference/k)) for k, v in MGI_URLS.items()}},
        gene_mapping_counts=mapping.status.value_counts().to_dict(),
        retained_mapping_methods=kept.mapping_method.value_counts().to_dict(),
        retained_feature_types=kept.feature_type.replace('', 'not in MGI').value_counts().to_dict(),
        ranks=dict(direction='ascending: larger rank means higher expression', ties='average', scale=2,
                   zeros='all unstored genes have rank zero_rank2 / 2; stored ranks are absolute',
                   universe='every MERFISH panel gene except the Blank control barcodes',
                   auroc='(sum(rank of set) - k*(k+1)/2) / (k*(G-k))'),
        annotation_levels={k: dict(kind=v['kind'], title=v['title'], n_labels=len(v['labels'])) for k, v in levels.items()},
        pseudobulk=dict(description='per section and annotation level: summed panel-gene counts per (label, gene), '
                                    '<level>_counts rows follow <level>_index into labels[level]; <level>_cells counts cells',
                        labels={level: label_index[level].tolist() for level in LEVELS}, cells={level: 'all' for level in LEVELS}),
        validation=dict(exact_rank_and_pairwise_auc_rows=validated, batches_roundtripped=True),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      polars=pl.__version__, scipy=scipy.__version__, h5py=h5py.__version__),
        slices=slice_entries, files=manifest_files, seconds=round(time.perf_counter()-started, 1))
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.rename(args.out)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['files', 'slices', 'pseudobulk']}, indent=2))
    print('OUTPUT_BYTES', sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()), flush=True)


if __name__ == '__main__':
    main()

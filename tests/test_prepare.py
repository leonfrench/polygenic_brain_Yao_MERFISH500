import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

# Preprocessing needs requirements-prep.txt; an app-only install (requirements.txt) skips these tests.
pytest.importorskip('h5py'); pytest.importorskip('polars')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_merfish_cells import panel_genes, read_counts, rank_rows, separate_colors, map_genes, pseudobulk  # noqa: E402


def test_panel_keeps_non_coding_genes_renames_withdrawn_and_drops_blanks():
    panel = pd.DataFrame({'gene_identifier': ['E1', 'E2', 'E3', 'E4', 'E5'],
                          'gene_symbol': ['Mbp', 'Car4', 'Rmst', 'Lhfp', 'Blank-1'],
                          'name': ['myelin', 'carbonic', 'rhabdo', 'lipoma HMGIC fusion partner', '']})
    current = pd.DataFrame({'Marker Symbol': ['Mbp', 'Ca4', 'Rmst'], 'MGI Accession ID': ['MGI:1', 'MGI:2', 'MGI:3'],
                            'Marker Name': ['myelin basic protein', 'carbonic anhydrase 4', 'rhabdomyosarcoma 2 associated transcript'],
                            'Feature Type': ['protein coding gene', 'protein coding gene', 'lncRNA gene']})
    pc = pd.DataFrame({'mgi_id': ['MGI:1', 'MGI:2'], 'symbol': ['Mbp', 'Ca4'], 'name': ['', '']})
    withdrawn = pd.DataFrame({'old_symbol': ['Car4', 'Lhfp', 'Lhfp'], 'mgi_id': ['MGI:2', 'MGI:4', 'MGI:5']})
    table = panel_genes(panel, current, pc, withdrawn).set_index('source_gene')
    assert table.symbol.to_dict() == {'Mbp': 'Mbp', 'Car4': 'Ca4', 'Rmst': 'Rmst', 'Lhfp': 'Lhfp', 'Blank-1': 'Blank-1'}
    assert table.status.to_dict() == {'Mbp': 'retained', 'Car4': 'retained', 'Rmst': 'retained', 'Lhfp': 'retained',
                                      'Blank-1': 'control_barcode'}
    assert table.loc['Rmst', 'feature_type'] == 'lncRNA gene' and table.loc['Car4', 'name'] == 'carbonic anhydrase 4'
    # An ambiguous withdrawn symbol keeps the panel symbol and name rather than guessing.
    assert table.loc['Lhfp', 'mapping_method'] == 'ambiguous_withdrawn_symbol' and table.loc['Lhfp', 'name'] == 'lipoma HMGIC fusion partner'
    assert table.ensembl_id.tolist() == ['E1', 'E2', 'E3', 'E4', 'E5']


def csr(dense):
    rows, cols = np.nonzero(dense)
    return dict(indptr=np.r_[0, np.cumsum(np.count_nonzero(dense, axis=1))].astype(np.int32),
                indices=cols.astype(np.int32), data=dense[rows, cols].astype(np.float64))


def test_read_counts_keeps_mapped_cells_and_counts_blanks_separately():
    dense = np.array([[3, 0, 1, 2], [0, 5, 0, 0], [1, 2, 1, 0], [0, 0, 0, 4]])  # the last barcode is a Blank
    remap = np.array([2, 0, 1, -1])
    local = np.array([-1, 0, 1])  # matrix rows 1-3; row 1 has no metadata, rows 2 and 3 are kept cells 0 and 1
    rows, cols, values, blank = read_counts(csr(dense), 1, 4, local, 2, remap)
    kept = np.zeros((2, 3), np.int64); kept[rows, cols] = values
    np.testing.assert_array_equal(kept, [[2, 1, 1], [0, 0, 0]])
    assert blank.tolist() == [0, 4] and values.dtype == np.int64
    with pytest.raises(ValueError, match='integers'):
        read_counts(csr(dense*0.5), 0, 4, np.arange(4), 4, remap)


def test_rank_rows_matches_rankdata_with_ties_zeros_and_empty_rows():
    rng = np.random.default_rng(3)
    g, n = 40, 25
    dense = rng.integers(0, 4, (n, g)) * (rng.random((n, g)) < .5)
    dense[0] = 0                    # no detected genes
    dense[1] = rng.integers(1, 3, g)  # no zeros
    rows, cols = np.nonzero(dense)
    shuffle = rng.permutation(len(rows))  # aggregation output is unordered
    indptr, gene_index, rank2, zero2 = rank_rows(rows[shuffle], cols[shuffle], dense[rows, cols][shuffle], n, g)
    for row in range(n):
        decoded = np.full(g, zero2[row]/2)
        a, b = indptr[row], indptr[row+1]
        assert (np.diff(gene_index[a:b].astype(int)) > 0).all()
        decoded[gene_index[a:b]] = rank2[a:b]/2
        np.testing.assert_array_equal(decoded, rankdata(dense[row], method='average'))
    assert zero2[1] == 0 and zero2[0] == g+1


def test_separate_colors_splits_identical_colours_only():
    colors = separate_colors({'a': '#7A944D', 'b': '#7C9650', 'c': '#5858E0'})
    assert colors['c'] == '#5858E0'
    assert colors['a'] != colors['b']


def test_gene_mapping_prefers_current_symbols_and_rejects_ambiguous_withdrawn():
    current = pd.DataFrame({'Marker Symbol': ['Mbp', 'Gm1'], 'MGI Accession ID': ['MGI:1', 'MGI:2']})
    pc = pd.DataFrame({'mgi_id': ['MGI:1', 'MGI:3', 'MGI:4'], 'symbol': ['Mbp', 'Pakap', 'Other'], 'name': ['', '', '']})
    withdrawn = pd.DataFrame({'old_symbol': ['Akap2', 'Split', 'Split'], 'mgi_id': ['MGI:3', 'MGI:3', 'MGI:4']})
    table = map_genes(['Mbp', 'Gm1', 'Akap2', 'Split', 'Nope'], current, pc, withdrawn).set_index('source_gene')
    assert table.loc['Mbp', 'symbol'] == 'Mbp' and table.loc['Akap2', 'symbol'] == 'Pakap'
    assert table.loc['Gm1', 'status'] == 'not_protein_coding'
    assert table.loc['Split', 'status'] == 'ambiguous_withdrawn_symbol'
    assert table.loc['Nope', 'status'] == 'unmapped'


def test_pseudobulk_sums_counts_per_present_label():
    rng = np.random.default_rng(5)
    dense = rng.poisson(1.5, (12, 7))
    codes = np.array([4, 4, 1, 4, 9, 1, 1, 4, 9, 9, 4, 1])  # global label indices; 0, 2, 3 ... absent
    rows, cols = np.nonzero(dense)
    present, n_cells, sums = pseudobulk(rows, cols, dense[rows, cols], codes, 7)
    assert present.tolist() == [1, 4, 9] and n_cells.tolist() == [4, 5, 3] and sums.dtype == np.uint32
    for i, label in enumerate(present):
        np.testing.assert_array_equal(sums[i], dense[codes == label].sum(axis=0))

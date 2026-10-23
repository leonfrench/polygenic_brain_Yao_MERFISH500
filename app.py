"""Polygenic gene-set tester for the Allen Brain Cell Atlas MERFISH mouse brain (C57BL6J-638850). Run with: streamlit run app.py"""
from functools import partial
from itertools import zip_longest
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
import streamlit as st

from scripts.rank_io import load_shard
from spatial_app.scoring import resolve_query, auroc_vectorized, genes_used
from spatial_app.view import score_limits
from spatial_app.chart import cell_chart
from spatial_app.enrichment import MIN_CELLS, specificity_z, rank_profiles, specificity_table, format_p_value
from spatial_app.orthologs import SPECIES, map_input

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('MERFISH_APP_DATA', str(ROOT/'data/merfish-638850'))).resolve()
MODES = ('Gene-set AUROC', 'Cell type', 'Region', 'Cell type AUROC', 'Region AUROC')
CELL_LEVELS = ('cell_class', 'cell_subclass', 'cell_supertype', 'cell_cluster')
REGION_LEVELS = ('region_division', 'region_structure', 'region_substructure')
PARENT = {'cell_cluster': 'cell_supertype', 'cell_supertype': 'cell_subclass', 'cell_subclass': 'cell_class'}
COLOUR_CLIP = 0.01  # colour scales span the 1st to 99th percentiles
GENE_COLUMNS = ('symbol', 'mgi_id', 'name', 'feature_type', 'ensembl_id', 'n_cells_detected')  # in the genes-used download
# The cell columns the app uses; labels load as categories, which keeps a million cells to ~80 MB.
LABEL_COLUMNS = ('brain_section_label', *CELL_LEVELS, *REGION_LEVELS)
CELL_COLUMNS = ('cell_index', 'slice', 'cell_label', 'rx', 'ry', 'draw_order', *LABEL_COLUMNS)
CITATION = ('Yao et al. A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain. '
            '*Nature* 624, 317–332 (2023). [doi:10.1038/s41586-023-06812-z](https://doi.org/10.1038/s41586-023-06812-z). '
            'Data: [Allen Brain Cell Atlas](https://brain-map.org/bkp/explore/abc-atlas).')

st.set_page_config(page_title='MERFISH mouse brain · Gene-set explorer', page_icon='◉', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top:3.5rem; padding-bottom:1rem; max-width:1700px;}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {gap:0.4rem;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {margin-bottom:0.25rem;}
[data-testid="stMetricValue"] {font-size:1.55rem; font-weight:600;}
[data-testid="stMetricLabel"] {color:#69788a; font-size:.8rem;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:0.45rem;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {margin-bottom:0; line-height:1.3;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {margin-bottom:0;}  /* its -1rem offsets a paragraph margin removed above */
.st-key-cell_chart_panel {gap:0.25rem;}
</style>''', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def metadata(path, version):
    """Read-only dataset tables shared across sessions."""
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    cells = (pd.read_parquet(root/'cells.parquet', columns=list(CELL_COLUMNS), read_dictionary=list(LABEL_COLUMNS))
             .sort_values('cell_index').reset_index(drop=True))
    genes = pd.read_parquet(root/'genes.parquet').sort_values('gene_index').reset_index(drop=True)
    annotations = json.loads((root/'annotations.json').read_text())
    detection = pd.read_parquet(root/'gene_detection.parquet').pivot(index='gene_index', columns='slice', values='n_cells_detected')
    if not np.array_equal(cells.cell_index.to_numpy(), np.arange(len(cells))):
        raise ValueError('Cell indices must be consecutive and unique')
    if not np.array_equal(genes.gene_index.to_numpy(), np.arange(len(genes))):
        raise ValueError('Gene indices must be consecutive and unique')
    if len(cells) != manifest['n_cells'] or len(genes) != manifest['n_genes']:
        raise ValueError('Metadata does not match the manifest')
    return manifest, cells, genes, annotations, detection


@st.cache_resource(max_entries=4, show_spinner=False)
def display_cells(path, version, slices):
    """Cells of the given slices in the fixed random drawing order."""
    cells = metadata(path, version)[1]
    view = cells[cells.slice.isin(slices)].sort_values('draw_order').reset_index(drop=True)
    return view, view.cell_index.to_numpy(np.int64)


@st.cache_resource(max_entries=12, show_spinner=False)
def specificity_profiles(path, version, slices, level):
    """Pseudobulk z-scores and gene ranks per label, summed over the given slices."""
    manifest = metadata(path, version)[0]
    labels = np.asarray(manifest['pseudobulk']['labels'][level])
    counts = np.zeros((len(labels), manifest['n_genes'])); cells = np.zeros(len(labels), np.int64)
    for entry in manifest['slices']:
        if entry['id'] in slices:
            with np.load(Path(path)/entry['pseudobulk'], allow_pickle=False) as z:
                index = z[f'{level}_index']
                counts[index] += z[f'{level}_counts']; cells[index] += z[f'{level}_cells']
    keep, z = specificity_z(counts, cells)
    ranks, ties = rank_profiles(z)
    return dict(labels=labels[keep], cells=cells[keep], z=z, ranks=ranks, ties=ties,
                excluded=int(((cells > 0) & ~keep).sum()))


# Retain only the current batch so progressive scoring does not accumulate
# every decoded batch in server memory.
@st.cache_resource(max_entries=1, show_spinner=False)
def cached_shard(path, checksum):
    shard = load_shard(path)
    for value in shard.values():
        value.setflags(write=False)
    return shard


try:
    data_version = (DATA/'manifest.json').stat().st_mtime_ns
    manifest, cells, genes, annotations, detection = metadata(str(DATA), data_version)
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Unable to load the prepared dataset: {exc}')
    st.stop()
checksums = {f['path']: f['sha256'] for f in manifest['files']}
slice_info = {s['id']: s for s in manifest['slices']}
slice_ids = list(slice_info)
version = f"{DATA}:{manifest['created_utc']}"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def reset_scores():
    st.session_state.scores = np.full(len(cells), np.nan)
    st.session_state.done = set()
    st.session_state.job = None
    st.session_state.query_error = None
    st.session_state.enrichment = {}


if st.session_state.get('dataset_version') != version:
    st.session_state.dataset_version = version
    reset_scores()

selected = slice_ids  # every processed section is shown, anterior to posterior

with st.sidebar:
    species_label, species_selector = st.columns([1.8, 1.4], gap='small', vertical_alignment='center')
    species_label.markdown('Input gene source:')
    species = species_selector.selectbox('Species of input genes', SPECIES, key='input_species', label_visibility='collapsed',
        help='Applies to both target and background genes. Human and Rhesus Macaque symbols are mapped to mouse orthologs (HomoloGene).')
    st.caption('Separate gene symbols with spaces, commas or new lines.')
    # Medial habenula markers on the panel.
    target_text = st.text_area('Target genes', value='Tac2\nChat\nSlc5a7\nChrnb3\nPou4f1', height=155, key='target_input',
        help='Scoring updates automatically when you finish editing and leave this field.')
    with st.expander('Background genes · optional'):
        background_text = st.text_area('Background universe', height=160, key='background_input',
            placeholder='Leave blank to use all panel genes.',
            help='The complete universe, including target and non-target genes. Targets outside it are excluded. Ranks are recomputed when you finish editing.')
        st.caption('Include your targets in this universe. Blank = all panel genes.')
    input_signature = (version, target_text, background_text, species)
    if st.session_state.get('input_signature') != input_signature:
        st.session_state.input_signature = input_signature
        reset_scores()
        st.session_state.colour_mode = MODES[0]  # a new query starts on its gene-set AUROC
        st.session_state.pop('pick', None)  # new tables start with no ticked row
        st.session_state.tables_stale = True  # clear the previous query's tables at once
        st.session_state.mapping_unmapped = ([], [])
        try:
            targets, unmapped_targets = map_input(target_text, species)
            background, unmapped_background = map_input(background_text, species)
            st.session_state.mapping_unmapped = (unmapped_targets, unmapped_background)
            if not targets:
                raise ValueError('No target genes could be mapped to mouse symbols. Check the selected species and symbol capitalization.')
            if background_text.strip() and not background:
                raise ValueError('No background genes could be mapped to mouse symbols. Check the selected species, or clear the background to use all panel genes.')
            query = resolve_query(' '.join(targets), ' '.join(background), genes.symbol)
            st.session_state.job = dict(query=query, compute_seconds=0.0, load_seconds=0.0, error=None)
        except (ValueError, OSError) as exc:
            st.session_state.query_error = str(exc)
    st.radio('Colour cells by', MODES, key='colour_mode', horizontal=False,
        help='Cell type and region AUROC colour each cell by the expression-specificity AUROC of its annotation (the tables below).')
    level_columns = st.columns(2, gap='small')
    cell_level = level_columns[0].selectbox('Cell type level', CELL_LEVELS, index=1, key='cell_level',
        format_func=lambda k: annotations[k]['title'], help='Annotation level used for cell type colours and the cell type table.')
    region_level = level_columns[1].selectbox('Region level', REGION_LEVELS, index=1, key='region_level',
        format_func=lambda k: annotations[k]['title'], help='CCF level used for region colours and the region table.')
    if st.session_state.query_error:
        st.error(st.session_state.query_error)
    unmapped_targets, unmapped_background = st.session_state.get('mapping_unmapped', ([], []))
    if species != 'Mouse':
        st.caption(f'{species} → mouse orthologs (HomoloGene). All mapped orthologs are retained and deduplicated.')
    if unmapped_targets or unmapped_background:
        with st.expander('Genes without a mouse ortholog', expanded=False):
            if unmapped_targets: st.warning('Targets: '+', '.join(unmapped_targets))
            if unmapped_background: st.warning('Background: '+', '.join(unmapped_background))
    st.divider()
    st.caption(f"MERFISH {manifest['dataset']} · {manifest['n_slices']} sections · {len(cells):,} cells\n\n"
               f"{len(genes):,} panel genes · {len(annotations['cell_cluster']['labels']):,} clusters")
    st.caption('Data source: '+CITATION)

mode = st.session_state.colour_mode
st.markdown('<div style="color:#000;font-size:24px;font-weight:400;line-height:1.3;">'
            'Polygenic enrichment tool for the MERFISH whole mouse brain atlas by Yao et al.</div>', unsafe_allow_html=True)
st.markdown(f"**{manifest['dataset']} · {len(selected)} coronal section{'s' if len(selected) > 1 else ''}, anterior to posterior**")
view, view_index = display_cells(str(DATA), data_version, tuple(selected))
plan = [b for group in zip_longest(*(slice_info[s]['batches'] for s in selected)) for b in group if b]
chart_version = f"{version}:{'|'.join(selected)}"


def enrichment_tables(query):
    """Expression-specificity tables for the cell type and region levels of all sections."""
    key = (input_signature, tuple(selected), cell_level, region_level)
    cache = st.session_state.enrichment
    if key not in cache:
        tables = []
        for level in (cell_level, region_level):
            profiles = specificity_profiles(str(DATA), data_version, tuple(selected), level)
            if query.background is None:
                ranks, ties, target = profiles['ranks'], profiles['ties'], list(query.target)
            else:  # rank genes again inside the custom universe
                universe = np.asarray(query.background)
                ranks, ties = rank_profiles(profiles['z'][:, universe])
                target = np.searchsorted(universe, query.target)
            table = specificity_table(ranks, ties, target, profiles['labels'], profiles['cells'],
                                      label_column=annotations[level]['title'])
            table.attrs['excluded'] = profiles['excluded']
            tables.append(table)
        cell_table, region_table = tables
        if cell_level in PARENT:
            parents = view.drop_duplicates(cell_level).set_index(cell_level)[PARENT[cell_level]]
            cell_table.insert(1, annotations[PARENT[cell_level]]['title'], cell_table.iloc[:, 0].map(parents).to_numpy())
        region_table.insert(1, 'Description', region_table.iloc[:, 0].map(annotations[region_level]['descriptions']).to_numpy())
        cache.clear()  # one selection at a time
        cache[key] = (cell_table, region_table)
    return cache[key]


def render_progress():
    timing_context = f"sections={len(selected)} mode={mode!r}"
    job = st.session_state.job
    pending = [b for b in plan if b not in st.session_state.done]
    if job and not job['error'] and pending:
        path = pending[0]
        try:
            started = time.perf_counter()
            shard = cached_shard(str(DATA/path), checksums[path])
            load_seconds = time.perf_counter()-started
            job['load_seconds'] += load_seconds
            started = time.perf_counter()
            q = job['query']
            batch_scores = auroc_vectorized(shard, q.target, len(genes), q.background)
            if not np.isfinite(batch_scores).all():
                raise ValueError('Scoring produced undefined values')
            st.session_state.scores[shard['cell_index']] = batch_scores
            compute_seconds = time.perf_counter()-started
            job['compute_seconds'] += compute_seconds
            st.session_state.done.add(path)
            print(f"[timing] {timing_context} batch={path} load_s={load_seconds:.3f} scoring_s={compute_seconds:.3f}", flush=True)
            if len(pending) == 1:
                st.rerun()  # Stop the fragment timer and render the completed result.
        except (OSError, ValueError, KeyError) as exc:
            job['error'] = str(exc)
            st.rerun()
    scores = st.session_state.scores[view_index]
    count = int(np.isfinite(scores).sum())
    limits = score_limits(scores)
    cols = st.columns([1, 1, 1, 1.25])
    cols[0].metric('CELLS SHOWN', f'{len(view):,}')
    cols[1].metric('SCORED', f'{count:,} / {len(view):,}')
    cols[2].metric('GENE TARGET / UNIVERSE', f"{len(job['query'].target):,} / {job['query'].universe_size:,}" if job else f'— / {len(genes):,}')
    cols[3].metric('OBSERVED AUROC RANGE', f'{limits[0]:.3f} – {limits[1]:.3f}' if limits else 'Not scored yet')
    if job:
        if job['error']:
            st.error(f"Scoring stopped: {job['error']}")
        q = job['query']
        used = genes_used(genes, q, GENE_COLUMNS)
        targets = used.symbol[used.role.eq('target')]
        excluded = len(q.unknown_target) + len(q.outside_background)
        title = f"Gene matching · {len(targets):,} target{'s' if len(targets) != 1 else ''} used in a {len(used):,}-gene universe"
        if excluded:
            title += f" · {excluded} target symbol{'s' if excluded != 1 else ''} excluded"
        with st.expander(title, expanded=False):
            st.markdown('**Targets used:** '+', '.join(targets))
            if q.unknown_target: st.warning('Targets not in dataset: '+', '.join(q.unknown_target))
            if q.unknown_background: st.warning('Background genes not in dataset: '+', '.join(q.unknown_background))
            if q.outside_background: st.warning('Targets outside background: '+', '.join(q.outside_background))
            st.caption('The download lists every gene of the scoring universe (all panel genes, or your background) with its role.')
            st.download_button('Download genes used', lambda used=used: used.to_csv(index=False).encode(),
                               file_name=f"{manifest['dataset']}-genes-used.csv", mime='text/csv', key='download_genes')
        done = sum(b in st.session_state.done for b in plan)
        label = 'Complete' if count == len(view) else ('Stopped' if job['error'] else 'Scoring')
        st.progress(count/len(view), text=f"{label} · {done} of {len(plan)} batches · vectorized scoring {job['compute_seconds']:.2f}s")
    else:
        st.info('All cells are shown. Enter a valid target gene set to calculate AUROC automatically, or colour by cell type or region.')


def render_chart(cell_table, region_table):
    job = st.session_state.job
    scores = st.session_state.scores[view_index]
    aucs = lambda table: dict(zip(table.iloc[:, 0], table['AUROC'])) if table is not None else {}
    low, high = ordinal(round(COLOUR_CLIP*100)), ordinal(round(100 - COLOUR_CLIP*100))
    with st.container(key='cell_chart_panel'):
        cell_chart(view, scores, slices=selected, annotations=annotations, version=chart_version, mode=mode,
                   cell_level=cell_level, region_level=region_level, cell_aurocs=aucs(cell_table), region_aurocs=aucs(region_table),
                   highlight=st.session_state.get('highlight_request'), clip=COLOUR_CLIP, filename=f"{manifest['dataset']}-merfish-spatial",
                   um_per_unit=manifest['coordinates']['um_per_unit'], noun='sections')
        st.caption('Each dot is one cell at its CCF-aligned (reconstructed) position; every panel uses the same frame and scale. '
                   f'Grey cells are not scored yet. Gene-set AUROC colours are relative within each section ({low} to {high} percentile, range shown above each panel), '
                   f'because AUROC ranges differ between sections; label AUROC modes share one scale clipped at the {low} and {high} percentiles. '
                   'Hover for annotations and the exact AUROC, drag to pan, scroll to zoom, double-click to reset.')
    detected = detection[selected].sum(axis=1).to_numpy()
    total = len(view)*len(genes)
    zeros = total-int(detected.sum())
    if job:
        target = list(job['query'].target)
        target_total = len(view)*len(target)
        target_zeros = target_total-int(detected[target].sum())
        st.caption(f'For expression sparsity, the matched target genes are {target_zeros/target_total:.2%} zero (not detected) across all cells, '
                   f'versus {zeros/total:.2%} across all panel genes, independent of the background input.')
    else:
        st.caption(f'For expression sparsity, {zeros/total:.2%} of entries across all cells and panel genes are zero (not detected).')


def render_tables():
    """Specificity tables; returned so the chart can colour by label AUROC.

    The first update after a new query shows a placeholder instead of the previous
    query's tables; the next update (a quarter-second later while scoring) fills them.
    """
    job = st.session_state.job
    scores = st.session_state.scores[view_index]
    complete = bool(job) and np.isfinite(scores).all()
    stale = st.session_state.pop('tables_stale', False)
    # Specificity uses pseudobulk profiles, not cell scores, so the tables need only a valid query.
    cell_table, region_table = enrichment_tables(job['query']) if job and not stale else (None, None)
    number = dict(AUROC=st.column_config.NumberColumn(format='%.4f', help='Probability that a target gene ranks above a non-target gene in this pseudobulk, by specificity z-score, with half credit for ties.'),
                  **{'p-value': st.column_config.NumberColumn(help='Two-sided Mann–Whitney U test over genes, with tie and continuity correction.'),
                     'pFDR': st.column_config.NumberColumn(help='Benjamini–Hochberg adjusted p-value (false discovery rate) across all labels at this level.')})
    # Standard notation from 1e-4 up, scientific below; the values (and sorting, downloads) stay numeric.
    p_format = {'AUROC': '{:.4f}', 'p-value': format_p_value, 'pFDR': format_p_value}
    left, right = st.columns(2, gap='medium')
    for column, kind, level, table in [(left, 'Cell type', cell_level, cell_table), (right, 'Region', region_level, region_table)]:
        with column:
            st.subheader(f"{kind} specificity · {annotations[level]['title'].lower()}")
            if table is None:
                st.info('Calculating expression specificity for the new gene set…' if job else
                        'Enter a valid target gene set to test expression specificity.')
                continue
            if table.attrs.get('excluded'):
                st.caption(f"{table.attrs['excluded']} labels with fewer than {MIN_CELLS} cells are not profiled.")
            # Region acronyms are short; their descriptions and cell type names need the room.
            first = 'small' if kind == 'Region' else 'medium'
            widths = {name: st.column_config.TextColumn(name, width=w) for name, w in zip(table.columns[:2], [first, 'medium'])}
            widths['Cells'] = st.column_config.NumberColumn('Cells', format='%d', width='small', help='Cells in this pseudobulk, across all sections.')
            # A new key resets the row selection: when the table changes, or when the other table was clicked.
            epoch = st.session_state.setdefault('pick_epochs', {}).get(kind, 0)
            pick_key = f"pick_{kind}_{level}_{abs(hash((input_signature, tuple(selected))))}_{epoch}"
            st.dataframe(table.style.format(p_format), width='stretch', hide_index=True, height=min(len(table)+1, 16)*35+3, column_config={**widths, **number},
                         key=pick_key, on_select=partial(pick_row, kind, pick_key, table.iloc[:, 0].tolist()), selection_mode='single-row')
            st.download_button(f'Download {kind.lower()} table', table.to_csv(index=False).encode(),
                               file_name=f"{manifest['dataset']}-{kind.lower().replace(' ', '')}-{level}-specificity.csv",
                               mime='text/csv', key=f'download_{kind}')
    st.caption('Expression specificity: each label\'s pseudobulk sums the transcript counts of its cells across all sections and is converted to '
               'log₂(CPM + 1). Each gene is z-scored across the labels of the chosen level, removing its average expression, and genes are '
               'ranked within each label. AUROC > 0.5 = the target genes are more specific to that label than other genes; '
               '< 0.5 = less specific; 0.5 = no preference. With a custom background, genes are ranked within that universe. '
               'Two-sided Mann–Whitney p-values over genes, adjusted per table with Benjamini–Hochberg (pFDR); values that underflow are '
               'reported at the smallest positive float (≈4.94 × 10⁻³²⁴). Co-expressed genes are not independent, so p-values are exploratory.')
    if complete:
        columns = view[['cell_index', 'cell_label', 'brain_section_label', 'rx', 'ry', *CELL_LEVELS, *REGION_LEVELS]]
        def scores_csv(columns=columns, values=scores.copy()):
            # Built only when clicked: hundreds of thousands of rows take seconds to format.
            return columns.assign(auroc=values).sort_values('cell_index').to_csv(index=False).encode()
        st.download_button('Download cell scores', scores_csv,
                           file_name=f"{manifest['dataset']}-auroc.csv", mime='text/csv', key='download_scores')
    return cell_table, region_table


pending_batches = st.session_state.job and not st.session_state.job['error'] and any(b not in st.session_state.done for b in plan)


def pick_row(kind, key, labels):
    """Row tick: colour by that table's specificity AUROC and highlight the label.

    Unticking restores the colour mode from before the first tick, unless the
    colour mode was changed by hand in between.
    """
    rows = st.session_state[key]['selection']['rows']
    label = labels[rows[0]] if rows else None
    pick = st.session_state.get('pick')
    if label is not None:
        mode = 'Cell type AUROC' if kind == 'Cell type' else 'Region AUROC'
        before = pick['before'] if pick and st.session_state.colour_mode == pick['mode'] else st.session_state.colour_mode
        st.session_state.pick = dict(before=before, mode=mode)
        st.session_state.colour_mode = mode
        other = 'Region' if kind == 'Cell type' else 'Cell type'
        epochs = st.session_state.setdefault('pick_epochs', {})
        epochs[other] = epochs.get(other, 0) + 1  # only one table shows a ticked row
    else:
        st.session_state.pop('pick', None)
        if pick and st.session_state.colour_mode == pick['mode']:
            st.session_state.colour_mode = pick['before']
    seq = (st.session_state.get('highlight_request') or {}).get('seq', 0) + 1
    st.session_state.highlight_request = dict(label=label, seq=seq)
    st.session_state.full_rerun = True  # the colour control lives outside the fragment


@st.fragment(run_every=0.25 if pending_batches else None)
def render_explorer():
    if st.session_state.pop('full_rerun', False):
        st.rerun(scope='app')
    # One timer prevents competing fragments from starving chart updates. Containers fix the
    # layout order; the tables are filled first, so a new query's tables replace the old ones
    # before the slower scoring and plot updates.
    progress_box, chart_box, tables_box = st.container(), st.container(), st.container()
    with tables_box:
        tables = render_tables()
    with progress_box:
        render_progress()
    with chart_box:
        render_chart(*tables)


render_explorer()

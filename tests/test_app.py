from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from spatial_app.enrichment import MIN_CELLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/merfish-638850'
pytestmark = pytest.mark.skipif(not (DATA/'manifest.json').exists(), reason='prepared MERFISH dataset not available')


@pytest.fixture
def manifest():
    return json.loads((DATA/'manifest.json').read_text())


def finish(app, manifest):
    """Advance the fragment timer until every batch of every section is scored."""
    batches = {b for s in manifest['slices'] for b in s['batches']}
    for _ in range(len(batches)+5):  # one batch per rerun
        if batches <= app.session_state['done']:
            return
        app.run()
        assert not app.exception
    raise AssertionError('scoring did not finish')


def test_sixteen_sections_score_and_fill_both_tables(manifest):
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=180).run()
    assert not app.exception
    assert manifest['n_slices'] == 16 and len(app.session_state['scores']) == manifest['n_cells']
    header = next(m.value for m in app.markdown if 'coronal sections' in m.value)
    assert all(s['id'] in header for s in manifest['slices'])
    z = [s['z_mm'] for s in manifest['slices']]
    assert z == sorted(z, reverse=True)  # anterior (olfactory bulb) first
    assert app.session_state['job']['query'].universe_size == manifest['n_genes'] == 500
    app.run()
    finish(app, manifest)
    assert np.isfinite(app.session_state['scores']).all()
    cell_table, region_table = app.dataframe[0].value, app.dataframe[1].value
    cells = pd.read_parquet(DATA/'cells.parquet')
    for table, column in [(cell_table, 'cell_subclass'), (region_table, 'region_structure')]:
        counts = cells[column].value_counts()  # both hemispheres are annotated alike, so every cell is profiled
        assert (table['Cells'] >= MIN_CELLS).all()
        assert table.set_index(table.columns[0])['Cells'].to_dict() == counts[counts >= MIN_CELLS].to_dict()
    assert cell_table.columns[:2].tolist() == ['Cell subclass', 'Cell class']
    assert region_table.columns[:2].tolist() == ['Brain structure', 'Description']
    # The default set (Tac2, Chat, Slc5a7, Chrnb3, Pou4f1) marks the medial habenula.
    assert app.session_state['job']['query'].target_symbols == ('CHAT', 'CHRNB3', 'POU4F1', 'SLC5A7', 'TAC2')
    used = next(e for e in app.expander if e.label.startswith('Genes used'))
    assert used.label == 'Genes used · 5 targets in a 500-gene universe'
    assert used.markdown[0].value == '**Targets:** Chat, Chrnb3, Pou4f1, Slc5a7, Tac2'
    assert 'MH' in region_table.iloc[:5, 0].tolist()


def test_supertype_and_substructure_levels_and_colour_modes(manifest):
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=180).run()
    app.selectbox(key='cell_level').set_value('cell_supertype').run()
    app.selectbox(key='region_level').set_value('region_substructure').run()
    assert not app.exception
    assert app.dataframe[0].value.columns[:2].tolist() == ['Cell supertype', 'Cell subclass']
    assert app.dataframe[1].value.columns[0] == 'Brain substructure'
    for mode in ['Cell type', 'Region', 'Cell type AUROC', 'Region AUROC']:
        app.radio(key='colour_mode').set_value(mode).run()
        assert not app.exception

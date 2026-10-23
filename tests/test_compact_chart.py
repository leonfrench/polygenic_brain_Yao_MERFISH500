import base64
import numpy as np
import pandas as pd
from spatial_app.chart import geometry_payload, score_patch, label_codes
from spatial_app.layout import slice_grid, quantize, grid_columns

ANNOTATIONS = {
    'cell_subclass': dict(kind='cell_type', title='Cell subclass', labels={'A': '#FF0000', 'B': '#00FF00'}, descriptions={}),
    'region': dict(kind='region', title='Finest region', labels={'MOB': '#9AD2BD'}, descriptions={'MOB': 'Main olfactory bulb'}),
}


def decode(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


def toy_cells(slices=('T1', 'T2'), per=3):
    rows = [dict(slice=s, rx=100.0+10*i, ry=200.0+20*i+5*k, cell_subclass='AB'[i % 2], region='MOB')
            for k, s in enumerate(slices) for i in range(per)]
    return pd.DataFrame(rows)


def test_patch_roundtrip_including_query_reset_and_unchanged_nans():
    before = np.array([np.nan, .2, .7, np.nan])
    after = np.array([.9, np.nan, .7, np.nan])
    patch = score_patch(before, after)
    ids = decode(patch['indices'], '<u4')
    assert ids.tolist() == [0, 1]
    result = before.copy()
    result[ids] = decode(patch['values'], '<f8')
    np.testing.assert_equal(result, after)
    initial = score_patch(None, after)
    result = np.full(4, np.nan)
    result[decode(initial['indices'], '<u4')] = decode(initial['values'], '<f8')
    np.testing.assert_equal(result, after)


def test_twelve_slices_form_three_rows_of_four_with_a_shared_frame():
    slices = [f'T{i:03d}' for i in range(12)]
    cells = toy_cells(slices)
    px, py, panels, width, height = slice_grid(cells, slices)
    assert len(panels) == 12 and width == 4*panels[0]['w'] and height == 3*panels[0]['h']
    assert {p['w'] for p in panels} == {panels[0]['w']} and {p['h'] for p in panels} == {panels[0]['h']}
    np.testing.assert_allclose([(p['x']/p['w'], p['y']/p['h']) for p in panels[:6]], [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)])
    # Each cell stays inside its own panel, below the title band, at its registered offset.
    for k, p in enumerate(panels):
        rows = cells.slice.eq(slices[k]).to_numpy()
        assert (px[rows] > p['x']).all() and (px[rows] < p['x']+p['w']).all()
        assert (py[rows] > p['y']+p['header']).all() and (py[rows] < p['y']+p['h']).all()
    first, second = cells.slice.eq(slices[0]).to_numpy(), cells.slice.eq(slices[1]).to_numpy()
    np.testing.assert_allclose(px[second]-px[first], panels[0]['w'])
    np.testing.assert_allclose(py[second]-py[first], cells.ry[second].to_numpy()-cells.ry[first].to_numpy())
    qx, qy, step = quantize(px*30, py*30, width*30, height*30)
    assert qx.dtype == np.uint16 and step >= 1
    np.testing.assert_allclose(qx*step, px*30, atol=step)


def test_sixteen_slices_form_two_rows_of_eight():
    assert [grid_columns(n) for n in (1, 3, 4, 5, 12, 13, 16)] == [1, 3, 4, 4, 4, 8, 8]
    slices = [f'T{i:03d}' for i in range(16)]
    px, py, panels, width, height = slice_grid(toy_cells(slices), slices)
    np.testing.assert_allclose([(p['x']/p['w'], p['y']/p['h']) for p in panels[7:9]], [(7, 0), (0, 1)])
    assert width == 8*panels[0]['w'] and height == 2*panels[0]['h']
    # Smaller panels get a proportionally taller title band for the fixed-size label.
    assert panels[0]['header'] > slice_grid(toy_cells(slices[:12]), slices[:12])[2][0]['header']


def test_geometry_keeps_cell_order_labels_and_colours():
    cells = toy_cells()
    g = geometry_payload(cells, ['T1', 'T2'], ANNOTATIONS)
    assert g['n'] == 6 and g['layout'] == 'T1|T2' and [p['label'] for p in g['panels']] == ['T1', 'T2']
    level = g['levels']['cell_subclass']
    codes = decode(level['codes'], level['dtype'])
    assert [level['labels'][c] for c in codes] == cells.cell_subclass.tolist()
    assert level['colors'] == ['#FF0000', '#00FF00']
    assert g['levels']['region']['descriptions'] == ['Main olfactory bulb']
    assert decode(g['slice'], '<u1').tolist() == [0, 0, 0, 1, 1, 1]
    assert len(decode(g['px'], '<u2')) == 6
    assert (g['um_per_unit'], g['noun']) == (0.5, 'slices')  # Stereo-seq spot units
    merfish = geometry_payload(cells, ['T1', 'T2'], ANNOTATIONS, um_per_unit=1, noun='sections')
    assert (merfish['um_per_unit'], merfish['noun'], merfish['dot']) == (1.0, 'sections', g['dot'])



def test_categorical_labels_give_the_same_sorted_labels_and_codes_as_strings():
    values = pd.Series(['b', 'a', 'c', 'a', 'b'])
    expected = label_codes(values)
    # Categories in file order, with one absent from these cells.
    categorical = values.astype(pd.CategoricalDtype(['c', 'zz', 'b', 'a']))
    labels, codes = label_codes(categorical)
    assert labels.tolist() == expected[0].tolist() == ['a', 'b', 'c']
    np.testing.assert_array_equal(codes, expected[1])

def test_delayed_ack_sends_only_new_scores_and_remount_recovers(monkeypatch):
    import spatial_app.chart as chart
    session = {}
    sent = []
    monkeypatch.setattr(chart.st, 'session_state', session)
    monkeypatch.setattr(chart, '_component', lambda **kwargs: sent.append(kwargs))
    cells = toy_cells(('T1',), per=4)
    args = dict(slices=['T1'], annotations=ANNOTATIONS, version='slices-1', mode='Gene-set AUROC', cell_level='cell_subclass',
                region_level='region', cell_aurocs={'A': .6, 'B': np.nan}, region_aurocs={}, filename='test')
    # Establish the browser before progressive score delivery.
    session['cell_plot'] = dict(client='browser', version=None, seq=-1)
    chart.cell_chart(cells, np.array([.1, np.nan, np.nan, np.nan]), **args)
    assert sent[-1]['cell_aurocs'] == {'A': .6} and sent[-1]['hover'] is None  # the default tooltip
    first = sent[-1]['packet']['seq']
    chart.cell_chart(cells, np.array([.1, .2, np.nan, np.nan]), **args)
    session['cell_plot'] = dict(client='browser', version='slices-1', seq=first)
    chart.cell_chart(cells, np.array([.1, .2, .3, np.nan]), **args)
    assert sent[-1]['geometry'] is None
    assert not sent[-1]['packet']['reset']
    assert decode(sent[-1]['packet']['indices'], '<u4').tolist() == [1, 2]
    # A replaced iframe gets geometry and a full numeric snapshot.
    session['cell_plot'] = dict(client='new-browser', version=None, seq=-1)
    chart.cell_chart(cells, np.array([.1, .2, .3, np.nan]), **args)
    assert sent[-1]['geometry'] is not None
    assert sent[-1]['packet']['reset']
    assert decode(sent[-1]['packet']['indices'], '<u4').tolist() == [0, 1, 2]
    hover = dict(levels=['cell_subclass', 'region'], slice_with='region')
    chart.cell_chart(cells, np.array([.1, .2, .3, np.nan]), **args, hover=hover)
    assert sent[-1]['hover'] == hover

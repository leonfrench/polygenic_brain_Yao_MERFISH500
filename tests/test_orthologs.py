from spatial_app.orthologs import map_input, load_ortholog_map
from spatial_app.scoring import resolve_query


def test_mouse_input_passes_through_and_matches_case_insensitively():
    targets, missing = map_input('Sox2, Pax6; Sox2 NOT_A_GENE', 'Mouse')
    assert targets == ['NOT_A_GENE', 'Pax6', 'Sox2'] and missing == []
    query = resolve_query(' '.join(targets), '', ['Sox2', 'Pax6', 'Actb', 'Gapdh'])
    assert query.target_symbols == ('PAX6', 'SOX2') and query.unknown_target == ('NOT_A_GENE',)


def test_human_and_macaque_symbols_map_to_mouse_orthologs():
    assert map_input('SOX2 sox2 PAX6', 'Human') == (['Pax6', 'Sox2'], [])
    assert map_input('NOT_A_GENE MBP', 'Human') == (['Mbp'], ['NOT_A_GENE'])
    assert map_input('', 'Human') == ([], [])
    to_human, human_to_mouse = load_ortholog_map()
    source, humans = next((s, h) for s, h in to_human['Rhesus Macaque'].items() if all(x.upper() in human_to_mouse for x in h))
    expected = sorted({m for h in humans for m in human_to_mouse[h.upper()]})
    assert map_input(f'{source} {source}', 'Rhesus Macaque') == (expected, [])
    # One-to-many human -> mouse orthologs are all retained.
    human, mice = next((h, m) for h, m in human_to_mouse.items() if len(m) > 1)
    assert set(map_input(human, 'Human')[0]) == mice

import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control
from spatial_app.enrichment import specificity_z, rank_profiles, specificity_table, format_p_value


def toy_counts(seed=3, profiles=6, genes=300):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(.5, 20, genes), (profiles, genes)).astype(float)
    counts[:, :5] = 0          # undetected genes tie at z = 0
    counts[1, 10:20] *= 8      # a block of genes specific to profile 1
    return counts


def test_z_scores_are_log_cpm_standardized_per_gene_and_skip_small_profiles():
    counts = toy_counts()
    cells = np.array([50, 50, 5, 50, 50, 50])
    keep, z = specificity_z(counts, cells, min_cells=20)
    assert keep.tolist() == [True, True, False, True, True, True]
    x = np.log2(counts[keep]/counts[keep].sum(axis=1, keepdims=True)*1e6 + 1)
    varies = x.std(axis=0) > 0
    expected = (x[:, varies] - x[:, varies].mean(axis=0))/x[:, varies].std(axis=0)
    np.testing.assert_allclose(z[:, varies], expected, rtol=1e-5, atol=1e-5)
    assert (z[:, ~varies] == 0).all() and not varies[:5].any()


def test_auroc_and_p_values_match_mannwhitneyu_per_profile_with_ties():
    keep, z = specificity_z(toy_counts(), np.full(6, 50))
    ranks, ties = rank_profiles(z)
    target = np.r_[2, 3, 10, 11, 12, 13, 14, 40]  # includes two all-zero genes
    labels = [f'P{i}' for i in range(len(z))]
    table = specificity_table(ranks, ties, target, labels, np.full(len(z), 50), 'Profile').set_index('Profile')
    others = np.setdiff1d(np.arange(z.shape[1]), target)
    ps = []
    for i, label in enumerate(labels):
        test = mannwhitneyu(z[i, target].astype(float), z[i, others].astype(float), alternative='two-sided', method='asymptotic')
        np.testing.assert_allclose(table.loc[label, 'AUROC'], test.statistic/(len(target)*len(others)), rtol=1e-6)
        np.testing.assert_allclose(table.loc[label, 'p-value'], test.pvalue, rtol=1e-6)
        ps.append(test.pvalue)
    np.testing.assert_allclose(table.loc[labels, 'pFDR'], false_discovery_control(ps), rtol=1e-6)
    assert table.index[0] == 'P1'  # the profile where the target block is specific


def test_custom_universe_reranks_within_the_background():
    keep, z = specificity_z(toy_counts(), np.full(6, 50))
    universe = np.r_[10:20, 100:160]
    ranks, ties = rank_profiles(z[:, universe])
    target = np.r_[10, 11, 12]  # positions 0..2 of the universe
    table = specificity_table(ranks, ties, [0, 1, 2], list('ABCDEF'), np.full(6, 50), 'P').set_index('P')
    others = np.r_[13:20, 100:160]
    for i, label in enumerate('ABCDEF'):
        test = mannwhitneyu(z[i, target].astype(float), z[i, others].astype(float), method='asymptotic')
        np.testing.assert_allclose(table.loc[label, 'AUROC'], test.statistic/(3*len(others)), rtol=1e-6)


def test_degenerate_inputs():
    ranks, ties = rank_profiles(np.zeros((2, 10), dtype=np.float32))  # every gene tied
    table = specificity_table(ranks, ties, [0, 1], ['A', 'B'], [30, 30], 'P')
    assert (table.AUROC == .5).all() and (table['p-value'] == 1).all()
    for target in ([], list(range(10))):
        try:
            specificity_table(ranks, ties, target, ['A', 'B'], [30, 30], 'P')
        except ValueError:
            pass
        else:
            raise AssertionError('expected an empty or full target set to be rejected')
    # A p-value that underflows to 0 is reported at the smallest positive float.
    z = np.tile(np.arange(4000, dtype=np.float32), (2, 1)); z[1] = z[1][::-1]
    ranks, ties = rank_profiles(z)
    table = specificity_table(ranks, ties, np.arange(3000, 4000), ['up', 'down'], [30, 30], 'P').set_index('P')
    assert table.loc['up', 'AUROC'] == 1 and table.loc['up', 'p-value'] == np.nextafter(0., 1.)


def test_p_values_use_standard_notation_down_to_one_in_ten_thousand():
    cases = {1.0: '1', 0.5: '0.5', 0.037184: '0.0372', 0.00123456: '0.00123', 1e-4: '0.0001',
             9.99e-5: '9.99e-5', 4.287e-6: '4.29e-6', 1.2e-30: '1.20e-30',
             np.nextafter(0., 1.): '4.94e-324', np.nan: ''}
    for value, text in cases.items():
        assert format_p_value(value) == text, (value, format_p_value(value))

import numpy as np
import pytest
from scipy.stats import rankdata
from spatial_app.scoring import auroc_vectorized, resolve_query, genes_used
from spatial_app.view import score_limits
from scripts.rank_io import load_shard, auroc
from pathlib import Path
import json
import pandas as pd


def pack(x):
    indices=[]; data=[]; ptr=[0]; zero=[]
    for row in x:
        idx=np.flatnonzero(row)
        indices.extend(idx); data.extend(2*rankdata(row)[idx]); ptr.append(len(indices))
        nzero=np.count_nonzero(row==0); zero.append(nzero+1 if nzero else 0)
    return {'cell_index':np.arange(len(x)), 'gene_index':np.array(indices,dtype=np.uint16),
            'indptr':np.array(ptr,dtype=np.uint32),'rank2':np.array(data,dtype=np.uint16),
            'zero_rank2':np.array(zero,dtype=np.uint16)}


def pairwise(x,target,universe):
    target=np.intersect1d(target,universe)
    negatives=np.setdiff1d(universe,target)
    if not len(target) or not len(negatives): return np.full(len(x),np.nan)
    return np.array([np.mean((r[target,None]>r[negatives])+0.5*(r[target,None]==r[negatives])) for r in x])


@pytest.mark.parametrize('universe',[None,[0,1,3],[0,2],[0,1,2,3,4],[],[2]])
@pytest.mark.parametrize('target',[[0],[1,3],[0,0,1],[],list(range(5))])
def test_ties_and_subset_universes(universe,target):
    x=np.array([[0,0,0,0,0],[1,1,2,3,3],[0,2,0,2,5],[9,0,0,0,1],[1,2,3,4,5]])
    expected=pairwise(x,target,np.arange(5) if universe is None else universe)
    actual=auroc_vectorized(pack(x),target,5,universe)
    np.testing.assert_allclose(actual,expected,atol=1e-15,equal_nan=True)


def test_randomized_against_pairwise_and_subset_ranking():
    rng=np.random.default_rng(18)
    for _ in range(30):
        x=rng.integers(0,9,size=(19,83)); x[rng.random(x.shape)<0.8]=0
        target=rng.choice(83,13,replace=False)
        universe=rng.choice(83,43,replace=False)
        actual=auroc_vectorized(pack(x),target,83,universe)
        expected=pairwise(x,target,universe)
        np.testing.assert_allclose(actual,expected,atol=1e-15)
        chosen=np.isin(universe,target); k=chosen.sum()
        reranked=rankdata(x[:,universe],axis=1,method='average')
        np.testing.assert_allclose(actual,(reranked[:,chosen].sum(axis=1)-k*(k+1)/2)/(k*(len(universe)-k)))


def test_real_shard_full_and_custom_background():
    root=Path(__file__).resolve().parents[1]/'data/merfish-638850'
    if not (root/'manifest.json').exists():
        pytest.skip('prepared dataset not available')
    manifest=json.loads((root/'manifest.json').read_text())
    z=load_shard(root/manifest['slices'][0]['batches'][0]); g=manifest['n_genes']
    selected=[10,120,250,499]
    np.testing.assert_allclose(auroc_vectorized(z,selected,g),auroc(z,selected,g))
    universe=np.r_[selected,np.arange(100,400)]
    universe=np.unique(universe)
    # Decode a few rows independently and compare a fresh rankdata calculation.
    dense=np.broadcast_to(z['zero_rank2'][:10,None],(10,g)).copy().astype(float)
    for row in range(10):
        a,b=z['indptr'][row:row+2]; dense[row,z['gene_index'][a:b]]=z['rank2'][a:b]
    expected=pairwise(dense,selected,universe)
    np.testing.assert_allclose(auroc_vectorized(z,selected,g,universe)[:10],expected)


def test_gene_matching_and_errors():
    symbols=['SOX2','PAX6','ACTB','GAPDH']
    q=resolve_query('sox2, SOX2; PAX6 UNKNOWN','SOX2 ACTB GAPDH missing',symbols)
    assert q.target==(0,) and q.universe_size==3
    assert q.unknown_target==('UNKNOWN',) and q.unknown_background==('MISSING',)
    assert q.outside_background==('PAX6',)
    for target,background in [('', ''),('unknown',''),('SOX2','SOX2'),('SOX2','ACTB'),('SOX2 PAX6 ACTB GAPDH','')]:
        with pytest.raises(ValueError): resolve_query(target,background,symbols)
    assert score_limits(np.array([np.nan,0.3,0.7]))==(0.3,0.7)
    assert score_limits(np.array([np.nan])) is None


def test_genes_used_lists_the_universe_with_targets_first():
    genes=pd.DataFrame({'symbol':['Sox2','Pax6','Actb','Gapdh'],'name':['a','b','c','d']})
    full=genes_used(genes,resolve_query('pax6 sox2','',genes.symbol),['symbol','name'])
    assert full.values.tolist()==[['target','Pax6','b'],['target','Sox2','a'],['background','Actb','c'],['background','Gapdh','d']]
    # A custom background is the whole universe: genes outside it are not listed, nor are targets outside it.
    custom=genes_used(genes,resolve_query('SOX2 PAX6','SOX2 GAPDH',genes.symbol))
    assert custom.columns.tolist()==['role','symbol'] and custom.values.tolist()==[['target','Sox2'],['background','Gapdh']]

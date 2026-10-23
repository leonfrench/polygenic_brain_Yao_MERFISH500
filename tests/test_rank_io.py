import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.stats import rankdata
from rank_io import auroc, spatial_paths


def run():
    # Empty rows, no-zero rows, positive and zero ties, singleton sets,
    # duplicates, empty/full sets, and independently enumerated pairwise AUC.
    x=np.array([[0,0,0,0,0],[1,1,2,3,3],[0,2,0,2,5],[9,0,0,0,1]])
    indices=[]; ranks=[]; ptr=[0]; zero=[]
    for row in x:
        idx=np.flatnonzero(row); indices.extend(idx)
        ranks.extend(2*rankdata(row)[idx]); ptr.append(len(indices))
        z=np.count_nonzero(row==0); zero.append(z+1 if z else 0)
    shard=dict(cell_index=np.arange(len(x)),gene_index=np.array(indices,dtype=np.uint16),
        rank2=np.array(ranks,dtype=np.uint16),indptr=np.array(ptr,dtype=np.uint32),
        zero_rank2=np.array(zero,dtype=np.uint16))
    for selected in [[0],[2],[0,1],[0,1,1],[1,2,3,4]]:
        chosen=np.unique(selected); other=np.setdiff1d(np.arange(5),chosen)
        expected=np.array([np.mean((r[chosen,None]>r[other])+0.5*(r[chosen,None]==r[other])) for r in x])
        np.testing.assert_allclose(auroc(shard,selected,5),expected,atol=1e-15)
    assert np.isnan(auroc(shard,[],5)).all()
    assert np.isnan(auroc(shard,range(5),5)).all()
    for invalid in [[-1],[5],[0.5]]:
        try: auroc(shard,invalid,5)
        except ValueError: pass
        else: raise AssertionError('invalid index accepted')
    manifest={'files':[{'path':'a','layout':'spatial','bbox':[0,0,2,2]},
                       {'path':'b','layout':'spatial','bbox':[4,4,5,5]}]}
    assert spatial_paths(manifest,[1,1,3,3])==['a']
    assert spatial_paths(manifest,[2,2,4,4])==['a','b']
    print('All rank reader edge-case tests passed')

if __name__=='__main__': run()

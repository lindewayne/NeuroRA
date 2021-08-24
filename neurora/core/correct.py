"""
Docs
"""


import numpy as np
import skimage.measure as skm


def fwe_correct(p, t):
    p *= (~np.isnan(p)).sum()
    p[p > t] = np.nan
    return p


def fdr_correct(p, t):
    idx = ~np.isnan(p)
    p[idx] = p[idx] * idx.sum() / (p[idx].argsort() + 1)
    p[p > t] = np.nan
    return p


def _cluster_correct(p, t0, t1, cfunc):
    p0 = (p < t0) * 1
    label = skm.label(p0, connectivity=1)
    nc = label.max()
    vic = (label.flatten()[:, None] == np.arange(1, nc+1)).sum(0)
    vcluster = np.r_[0, vic].take(label)
    
    permvox = np.zeros(1000, dtype=int)
    for i in range(1000):
        np.random.shuffle(p0.ravel())
        labeli = skm.label(p0, connectivity=1)
        nci = labeli.max()
        vici = (labeli.flatten()[:, None] == np.arange(1, nci+1)).sum(0)
        permvox[i] = vici.max()
    
    permvox.sort()
    tvox = np.quantile(permvox, 1 - t1, interpolation='nearest')
    pc = (1000 - permvox.searchsorted(vic, 'right')) / 1000 * nc
    pc = pc / cfunc(pc)
    
    p_correct = np.r_[0, pc].take(label)
    idx = (~np.isnan(p)) * (label != 0) * (p_correct < t0) * (vcluster >= tvox)
    p_correct[~idx] = np.nan
    
    return p_correct


def cluster_fwe_correct(p, t0, t1):
    return _cluster_correct(p, t0, t1, np.ones_like)


def cluster_fdr_correct(p, t0, t1):
    return _cluster_correct(p, t0, t1, lambda x: x.argsort()+1)
"""
Docs
"""


import numpy as np
from scipy.stats import ttest_1samp, ttest_rel, ttest_ind


def fz_transform(x):
    return .5 * np.log((1 + x) / (1 - x))


def permutation_test(v0, v1, it=1000):
    diff = abs(v0.mean(-1) - v1.mean(-1))
    idx = np.indices((it, v0.shape[-1]*2))[-1]
    idx = np.apply_along_axis(np.random.permutation, -1, idx)
    v = np.concatenate((v0, v1), -1)[..., idx]
    v = v.reshape(*v0.shape[:-1], it, 2, -1).mean(-1)
    return (np.diff(v, axis=-1)[..., 0] <= -diff[..., None]).sum(-1) / it


def stats_onegroup(corr, p=True, fisherz=True, perm=False):
    if corr.shape[0] < 6:
        raise ValueError()
    if p:
        corr = corr[..., 0]
    if fisherz:
        corr = fz_transform(corr)
    
    t, p = ttest_1samp(corr, 0, alternative='greater')
    if perm:
        if perm == True:
            perm = 1000
        p = permutation_test(corr, np.zeros_like(corr), it=perm)
    res = np.stack((t, p), -1)
    return res


def stats_relgroups(corr0, corr1, p=True, fisherz=True, perm=False):
    if corr0.shape != corr1.shape:
        raise ValueError()
    if corr0.shape[0] < 6:
        raise ValueError()
    if p:
        corr0 = corr0[..., 0]
        corr1 = corr1[..., 0]
    if fisherz:
        corr0 = fz_transform(corr0)
        corr1 = fz_transform(corr1)
    
    t, p = ttest_rel(corr0, corr1)
    if perm:
        if perm == True:
            perm = 1000
        p = permutation_test(corr0, corr1, it=perm)
    res = np.stack((t, p), -1)
    return res


def stats_indgroups(corr0, corr1, p=True, fisherz=True, perm=False):
    if corr0.shape != corr1.shape:
        raise ValueError()
    if corr0.shape[0] < 6:
        raise ValueError()
    if p:
        corr0 = corr0[..., 0]
        corr1 = corr1[..., 1]
    if fisherz:
        corr0 = fz_transform(corr0)
        corr1 = fz_transform(corr1)
    
    t, p = ttest_ind(corr0, corr1)
    if perm:
        if perm == True:
            perm = 1000
        p = permutation_test(corr0, corr1, it=perm)
    res = np.stack((t, p), -1)
    return res
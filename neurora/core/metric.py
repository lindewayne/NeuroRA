"""
Broad sense metric, functions of estimation between two data.
"""


import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from .decoding import svc_pipe


def _perm_corr(x, y, corr_func, dropp=False, it=1000):
    mp = corr_func(x, y)
    if dropp:
        mp = mp[0]
    elif it:
        ni = 1
        for i in range(it):
            xp = np.random.permutation(x)
            yp = np.random.permutation(y)
            rp = corr_func(xp, yp)[0]
            if rp > mp[0]:
                ni += 1
        p = ni / (it + 1)
        mp[1] = p
    return mp


def pearson(x, y, dropp=False, perm=False, **kwargs):
    if perm == True:
        perm = 1000
    return _perm_corr(x, y, pearsonr, dropp, perm)


def spearman(x, y, dropp=False, perm=False, **kwargs):
    if perm == True:
        perm = 1000
    return _perm_corr(x, y, spearmanr, dropp, perm)


def kendall(x, y, dropp=False, permutation=False, **kwargs):
    if perm == True:
        perm = 1000
    return _perm_corr(x, y, kendalltau, dropp, perm)


def euclidean(x, y, dropp=False, **kwargs):
    mp = np.linalg.norm(x - y)
    if not dropp:
        mp = mp, 0
    return mp


def cossim(x, y, dropp=False, **kwargs):
    sim = x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)
    mp = .5 + .5 * sim
    if not dropp:
        mp = mp, 0
    return mp


def noncorr(x, y, absolute=False, dropp=False, **kwargs):
    r = pearsonr(x, y)[0]
    if absolute:
        r = abs(r)
    mp = 1 - r
    if mp < 1e-15:
        mp = 0
    if not dropp:
        mp = mp, 0
    return mp


def mahalanobis(x, y, dropp=False, **kwargs):
    X = np.vstack((x, y)).T
    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
    mp = np.linalg.norm(X[:, 0] - X[:, 1])
    if not dropp:
        mp = mp, 0
    return mp


def decoding(x, y, **kwargs):
    data = np.concatenate((x, y))
    label = np.array([0]*len(x) + [1]*len(x))
    mp = svc_pipe((data, label), **kwargs)
    return mp
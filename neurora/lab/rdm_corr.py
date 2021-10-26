import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding
from numba import njit, pndindex, prange


def rdm_corr(x, y, s=1, metric='spearman'):
    
    # flatten upper triangle
    t0, t1 = np.triu_indices(x.shape[-1], 1)
    x = x[..., t0, t1]
    y = y[..., t0, t1]
    
    # broadcast
    mv = tuple(range(s))
    to = tuple(range(-1-s, -1))
    x = np.moveaxis(x, mv, to)
    y = np.moveaxis(y, mv, to)

    xbshape = y.shape[:-1-s]
    ybshape = x.shape[:-1-s]

    mv2 = tuple(range(len(xbshape)))
    to2 = tuple(range(len(ybshape), len(xbshape+ybshape)))
    x = np.moveaxis(np.broadcast_to(x, xbshape+x.shape), mv2, to2)
    y = np.broadcast_to(y, ybshape+y.shape)
    x = np.moveaxis(x, to, mv)
    y = np.moveaxis(y, to, mv)
    
    # cal
    mf = {
        'spearman' : pwspearmanr,
        'pearson' : pwpearsonr,
        'euclidean' : pweuclidean,
        'cosine' : pwcossim
    }.get(metric, pwspearmanr)
    
    r = mf(x, y)
    return r


# pair wise metric function

@njit(parallel=True)
def pwpearsonr(x, y):
    shape = x.shape[:-1]
    r = np.empty(shape)
    for i in pndindex(shape):
        r[i] = np.corrcoef(x[i], y[i])[0, 1]
    return r


@njit
def rank(arr):
    sorter = np.argsort(arr)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.hstack((np.array([True]), arr[1:] != arr[:-1]))
    dense = obs.cumsum()[inv]
    count = np.hstack((np.nonzero(obs)[0], np.array([len(obs)])))
    return .5 * (count[dense] + count[dense - 1] + 1)


@njit(parallel=True)
def pwspearmanr(x, y):
    shape = x.shape[:-1]
    r = np.empty(shape)
    for i in pndindex(shape):
        if np.isnan(x[i]).any() or np.isnan(y[i]).any():
            r[i] = np.nan
        else:
            r[i] = np.corrcoef(rank(x[i]), rank(y[i]))[0, 1]
    return r


@njit(parallel=True)
def pweuclidean(x, y):
    shape = x.shape[:-1]
    d = np.empty(shape)
    for i in pndindex(shape):
        if np.isnan(x[i]).any() or np.isnan(y[i]).any():
            d[i] = np.nan
        else:
            d[i] = np.linalg.norm(x[i] - y[i])
    return d


@njit(parallel=True)
def pwcossim(x, y):
    shape = x.shape[:-1]
    s = np.empty(shape)
    for i in pndindex(shape):
        if np.isnan(x[i]).any() or np.isnan(y[i]).any():
            s[i] = np.nan
        else:
            s[i] = np.dot(x[i], y[i]) / np.linalg.norm(x[i]) / np.linalg.norm(y[i])
    return s
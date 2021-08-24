"""
Core function in calculation of RDM, ISC, NPS, STPS
"""


import numpy as np
from math import factorial
from .metric import (
    pearson, 
    spearman,
    kendall,
    cossim,
    noncorr,
    euclidean,
    mahalanobis,
    decoding
)
from .datahandle import minmaxscale
from tqdm import tqdm


def cross_rel(
    data,
    metric='spearman',
    caxis=0,
    iaxes=-1,
    out=2,
    verbose=True,
    **kwargs
):
    
    """
    Parameters
    ----------
    data : ndarray
    metric : str, default 'spearman'
        Relation metric between two array (or ndarray), can be any of
        - pearson : pearson r
        - spearman : spearman r
        - kendall : kendall tau
        - cosine : cosine similarity
        - correlation : 1 - pearson r, or 1 - abs(pearson r)
        - euclidean : euclidean distance
        - mahalanobis : mahalanobis distance
        - decoding : svc-base time decoding
    caxis : int, default 0
        Calculate metric between every two subdata across specific axis.
    iaxes : int or tuple, default -1
        Specify subdata space (axes), a base unit for metric calculation.
    out : int or tuple, default 2
        Output shape of metric function.
    verbose : bool, default True
        If True, show progress in iteration shape level.
        Iteration shape = data shape remove caxis and iaxes.
    **kwarge :
        Any key word argument send to metric function.
    
    Returns
    -------
    rct : ndarray
        Result in shape [*iter-shape, cross-shape, cross-shape, *output-shape]
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(5, 3, 4, 10)
    >>> result = cross_rel(data, caixs=0, iaxes=-1, out=2)
    >>> result.shape
    (3, 4, 5, 5, 2)
    
    >>> result = cross_rel(data, caixs=0, iaxes=-1, out=1)
    >>> result.shape
    (3, 4, 5, 5)
    
    >>> result = cross_rel(data, caixs=1, iaxes=-1, out=2)
    >>> result.shape
    (5, 4, 3, 3, 2)
    """
    
    # cross-dim to head, input-dim to tail, iteration-dim in mid
    if isinstance(iaxes, int):
        iaxes = (iaxes, )
    axes_pre = (caxis, *iaxes)
    axes_post = (0,) + tuple(range(-len(iaxes), 0))
    data = np.moveaxis(data, axes_pre, axes_post)
    
    rel_func = {
        'pearson' : pearson, 
        'spearman' : spearman,
        'kendall' : kendall,
        'cosine' : cossim,
        'correlation' : noncorr,
        'euclidean' : euclidean,
        'mahalanobis' : mahalanobis,
        'decoding' : decoding
    }[metric]
    
    # nan indicate
    nanind = np.isnan(data).any(axes_post[1:])
    
    # init cross table (diagonal matrix)
    ncross = data.shape[0]
    singlet = np.eye(ncross)
    if metric in ['correlation', 'euclidean', 'mahalanobis']:
        np.fill_diagonal(singlet, 0)
    elif metric == 'decoding':
        np.fill_diagonal(singlet, np.nan)
    
    # broadcast cross table to result container
    nshape = data.shape[1:-len(iaxes)]
    out = (out, ) if isinstance(out, int) else out
    # -> [*outshape, *nshape, ncross, ncross]
    rct = np.broadcast_to(singlet, (*out, *nshape, *singlet.shape)).copy()
    # -> [*nshape, ncross, ncross, *outshape]
    rct = np.moveaxis(rct, tuple(range(len(out))), tuple(range(-len(out), 0)))
    
    # core calculation loop
    outloop = np.ndindex(nshape)
    if verbose: # show progress
        outloop = tqdm(outloop, total=np.prod(nshape), ncols=100)
    for idx in outloop:
        for i, j in zip(*np.triu_indices(ncross, 1)): # just cal upper triangle 
            if nanind[([i, j], *idx)].any():
                r, p  = np.nan, np.nan
            else:
                mp = rel_func(data[(i, *idx)], data[(j, *idx)], **kwargs)
            rct[(*idx, i, j)] = mp
            rct[(*idx, j, i)] = mp
    
    if metric in ['euclidean', 'mahalanobis']:
        rct[..., 0] = minmaxscale(rct[..., 0], (-2, -1))
    
    if rct.shape[-1] == 1:
        rct = rct[..., 0]
    
    return rct
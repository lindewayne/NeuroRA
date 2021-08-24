"""
Docs
"""


import numpy as np
from .datahandle import minmaxscale, rolling
from .decoding import _dc_handle
from .cross import cross_rel
from .metric import pearson, spearman, kendall, euclidean, cossim
from tqdm import tqdm


def rdm_bhv(
    data,
    sub_opt=1,
    metric='correlation',
    absolute=False
):
    
    """
    data : ndarray
        [n_cons, n_subs, n_trials]
    """
    
    if data.ndim != 3:
        raise ValueError()
    
    # data handle
    data = data.mean(-1)
    
    if sub_opt:
        rdm = data.T[:, :, None] - data.T[:, None, :]
        if absolute:
            rdm = abs(rdm)
        rdm = minmaxscale(rdm, (-2, -1))
    else:
        rdm = cross_rel(data, metric=metric, out=1, dropp=True, absolute=absolute)
    
    return rdm


def rdm_eeg_bydecoding(
    data,
    sub_opt=1,
    time_win=5,
    time_step=5,
    win_avg=True,
    navg=5,
    split=5,
    repeat=2,
    norm=False,
    verbose=True
):
    
    """
    data : ndarray [n_cons, n_subs, n_trials, n_chls, n_ts]
    """
    # check
    if data.ndim != 5:
        raise ValueError()
    
    # data handle (ncons, nsubs, ntrials, nchls*time_win, nwins)
    data = _dc_handle(data, time_win, time_step, win_avg)
    
    # cal rdm
    params = dict(
        iaxes=(-3, -2, -1),
        out=data.shape[-1],
        navg=navg,
        split=split,
        repeat=repeat,
        norm=norm,
        ct=False,
        verbose=verbose
    )
    rdm = cross_rel(data, metric='decoding', **params)
    
    # reshape result
    rdm = np.moveaxis(rdm, -1, 1)
    
    if not sub_opt:
        rdm = rdm.mean(0)
    
    return rdm


def rdm_eeg(
    data,
    sub_opt=1,
    chl_opt=0,
    time_opt=0,
    time_win=5,
    time_step=5,
    metric='correlation',
    absolute=False,
    verbose=False
):
    
    """
    data : ndarray
        [n_cons, n_subs, n_trials, n_chls, n_ts]
    """
    
    if metric not in ['correlation', 'euclidean']:
        raise ValueError(f'Unknown dissimilarity metric: {metric}')
    
    # data handle
    data = data.mean(-3) # [n_cons, n_subs, n_chls, n_ts]
    if time_opt: # [n_cons, n_subs, n_chls, n_wins, n_ts(time_win)]
        data = rolling(data, time_win, time_step)
    if not chl_opt: # [..., n_chls*n_ts]
        data = np.moveaxis(data, 2, -2)
        data = data.reshape(*data.shape[:-2], -1)
    
    # cal rdm
    rdm = cross_rel(data, metric=metric, out=1, dropp=True, absolute=absolute,
                    verbose=verbose)
    
    if not sub_opt:
        rdm = rdm.mean(0)
    
    return rdm


def rdm_fmri(
    data,
    kernel=[3, 3, 3],
    stride=[1, 1, 1],
    sub_opt=1,
    metric='correlation',
    absolute=False,
    verbose=True
):
    
    """
    data : ndarray
        [n_cons, n_subs, nx, ny, nz]
    """
    
    if metric not in ['correlation', 'euclidean']:
        raise ValueError(f'Unknown dissimilarity metric: {metric}')
    
    # data handle
    # [n_cons, n_subs, n_x, n_y, n_z, kx, ky, kz]
    data = rolling(data, kernel, stride)
    # [n_cons, n_subs, n_x, n_y, n_z, kx*ky*kz]
    data = data.reshape(*data.shape[:-3], -1)
    
    # cal rdm
    rdm = cross_rel(data, metric=metric, out=1, dropp=True, absolute=absolute,
                    verbose=verbose)
    
    if not sub_opt:
        rdm = rdm.mean(0)
    
    return rdm


def rdm_fmri_roi(
    data,
    mask,
    sub_opt=1,
    metric='correlation',
    absolute=False,
    verbose=False
):
    
    """
    data : ndarray
        [n_cons, n_subs, nx, ny, nz]
    """
    
    if metric not in ['correlation', 'euclidean']:
        raise ValueError(f'Unknown dissimilarity metric: {metric}')
    
    # data handle
    x, y, z = np.where(mask > 0)
    data = data[..., x, y, z] # [n_cons, n_subs, n_voxs]
    
    # cal rdm
    rdm = cross_rel(data, metric=metric, out=1, dropp=True, absolute=absolute,
                    verbose=verbose)
    
    if not sub_opt:
        rdm = rdm.mean(0)
    
    return rdm


def rdm_corr(
    rdm0,
    rdm1,
    metric='spearman',
    rescale=False,
    perm=False,
    verbose=False
):
    
    """
    Calculate Similarities between two RDM(RDMs).
    
    Parameters
    ----------
    rdm0 : ndarray
        Shape can be :
        - [n_conditions, n_conditions]
        - [n_subjects, n_conditions, n_conditions]
        - [n_subjects, ..., n_conditions, n_conditions]
        See Notes. <Important!>
    rdm1 : ndarray
        Requirement as rdm0, should have same n_conditions with rdm0.
        See Notes. <Important!>
    metric : str, default 'spearman'
        RDM metric function, can be :
        - 'pearson' for pearson r, 
        - 'spearman' for spearman r,
        - 'kendall' for kendall tau,
        - 'euclidean' for euclidean distance
        - 'cosine' for cosine similarity
    rescale : bool, default False
        If True, min-max-scale within each RDM, ingore diagonal, and replace
        constant array with 1.
    permutation : bool or int, default False
        If int or True (will be set to 1000), conduct permutation test for p
        value when use pearson, spearman and kendall metrics, with iteration
        time = permutation.
    
    Returns
    -------
    corr : ndarray
        Similarities between rdm0 and rdm1.
        Shape of corr depends on shape of rdm0 and rdm1.
    
    See also
    --------
    rdm_corr_from_data, bhvRDM, eegRDM, fmriRDM
    
    Notes
    -----
    Matching requirement between rdm0 and rdm1 :
    > Same n_conditions square in last two dimension.
    > Regarding last two dimension as RDM element, the rest regard as shape of
      RDMs.
    > Shape of rdm0 and rdm1 should have sam 'head', for example :
      rdm0.shape = [n_subs, n_cons, n_cons],
      rdm1.shape = [n_subs, x, y, z, n_cons, n_cons].
      Returns(corr) shape will be [n_sub, x, y, z, 2], for rdm0 will broadcast
      to rdm1.
      Understanding broadcast : For each RDM (in shape [n_cons, n_cons]) in
      subspace [x, y, z] of sub0 of rdm1, we keeps the sam RDM 
      (in shape [n_cons, n_cons]) of sub0 of rdm0, for rdm corr calculate.
    > Some vaild shape pair example :
      - [n, n] & [s, c, t, n, n]
      - [x, y, n, n] & [x, y, z, n, n]
    
    Examples
    --------
    Example 1
    Fake some data for examples
    >>> import numpy as np
    >>> eeg = np.random.randn(3, 5, 100, 64, 100) # [con, sub, trl, chl, tp]
    >>> fmri = np.random.randn(3, 5, 20, 30, 20) # [con, sub, x, y, z]
    
    >>> erdm = rdm_eeg(eeg, chl_opt=0)
    >>> erdm.shape
    (5, 3, 3)
    >>> frdm = rdm_fmri(fmri)
    >>> frdm.shape
    (5, 18, 28, 18, 3, 3)
    
    >>> result = rdm_corr(erdm, frdm)
    >>> result.shape
    (5, 18, 28, 18, 2)
    
    Example 2
    Fake a RDM demo
    >>> demo = np.random.randn(3, 3)
    
    >>> erdm = rdm_eeg(eeg, time_opt=1)
    >>> erdm.shape
    (5, 20, 3, 3)
    
    >>> result = rdm_corr(demo, eeg)
    >>> result.shape
    (5, 20, 2)
    """
    
    # shape check
    if rdm0.ndim > rdm1.ndim: # let rdm0 be the smaller one
        rdm0, rdm1 = rdm1, rdm0    
    if rdm0.ndim < 2:
        raise ValueError('Not RDM data')
    if np.not_equal(*rdm0.shape[-2:]) or np.not_equal(*rdm1.shape[-2:]):
        raise ValueError('Not RDM data')
    
    # pair check
    if rdm0.shape[-2:] == rdm1.shape[-2:]: # rdm shape match
        if np.prod(rdm0.shape[:-2]) == 1: # rdm0 singlet
            rdm0 = rdm0.reshape(rdm0.shape[-2:])
        elif rdm0.shape[:-2] == rdm1.shape[:rdm0.ndim-2]:
            pass
        else:
            raise ValueError('Fail to broadcast one RDMs to another.')
    else:
        raise ValueError('RDM shape not match.')
    
    # metric check
    try:
        corr_func = {'pearson' : pearson, 
                     'spearman' : spearman,
                     'kendall' : kendall,
                     'euclidean' : euclidean,
                     'cosine' : cossim}[metric]
    except:
        raise ValueError(f'Unknown correlation metric: {metric}')
    
    # flatten upper triangle of rdms
    x, y = np.triu_indices(rdm0.shape[-1], 1)
    v0 = rdm0[..., x, y]
    v1 = rdm1[..., x, y]
    
    if rescale:
        v0 = minmaxscale(v0, -1)
        v1 = minmaxscale(v1, -1)
    
    # (bigger one's) shape of RDMs (with each RDM as an unit)
    nshape = v1.shape[:-1] 
    # init corr result container
    corr = np.zeros((*nshape, 2))
    
    loop = np.ndindex(nshape)
    if verbose:
        loop = tqdm(loop, total=np.prod(nshape), ncols=100)
    for idx1 in loop:
        idx0 = idx1[:v0.ndim-1]
        corr[idx1] = corr_func(v0[idx0], v1[idx1], perm=perm)
    
    return corr
"""
Docs
"""


import numpy as np
from .cross import cross_rel
from .datahandle import rolling


def stps(data, litem, lrf, verbose=True):
    
    """
    data : ndarray [n_subs, n_trials, ...]
    litem : ndarray [n_trials]
    lrf : ndarray [n_subs, n_trials]
    """
    
    ct = cross_rel(data, metric='pearson', caxis=1, out=1, dropp=True,
                   verbose=verbose) # [n_subs, ..., n_trials, n_trials]
    x, y = np.triu_indices(len(litem), 1)
    ct = ct[..., x, y] # [n_subs, ..., n_triu1]
    # bool index in shape [n_subs, n_triu1]
    rf = (lrf[..., None] + lrf[:, None, :])[:, x, y] # [n_subs, n_triu1]
    ri = (litem == litem[:, None])[x, y] # [n_triu1]
    i0 = np.broadcast_to(ri, rf.shape).copy() # [n_subs, n_triu1]
    i1 = ~i0
    i2 = rf == 0
    i3 = rf == 2
    i4 = i0 * i2
    i5 = i0 * i3
    i6 = i1 * i2
    i7 = i1 * i3
    iset = [i0, i1, i2, i3, i4, i5, i6, i7]
    for i in iset: # set False to nan (kick out from calculation)
        i[~i] = np.nan
    expdim = tuple(range(1, ct.ndim-1))
    res = np.concatenate(
        tuple(np.nanmean(ct*np.expand_dims(i, expdim), -1)[:, None, ...]
              for i in iset),
        axis=1
    )
    
    return res


def stps_eeg(
    data,
    label_item,
    label_rf,
    time_win=20,
    time_step=1,
    verbose=True
):
    
    """
    data : ndarray
        [n_subs, n_trials, n_chls, n_ts]
    """
    
    if data.ndim != 4:
        raise ValueError()
    
    # data handle
    # [n_subs, n_trials, n_chls, n_wins, n_ts(time_win)]
    data = rolling(data, time_win, time_step)
    
    return stps(data, label_item, label_rf, verbose)


def stps_fmri(
    data,
    label_item,
    label_rf,
    kernel=[3, 3, 3],
    stride=[1, 1, 1],
    verbose=True
):
    
    """
    data : ndarray
        [n_subs, n_trials, nx, ny, nz]
    """
    
    if data.ndim != 5:
        raise ValueError()
    
    # data handle
    # [n_ts, n_subs, n_x, n_y, n_z, kx, ky, kz]
    data = rolling(data, kernel, stride)
    # [n_ts, n_subs, n_x, n_y, n_z, kx*ky*kz]
    data = data.reshape(*data.shape[:-3], -1)
    
    return stps(data, label_item, label_rf, verbose)


def stps_fmri_roi(
    data,
    label_item,
    label_rf,
    mask,
    verbose=True
):
    
    """
    data : ndarray
        [n_subs, n_trials, nx, ny, nz]
    """
    
    if data.ndim !=5:
        raise ValueError()
    if mask.shape != data.shape[-3:]:
        raise ValueError()
    
    # data handle
    x, y, z = np.where(mask > 0)
    data = data[..., x, y, z] # [n_subs, n_trials, n_voxs]
    
    return stps(data, label_item, label_rf, verbose)
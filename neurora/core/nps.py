"""
Docs
"""


import numpy as np
from .datahandle import rolling
from .cross import cross_rel


def nps_eeg(
    data,
    time_win=5,
    time_step=5,
    sub_opt=1,
    verbose=False
):
    
    """
    data : ndarray
        [2, n_subs, n_trials, n_chls, n_ts]
    """
    
    if data.ndim !=5 or len(data) !=2:
        raise ValueError()
    
    # data handle
    # [2, n_subs, n_trials, n_chls, n_wins, n_ts(time_win)]
    data = rolling(data, time_win, time_step)
    # [2, n_subs, n_chls, n_wins, n_trials*time_win]
    data = np.moveaxis(data, 2, -2)
    data = data.reshape(*data.shape[:-2], -1)
    
    # cal nps
    nps = cross_rel(data, metric='pearson', verbose=verbose)[..., 0, 1, :]
    if not sub_opt:
        nps = nps.mean(0)
    
    return nps


def nps_fmri(
    data,
    kernel=[3, 3, 3],
    stride=[1, 1, 1],
    verbose=True
):
    
    """
    data : ndarray
        [2, n_subs, nx, ny, nz]
    """
    
    if data.ndim != 5 or len(data) != 2:
        raise ValueError()
    
    # data handle
    # [2, n_subs, n_x, n_y, n_z, kx, ky, kz]
    data = rolling(data, kernel, stride)
    # [2, n_subs, n_x, n_y, n_z, kx*ky*kz]
    data = data.reshape(*data.shape[:-3], -1)
    
    # cal nps
    nps = cross_rel(data, metric='pearson', verbose=verbose)[..., 0, 1, :]
    
    return nps


def nps_fmri_roi(data, mask, verbose=False):
    
    if data.ndim !=5 or len(data) != 2:
        raise ValueError()
    if mask.shape != data.shape[-3:]:
        raise ValueError()
    
    # data handle
    x, y, z = np.where(mask > 0)
    data = data[..., x, y, z] # [2, n_subs, n_voxs]
    
    # cal nps
    nps = cross_rel(data, metric='pearson', verbose=verbose)[..., 0, 1, :]
    
    return nps
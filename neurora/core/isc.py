"""
Docs
"""


import numpy as np
from .datahandle import rolling
from .cross import cross_rel


def isc_eeg(data, time_win=5, time_step=5, verbose=False):
    
    """
    data : ndarray
        [n_subs, n_chls, n_ts]
    """
    
    if data.ndim != 3:
        raise ValueError()
    
    # data handle
    # [n_subs, n_chls, n_wins, n_ts(time_win)]
    data = rolling(data, time_win, time_step)
    
    # cal isc
    # [n_chls, n_wins, n_subs, n_subs, 2]
    isc = cross_rel(data, metric='pearson', verbose=verbose)
    
    # reshape isc
    x, y = np.triu_indices(len(data), 1)
    isc = isc[..., x, y, :] # [n_chls, n_wins, n_up_triangle, 2]
    isc = np.moveaxis(isc, -2, 0) # [n_up_triangle, n_chls, n_wins, 2]
    
    return isc


def isc_fmri(data, kernel=[3, 3, 3], stride=[1, 1, 1], verbose=True):
    
    """
    data : ndarray
        [n_ts, n_subs, nx, ny, nz]
    """
    
    if data.ndim != 5:
        raise ValueError()
    
    # data handle
    # [n_ts, n_subs, n_x, n_y, n_z, kx, ky, kz]
    data = rolling(data, kernel, stride)
    # [n_ts, n_subs, n_x, n_y, n_z, kx*ky*kz]
    data = data.reshape(*data.shape[:-3], -1)
    
    # cal isc
    # [n_ts, n_x, n_y, n_z, n_subs, n_subs, 2]
    isc = cross_rel(data, metric='pearson', caxis=1, verbose=verbose)
    
    # reshape isc
    x, y = np.triu_indices(data.shape[1], 1)
    isc = isc[..., x, y, :] # [n_ts, n_x, n_y, n_z, n_up_triangle, 2]
    isc = np.moveaxis(isc, -2, 0) # [n_up_triangle, n_ts, n_x, n_y, n_z, 2]
    
    return isc


def isc_fmri_roi(data, mask, verbose=False):
    
    """
    data : ndarray
        [n_ts, n_subs, nx, ny, nz]
    """
    
    if data.ndim !=5 or len(data) != 2:
        raise ValueError()
    if mask.shape != data.shape[-3:]:
        raise ValueError()
    
    # data handle
    x, y, z = np.where(mask > 0)
    data = data[..., x, y, z] # [n_ts, n_subs, n_voxs]
    
    # cal isc
    # [n_ts, n_subs, n_subs, 2]
    isc = cross_rel(data, metric='pearson', caxis=1, verbose=verbose)
    
    # reshape isc
    x, y = np.triu_indices(data.shape[1], 1)
    isc = isc[..., x, y, :] # [n_ts, n_up_triangle, 2]
    
    return isc
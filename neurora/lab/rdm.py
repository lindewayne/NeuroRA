import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding
from numba import njit, pndindex


@njit(parallel=True)
def omp(arr):
    ax = arr.shape[:-2]
    cx = arr.shape[-2:-1]
    r = np.zeros(ax + cx + cx)
    for i in pndindex(ax):
        ri = 1 - np.corrcoef(arr[i])
        ri[ri<1e-15] = 0
        r[i] = ri
    return r

def rdm_fmri(data):
    
    '''
    Parameters
    ----------
    data : ndarray
        With shape [n_conditions, x, y, z]
    '''
    
    # data handle
    data = sliding(data, (3, 3, 3), (-3, -2, -1))
    data = data.reshape(*data.shape[:-3], -1)
    data = np.moveaxis(data, 0, -2)
    
    # cal rdm
    rdm = omp(data)
    
    return rdm


def rdm_eeg(data):
    
    """
    Parameters
    ----------
    data : ndarray
        [n_cons, n_subs, n_trials, n_chls, n_ts]
    """
    
    # data handle
    data = data.mean(-3) # [n_cons, n_subs, n_chls, n_ts]
    data = data.reshape(*data.shape[:-2], -1)
    data = np.moveaxis(data, 0, -2)
    
    # cal rdm
    rdm = omp(data)
    
    return rdm

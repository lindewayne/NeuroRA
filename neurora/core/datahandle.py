"""
Docs
"""


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding


np.seterr(divide='ignore', invalid='ignore')


def minmaxscale(data, axis=-1):
    ptp = data.ptp(axis, keepdims=True)
    m = data.min(axis, keepdims=True)
    res = (data - m) / ptp
    res[np.isnan(res)] = 1
    return res


def rolling(
    data,
    kernel=5,
    stride=1,
    pad=None,
    **kwargs
):
    
    """
    Slide last n dimention of data, with n-dim kernel and n-dim stride.
    
    Parameters
    ----------
    data : ndarray
    kernel : int or tuple, default 5
        Sliding kernel size.
    stride : int or tuple, default 1
        Sliding stride.
    pad : tuple or None
        Pad size. See numpy.pad .
    
    Returns
    -------
    data_tran : ndarray
    
    Notes
    -----
    Ensure lenth of kernel, stride, pad (if pass) exactly the same.
    
    Understanding 1d sliding
    [1 2 3]4 5 6 ->  1[2 3 4]5 6 -> 1 2[3 4 5]6 -> 1 2 3[4 5 6]
    So we get a 2d-array, with 1d kernel, and 1d window sliding
    [[1, 2, 3],
     [2, 3, 4],
     [3, 4, 5],
     [4, 5, 6]]
    
    Understanding 2d sliding
    [1 2]3      1[2 3]     1 2 3      1 2 3
    [4 5]6  ->  4[5 6] -> [4 5]6  ->  4[5 6]
     7 8 9      7 8 9     [7 8]9      7[8 9]
    So we get a 4d-array, with 2d kernel, and 2d window sliding
    [[[[1, 2],
       [4, 5]],
      [[2, 3],
       [5, 6]]],
       
     [[[4, 5],
       [7, 8]],
      [[5, 6],
       [8, 9]]]]
    
    Examples
    --------
    >>> import numpy as np
    
    1d rolling
    >>> data = np.random.randn(2, 3, 100)
    >>> rolling(data, 5, 1, None).shape
    (2, 3, 96, 5)
    >>> data = np.random.randn(2, 10), kernel 5, stride 2, pad None,
    >>> rolling(data, 5, 2, None).shape
    (2, 3, 5)
    >>> np.random.randn(2, 5)
    >>> rolling(data, (5,), (1,), ((2, 2),)).shape
    (2, 5, 5)
    
    As above, shape of last dim of result, equal to kernel.
    
    2d rolling
    >>> data = np.random.randn(2, 3, 100, 100),
    >>> rolling(data, (5, 5), (1, 1), None).shape
    (2, 3, 96, 96, 5, 5)
    >>> data = np.random.randn(2, 10, 10)
    >>> rolling(data, (5, 5), (2, 2), None).shape
    (2, 3, 3, 5, 5)
    
    As above, shape of last two dim of result, equal to kernel.
    
    And so forth.
    """
    
    # check
    if isinstance(kernel, int):
        kernel = (kernel,)
    if isinstance(stride, int):
        stride = (stride,)
    if len(kernel) != len(stride):
        raise ValueError()
    if pad and len(kernel) != len(pad):
        raise ValueError()
    if len(kernel) > 3:
        raise ValueError()
    
    dnd = data.ndim
    knd = len(kernel)
    
    # padding
    if pad and any(p != (0, 0) for p in pad):
        pw = [(0, 0)] * dnd
        pw[-knd:] = pad
        cv = kwargs.get('cv', 0)
        data = np.pad(data, pad_width=pw, constant_values=cv)
    axis = tuple(range(dnd-knd, dnd))
    
    # rolling with step 1
    data_slide = sliding(data, kernel, axis)
    
    # now stride
    ns = len(stride)
    nr = data.ndim - ns
    sidx = (
        *[slice(None)]*nr,
        *[slice(None, None, i) for i in stride],
        *[slice(None)]*ns
    )
    data_tran = data_slide[sidx]
    
    return data_tran


def smoothing(data, p=2):
    w = p * 2 + 1
    data_tran = rolling(data, w, 1, ((p, p),), cv=np.nan)
    data_tran = np.nanmean(data_tran, -1)
    return data_tran


def smoothing2d(data, p=2):
    w = p * 2 + 1
    data_tran = rolling(data, (w, w), (1, 1), ((p, p), (p, p)), cv=np.nan)
    data_tran = np.nanmean(data_tran, (-2, -1))
    return data_tran
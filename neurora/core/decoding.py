"""
Classification-based neural decoding.
As a classifier, support vector machine (SVM) is used.
"""


import numpy as np
from .datahandle import rolling, smoothing, smoothing2d
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _dc_check(data, label):
    
    "Data preprocessing (check) for decoding."
    
    cat = np.unique(label)
    if len(data) != len(label):
        raise ValueError(
            f"The number of epochs doesn't match the number of labels.")
    if not (label[..., None] == cat).sum(1).min(1).all():
        raise ValueError("Categories doesn't match between subjects.")
    return cat


def _dc_handle(data, time_win, time_step, win_avg):
    
    "Data preprocessing (time dimension transform) for decoding."
    
    if time_win > data.shape[-1]:
        raise ValueError("Time window overlength.")
    # [..., n_ts] -> [..., n_wins, time_win]
    data = rolling(data, time_win, time_step)
    if win_avg: # [..., n_wins, time_win] -> [..., n_wins]
        data = data.mean(-1)
    else: # [..., n_chls, n_wins, time_win] -> [..., n_chls*time_win, n_wins]
        data = np.moveaxis(data, -1, -2)
        data = data.reshape(*data.shape[:-3], -1, data.shape[-1])
    return data


def _dc_extract(data, label, navg):
    
    """
    Categories-balance (down sampling) and average-every-n-trials.
    
    data : ndarray [n_trials, ...]
    label : array [n_trials]
    """
    
    # undersampling index
    labidx = label[:, None] == np.unique(label)
    n = labidx.sum(0).min() // navg * navg
    bidx = np.concatenate(
        [np.random.choice(np.argwhere(cat).ravel(), n, replace=False)
         for cat in labidx.T]
    ) # [*n_idxs_of_cat0, *n_idxs_of_cat1, ..., *n_idxs_of_catn]
    
    # average-every-n-trials
    data_out = np.moveaxis(data[bidx], 0, -1)
    data_out = rolling(data_out, navg, navg).mean(-1)
    data_out = np.moveaxis(data_out, -1, 0)
    label_out = label[bidx][::navg]

    return data_out, label_out


def _pair_gener(x, y, x1, y1):
    
    "Train in (x, y) and test in (x1, y1)."
    
    yield x, x1, y, y1


def _skf_gener(n):
    
    "SKF split in (x, y), ignore (x1, y1)."
    
    def skf(x, y, x1, y1):
        for tid, did in StratifiedKFold(n_splits=n, shuffle=True).split(x, y):
            yield x[tid], x[did], y[tid], y[did]
            
    return skf


def _kog_gener(n):
    
    "Keep one group in (x, y) for test, ignore (x1, y1)."
    
    def kog(x, y, x1, y1):
        yield train_test_split(x, y, test_size=n, stratify=y)
    
    return kog


def svc_pipe(
    dataset,
    dataset1=None,
    navg=5,
    split=5,
    repeat=2,
    norm=True,
    ct=False,
    **kwargs
):
    
    """
    Machine learning pipeline for single subject dataset.
    
    Parameters
    ----------
    dataset : tuple
        A (data, label) set as feature-target ML dataset.
        Which :
        - data in shape [n_trials, ..., n_time_windows]
        - label in shape [n_trials]
    dataset1 : tuple or None, default None
        If not None, perform transfer decoding.
    navg : int, default 5
        Average in every 'navg' trials.
    split: int or float, default 5
        Train-test strategy.
        If < 1, keep one group hold out validation, be the test proportion.
        If > 1, cross validation, be the k (fold).
        Automatically set to 1 in transfer decoding.
    repeat : int, default 2
        Repeat whole pipeline 'repeat' times, get average.
    norm : bool, default True
        If True, normalize the data (channel dimention).
    ct : bool, default False
        If True, perform cross-temporal decoding, else time-by-time decoding.
        Automatically set to True in transfer decoding.
    
    Returns
    -------
    acc : float or ndarray
        If ct, 2d array output, else float scalar.
    """
    
    # score index
    nwins = dataset[0].shape[-1]
    if ct: # score every time window (broad sense time point) for test set
        ncwins = dataset1[0].shape[-1] if dataset1 else nwins
        sidx = np.indices((nwins, ncwins))[1]
    else: # only score current (in train) time point for test set
        ncwins = 1
        sidx = np.indices((nwins, ncwins))[0]
    
    # result container
    folds = int(np.ceil(split))
    acc = np.zeros([repeat, folds, nwins, ncwins])
    
    # train test split generator
    if split < 1: # keep one group
        spliter = _kog_gener(split)
    elif split > 1 and isinstance(split, int): # k fold cross
        spliter = _skf_gener(split)
    elif split == 1: # transfer
        spliter = _pair_gener
    
    for rpt in range(repeat):
        # Categories-balance (down sampling) and average-every-n-trials
        x, y = _dc_extract(*dataset, navg)
        x1, y1 = _dc_extract(*dataset1, navg) if dataset1 else (None, None)
        
        for fold, (x_trn, x_tet, y_trn, y_tet) in \
        enumerate(spliter(x, y, x1, y1)):
            
            for win, cwins in enumerate(sidx): # for each time window
                x_trn_win = x_trn[:, :, win]
                if norm:
                    scaler = StandardScaler()
                    x_trn_win = scaler.fit_transform(x_trn_win)
                svc = SVC(kernel='linear', tol=1e-4, probability=False)
                svc.fit(x_trn_win, y_trn)
                
                for icwin, cwin in enumerate(cwins): # score
                    x_tet_win = x_tet[:, :, cwin]
                    if norm:
                        x_tet_win = scaler.transform(x_tet_win)
                    acc[rpt, fold, win, icwin] = svc.score(x_tet_win, y_tet)
    
    acc = acc.mean((0, 1))
    if not ct:
        acc = acc[:, 0]
    
    return acc


def time_decoding(
    data,
    label,
    *args,
    navg=5,
    time_win=5,
    time_step=5,
    win_avg=True,
    split=5,
    repeat=2,
    norm=True,
    ct=False,
    smooth=2,
    verbose=True
):
    
    """
    EEG-like data decoding
    - Time by Time
    - Cross Temporal
    - Transfer    
    
    Parameters
    ----------
    data : ndarray
        Neural data in shape [n_subjects, n_trials, n_channels, n_time_points]
    label : ndarray
        Trials label in shape [n_subjects, n_trials]
    *args : optional
        With data1 and label1 passing to args, perform transfer decoding
        from (data, label) to (data1, label1).
        Require exactly same n_subjects and n_channels between data and data1.
    navg : int, default 5
        Average in every 'navg' trials.
    time_win : int, default 5
        Time granularity for decoding.
        A time_win=n refer to n_time_points decoding units.
    time_step : int, default 5
        Time step of decoding units.
        A time_step < time_win provides time-overlap decoding units.
    win_avg : boolean, default True
        If True, elements in a decoding unit(time dimension) will be averaged.
        Or be used as features and flatten into channel dimension.
    split: int or float, default 5
        Train-test strategy.
        If < 1, keep one group hold out validation, be the test proportion.
        If > 1, cross validation, be the k (fold).
        Automatically set to 1 in transfer decoding.
    repeat : int, default 2
        Repeat whole pipeline 'repeat' times, get average.
    norm : bool, default True
        If True, normalize the data (channel dimention).
    ct : bool, default False
        If True, perform cross-temporal decoding, else time-by-time decoding.
        Automatically set to True in transfer decoding.
    smooth : int or bool, default 2
        Smooth the decoding result in decoding units level, with
        window_width = smooth * 2 + 1.
        If True, set to 2.
        If 0 or False, keep origin result.
    
    Returns
    -------
    acc : ndarray
        If ct, i.e. cross-temporal decoding or transfer decoding case,
        output shape [n_subjects, n_time_windows, n_time_windows'].
        Else, time-by-time decoding case, [n_subjects, n_time_windows].
    
    Examples
    --------
    Fake some data for examples.
    Just let it be 5 subjects, 100 trials, 64 channels, 200 timepoints.
    >>> import numpy as np
    >>> dshape = 5, 100, 64, 200
    >>> lshape = 5, 100
    >>> data = np.random.randn(shape)
    >>> label = np.random.randint(lshape)
    >>> data1 = np.random.randn(shape)
    >>> label1 = np.random.randint(lshape)
    
    Time by Time Decoding
    >>> result = time_decoding(data, label, ct=False)
    
    Cross Temporal Decoding
    >>> result = time_decoding(data, label, ct=True)
    
    Transfer Decoding
    >>> forward = time_decoding(data, label, data1, label1)
    >>> bakward = time_decoding(data1, label1, data, label)
    """
    
    # check!
    cat = _dc_check(data, label)
    if len(args) == 2: # transfer signal
        data1, label1 = args
        cat1 = _dc_check(data1, label1)
        # pair data check
        ps = "doesn't match between pair data."
        if set(cat) != set(cat1):
            raise ValueError(f"Categories {ps}")
        if data.shape[0] != data1.shape[0]:
            raise ValueError(f"Number of subjects {ps}")
        if data.shape[-2] != data1.shape[-2]:
            raise ValueError(f"Number of channels {ps}")
        ct = True
        split = 1
    
    # data handle
    data = _dc_handle(data, time_win, time_step, win_avg)
    nwins = data.shape[-1]
    ncwins = nwins if ct else 1
    if args:
        data1 = _dc_handle(data1, time_win, time_step, win_avg)
        ncwins = data1.shape[-1]
    
    # result container
    nsubs = data.shape[0]
    if ct:
        acc = np.zeros((nsubs, nwins, ncwins))
    else:
        acc = np.zeros((nsubs, nwins))
    
    repeat = np.clip(repeat, 1, None) # last check repeat
    
    # decode using svc
    outloop = range(nsubs)
    if verbose:
        outloop = tqdm(outloop, ncols=100)
    for sub in outloop:
        d0 = data[sub], label[sub]
        d1 = (data1[sub], label1[sub]) if args else None
        acc[sub] = svc_pipe(d0, d1, navg, split, repeat, norm, ct)
    
    if smooth:
        if smooth == True:
            smooth = 2
        acc = smoothing2d(acc, smooth) if ct else smoothing(acc, smooth)
    
    return acc
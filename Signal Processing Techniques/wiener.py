import numpy as np
import scipy.signal as sp

def wiener(signal, filter_size=None):
    '''
    Apply a Wiener filter to a signal, filter_size must have odd dimensions
    Usage:
    --------
    >>> rng = np.random.default_rng()
    >>> rvect = rng.random((40))    #Create random vector
    >>> filtered_signal = wiener(rvect, (5))  #Filter the vector
    '''
    signal = np.asarray(signal)
    if filter_size is None:
        filter_size = [1] * signal.ndim
    filter_size = np.asarray(filter_size)
    if filter_size.shape == ():
        filter_size = np.repeat(filter_size.item(), signal.ndim)
    # Estimating the local mean
    lMean_hat = sp.correlate(signal, np.ones(filter_size), 'same') / np.prod(filter_size, axis=0)
    # Estimate the local variance
    lVar_hat = (sp.correlate(signal ** 2, np.ones(filter_size), 'same') / np.prod(filter_size, axis=0) - lMean ** 2)
    # Estimating the noise
    noise_hat = np.mean(np.ravel(lVar_hat), axis=0)
    res = (signal - lMean_hat)
    res *= (1 - noise_hat / lVar_hat)
    res += lMean
    out = np.where(lVar_hat < noise_hat, lMean_hat, res)
    return out

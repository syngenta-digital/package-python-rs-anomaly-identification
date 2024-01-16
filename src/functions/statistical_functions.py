from numba import njit
import numpy as np

@njit(fastmath=True)
def _select_sigma(x):
    normalize = 1.349
    IQR = (np.percentile(x, q=75) - np.percentile(x, q=25))/normalize
    std_dev = np.std(x)
    if IQR > 0:
        return np.minimum(std_dev, IQR)
    else:
        return std_dev
    
@njit(fastmath=True)
def bw_silverman(x):
    A = _select_sigma(x)
    n = len(x)
    return .9 * A * n ** (-0.2)

@njit(fastmath=True)
def gaussian(h, Xi, x):
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))

    
@njit(fastmath=True)
def gpke(bw, data, data_predict, var_type):
    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        Kval[:, ii] = gaussian(bw[ii], data[:, ii], data_predict[ii])
    Kval_prod=np.array([np.prod(Kval[idx,:]) for idx in range(np.shape(Kval)[0])])
    dens =Kval_prod / np.prod(bw)
    return dens.sum()
    
    
@njit(fastmath=True)
def pdf(bw, data, var_type,data_predict):

    nobs,_ = np.shape(data)
    pdf_est = [gpke(bw, data, data_predict[i, :], var_type) / nobs for i in range(np.shape(data_predict)[0])]
    return np.array(pdf_est)
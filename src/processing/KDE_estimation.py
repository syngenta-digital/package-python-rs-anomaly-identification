from numba import njit, typed, prange
import numpy as np

@njit(fastmath=True)
def _normal_reference(data:np.array) -> np.array:
        """
        A function that returns Scott's normal reference rule of thumb bandwidth parameter.

        Parameters:
        data: np.array that contains NDVI values and their corresponding day of year.

        Returns:
        Array with 2 bandwidth parameters (for NDVI and day of year).
        """
        nobs,_ = np.shape(data)
        X = np.array([np.std(data[:,idx]) for idx in range(np.shape(data)[1])])
        return 1.06 * X * nobs ** (- 1. / (4 + data.shape[1]))

@njit(fastmath=True)
def gaussian_numba(h, Xi, x):
    """
    Gaussian Kernel for continuous variables
    Parameters

    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns

    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.
    """
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))


@njit(fastmath=True)
def gpke_numba(bw:np.array, data:np.array, data_predict:np.array, var_type:str):
    """
    A function to return the non-normalized Generalized Product Kernel Estimator

    Parameters:
    bw: Bandwidth of parameters (NDVI and day of year).
    data: the data which will be used to estimate KD
    data_predict: the data which PDF will be estimated for
    var_type: string showing number of variables

    Returns:
    PDF array
    """
    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        Kval[:, ii] = gaussian_numba(bw[ii], data[:, ii], data_predict[ii])
    Kval_prod=np.array([np.prod(Kval[idx,:]) for idx in range(np.shape(Kval)[0])])
    dens =Kval_prod / np.prod(bw)
    return dens.sum()

@njit(fastmath=True)
def get_pdf_numba(bw, data, var_type,data_predict):
    """
    A function to calculate PDF values

    Parameters:
    bw: Bandwidth of parameters (NDVI and day of year).
    data: the data which will be used to estimate KD
    data_predict: the data which PDF will be estimated for
    var_type: string showing number of variables

    Returns:
    PDF array
    """
    nobs,_ = np.shape(data)
    pdf_est = [gpke_numba(bw, data, data_predict[i, :], var_type) / nobs for i in range(np.shape(data_predict)[0])]
    return np.array(pdf_est)

@njit(parallel=True)
def get_finaloutput_numba(doy=np.array([]), ndvi=np.array([[[]]]),ndim=np.array([])) -> np.array:
    """
    A function to calculate PDF values for the entire field

    Parameters:
    doy: day of year.
    ndvi: ndvi values
    ndim: dimenstion of the training dataset

    Returns:
    PDF array for all pixels in the entire field.
    """
    output=np.zeros((50,365,ndim[1],ndim[2]),np.float32)
    ndvi_range = np.linspace(0, 10000, num=50)
    for y in prange(ndim[1]):
        for x in prange(ndim[2]):
            ndvi_px=ndvi[:,y,x]
            ndvi_dates=np.zeros((len(doy),2))
            ndvi_dates[:,0]=doy
            ndvi_dates[:,1]=ndvi_px
            bw=_normal_reference(ndvi_dates)
            for i in range(1,366):
                days=np.array([i]*50)
                data=np.stack((days,ndvi_range), axis=1)
                pdf=get_pdf_numba(bw, ndvi_dates,'cc',data)
                output[:,i-1,y,x]=pdf
    return output
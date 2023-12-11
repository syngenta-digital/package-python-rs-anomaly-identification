from numba import njit, typed, prange
import numpy as np

@njit(fastmath=True)
def _normal_reference(data):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where ``n`` is the number of observations and ``q`` is the number of
        variables.
        """
        nobs,_ = np.shape(data)
        X = np.array([np.std(data[:,idx]) for idx in range(np.shape(data)[1])])
        return 1.06 * X * nobs ** (- 1. / (4 + data.shape[1]))

@njit(fastmath=True)
def gaussian_numba(h, Xi, x):
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))


@njit(fastmath=True)
def gpke_numba(bw, data, data_predict, var_type):
    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        Kval[:, ii] = gaussian_numba(bw[ii], data[:, ii], data_predict[ii])
    Kval_prod=np.array([np.prod(Kval[idx,:]) for idx in range(np.shape(Kval)[0])])
    dens =Kval_prod / np.prod(bw)
    return dens.sum()

@njit(fastmath=True)
def get_pdf_numba(bw, data, var_type,data_predict):
    nobs,_ = np.shape(data)
    pdf_est = [gpke_numba(bw, data, data_predict[i, :], var_type) / nobs for i in range(np.shape(data_predict)[0])]
    #pdf_est = np.squeeze(pdf_est)
    return np.array(pdf_est)

@njit(parallel=True)
def get_finaloutput_numba(doy=np.array([]), ndvi=np.array([[[]]]),ndim=np.array([])):
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
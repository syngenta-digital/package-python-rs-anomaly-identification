from numba import njit
import numpy as np
import itertools

@njit(fastmath=True)
def _select_sigma(x:list)->float:
    normalize = 1.349
    IQR = (np.percentile(x, q=75) - np.percentile(x, q=25))/normalize
    std_dev = np.std(x)
    if IQR > 0:
        return np.minimum(std_dev, IQR)
    else:
        return std_dev
    
@njit(fastmath=True)
def bw_silverman(x:list)->float:
    A = _select_sigma(x)
    n = len(x)
    return .9 * A * n ** (-0.2)

@njit(fastmath=True)
def gaussian(h, Xi, x)->float:
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))

    
@njit(fastmath=True)
def gpke(bw:np.array, data:list, data_predict:list, var_type:str)->list:
    Kval = np.empty(data.shape)
    for ii, _ in enumerate(var_type):
        Kval[:, ii] = gaussian(bw[ii], data[:, ii], data_predict[ii])
    Kval_prod=np.array([np.prod(Kval[idx,:]) for idx in range(np.shape(Kval)[0])])
    dens =Kval_prod / np.prod(bw)
    return dens.sum()
    
    
@njit(fastmath=True)
def pdf(bw:list, data:list, var_type:str,data_predict:list)->list:

    nobs,_ = np.shape(data)
    pdf_est = [gpke(bw, data, data_predict[i, :], var_type) / nobs for i in range(np.shape(data_predict)[0])]
    return np.array(pdf_est)


def get_prob(test_data:list, prob_arr:list)->list:
    arr=np.zeros(test_data.shape)
    for x, y in itertools.product(range(test_data.shape[1]), range(test_data.shape[2])):
        for idx, val in enumerate(test_data[:,x,y]):
            arr[idx,x,y]=prob_arr[idx,val,x,y]    
    return arr


def ndvi_index_raster(test_data:list, ndvi_range:list)->list:
    list1=[]
    for val in test_data:
        list2=[np.abs(d-val) for d in ndvi_range]
        list1.append(list2.index(min(list2)))
    return list1


def unique_cumsum(arr:list)->list:
    arr[np.isnan(arr)]=0
    b=np.unique(np.sort(arr))[::-1]
    c=np.cumsum(b)
    vals_dict={x:y for x,y in zip(b, c)}
    e=np.array([vals_dict[x] for x in arr])
    return e
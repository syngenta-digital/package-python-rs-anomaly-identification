from numba import njit, prange
import numpy as np
from src.functions.statistical_functions import  bw_silverman, pdf

class PDF():
    
    def __init__(self, doy, ndvi, ndim):
        self.doy=doy
        self.ndvi=ndvi
        self.ndim=ndim
        
    @staticmethod
    @njit(parallel=True)
    def calculation(doy=np.array([]), ndvi=np.array([[[]]]),ndim=np.array([])):
            output=np.zeros((50,365,ndim[1],ndim[2]),np.float32)
            ndvi_range = np.linspace(0, 10000, num=50)
            for y in prange(ndim[1]):
                for x in prange(ndim[2]):
                    ndvi_px=ndvi[:,y,x]
                    ndvi_dates=np.zeros((len(doy),2))
                    ndvi_dates[:,0]=doy
                    ndvi_dates[:,1]=ndvi_px
                    bw0=bw_silverman(doy)
                    bw1=bw_silverman(ndvi_px)
                    for i in range(1,366):
                        days=np.array([i]*50)
                        data=np.stack((days,ndvi_range), axis=1)
                        pdf_result=pdf(np.array([bw0,bw1]), ndvi_dates,'cc',data)
                        output[:,i-1,y,x]=pdf_result
            return output
        
    def PDF_estimation(self):
        doy=self.doy
        ndvi=self.ndvi
        ndim=self.ndim
        return self.calculation(doy,ndvi,ndim)
    
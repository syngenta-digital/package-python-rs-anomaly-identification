from src.processing.image_processing import get_prob, ndvi_index_raster, unique_cumsum
import numpy as np
import pandas as pd

class ExtremeAnoMap:
    """
    A class to show anomalous area in the field, the calculation is carried out at pixel level.
    each pixel is assigned two values (propability of being anomalous and the NDVI difference). 
    Propabilities and NDVI difference are obtained by comparing NDVI value of a pixel to the NDVI value of the refrence
    of the same pixel.
    
    Parameters:
    
    doy: Day of year (of the NDVI image).
    NDVI: NDVI array of all images.

    Returns:

    RFD: Propability maps.
    delta: NDVI difference maps.
    """
    def __init__(self, file,doy:np.array, NDVI:np.array)-> None:
        self.file=file
        self.doy=doy
        self.NDVI=NDVI

    def rfd(self):
        f=np.apply_along_axis(lambda x: x/np.sum(x), 0, self.file)
        temp1=f.reshape(18250, f.shape[2], f.shape[3])
        temp2=np.apply_along_axis(lambda x: x/np.sum(x), 0, temp1)
        temp2=np.apply_along_axis(lambda x: unique_cumsum(x), 0, temp2)
        temp3=temp2.reshape((50,365,f.shape[2],f.shape[3]))
        ndvi_range=np.linspace(0, 10000, num=50)
        temp4=np.array([temp3[:,int(idx),:,:] for idx in self.doy-1])
        temp5=np.apply_along_axis(lambda x: ndvi_index_raster(x, ndvi_range), 0, self.NDVI)
        rfd=get_prob(temp5, temp4)
        return rfd

    def delta(self):
        f=np.apply_along_axis(lambda x: x/np.sum(x), 0, self.file)
        temp1=np.apply_along_axis(lambda x: np.argmax(x), 0, f)
        ndvi_range=np.linspace(0, 10000, num=50)
        ndvi_dict={key:val for key, val in enumerate(ndvi_range)}
        temp_ndvi=np.apply_along_axis(lambda x: pd.Series(x).map(ndvi_dict).values, 0, temp1)
        deltas=np.array([self.NDVI[idx,:,:] - temp_ndvi[int(day),:,:] for idx, day in enumerate(self.doy)])
        return deltas
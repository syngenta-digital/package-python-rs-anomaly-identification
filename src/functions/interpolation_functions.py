import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.signal import savgol_filter

def interpolate(dates:list, ndvi:list[float], interpolation_step=1, time_series_similarity=False, window_length=50, polyorder=3)->tuple:
    """
    A function to apply linear interpolation to vegetative index value, interpolated series are next smoothed by applying Savitsky Golay filter.
    User can specify interpolation steps in days.
    The output is interpolated vegetative index values along with interpolated time of the image which can be expressed in day of year
    or dates depending of user selection.

    Parameters:

    dates: dates array.
    ndvi: vegetative index value
    interpolation_step: frequency of interpolation in days.
    time_series_similarity: it defines the format of the output for interpolated dates. 
    False outputs interpolated dates while True gives interpolated day of year.
    window_length: number of samples to be considered in smoothing window. Large number of samples yields smoother series.
    polyorder:The order of the polynomial used to fit the samples. polyorder must be less than window_length.

    Returns:

    final_doy: interpolated dates or day of year.
    final_ndvi: interpolated vegetative index.
        
    """
    interp_doy=[]
    interp_ndvi=[]
    dates=pd.DatetimeIndex(dates)
    years=np.unique(dates.year)
    for year in years:
        year_dates=[date for date in dates if str(year) in str(date)]
        year_index=[i for i,date in enumerate(dates) if str(year) in str(date)]
        doys=np.array([pd.to_datetime(d).dayofyear for d in year_dates])
        minval=np.min(doys)
        maxval=np.max(doys)
        doy_range=np.arange(start=minval, stop=maxval+1, step=interpolation_step, dtype='float')
        seasonal_ndvi=ndvi[np.min(year_index):np.max(year_index)+1]
        ndvi_interp=np.interp(doy_range, doys, seasonal_ndvi)
        ndvi_smth=savgol_filter(ndvi_interp, window_length = window_length, polyorder = polyorder)
        interp_ndvi.append(ndvi_smth)
        if time_series_similarity:
            date_range=[datetime(year, 1, 1) + timedelta(d - 1) for d in doy_range]
            interp_doy.append(date_range)
        else:
            interp_doy.append(doy_range)
            
    final_doy=[v for item in interp_doy for v in item]
    final_ndvi=[v for item in interp_ndvi for v in item]
    return np.array(final_doy), np.array(final_ndvi)

def dimension_calculation(dates:np.array, vi:np.array, interpolation_step=1)->tuple:
    """
    A function to estimate the final interpolated data.

    Parameters:

    dates: dates array.
    ndvi: vegetative index value
    interpolation_step: frequency of interpolation in days.

    Returns:

    dimension of the interpolated data.
        
    """
    doy_list=[]
    dates=pd.DatetimeIndex(dates)
    years=np.unique(dates.year)
    for year in years:
        year_dates=[date for date in dates if str(year) in str(date)]
        doy=np.array([pd.to_datetime(d).dayofyear for d in year_dates])
        minval=np.min(doy)
        maxval=np.max(doy)
        doy_range=np.arange(start=minval, stop=maxval+1, step=interpolation_step)
        doy_list.append(doy_range.shape[0])
    return (np.sum(doy_list),vi.shape[1], vi.shape[2])

def field_interpolation(dates:list, vi:list[float], interpolation_step=1, output_dates=False)->tuple:
    """
    A function to apply linear interpolation to all pixels in the field, interpolated series are next smoothed by applying Savitsky Golay filter.
    User can specify interpolation steps in days.
    The output is interpolated vegetative index values along with interpolated time of the image which can be expressed in day of year
    or dates depending of user selection.

    Parameters:

    dates: dates array.
    ndvi: vegetative index value
    interpolation_step: frequency of interpolation in days.
    time_series_similarity: it defines the format of the output for interpolated dates. 
    False outputs interpolated dates while True gives interpolated day of year.
    window_length: number of samples to be considered in smoothing window. Large number of samples yields smoother series.
    polyorder:The order of the polynomial used to fit the samples. polyorder must be less than window_length.

    Returns:

    final_doy: interpolated day of year.
    final_ndvi: interpolated vegetative index.
        
    """
    shape=dimension_calculation(dates,vi, interpolation_step=interpolation_step)
    vi_interp=np.zeros(shape,np.float32)
    for y in range(shape[1]):
        for x in range(shape[2]):
            ndvi_px=vi[:,y,x]
            fin_doy,fin_vi=interpolate(dates,ndvi_px, interpolation_step=interpolation_step, time_series_similarity=output_dates)
            vi_interp[:,y,x]=fin_vi
    return fin_doy,vi_interp
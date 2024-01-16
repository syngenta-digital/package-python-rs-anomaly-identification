import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.signal import savgol_filter

def interpolate(dates, ndvi, interpolation_step=1, output_dates=False, window_length=50, polyorder=3):
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
        if output_dates:
            date_range=[datetime(year, 1, 1) + timedelta(d - 1) for d in doy_range]
            interp_doy.append(date_range)
        else:
            interp_doy.append(doy_range)
            
    final_doy=[v for item in interp_doy for v in item]
    final_ndvi=[v for item in interp_ndvi for v in item]
    return np.array(final_doy), np.array(final_ndvi)

def dimension_calculation(dates, vi, interpolation_step=1):
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

def field_interpolation(dates, vi, interpolation_step=1, output_dates=False):
    shape=dimension_calculation(dates,vi, interpolation_step=interpolation_step)
    vi_interp=np.zeros(shape,np.float32)
    for y in range(shape[1]):
        for x in range(shape[2]):
            ndvi_px=vi[:,y,x]
            fin_doy,fin_vi=interpolate(dates,ndvi_px, interpolation_step=interpolation_step, output_dates=output_dates)
            vi_interp[:,y,x]=fin_vi
    return fin_doy,vi_interp
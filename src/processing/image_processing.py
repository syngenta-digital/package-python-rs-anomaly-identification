import xarray as xr
import numpy as np
import pandas as pd

def data_split(files, test_year='2023'):
    if isinstance(test_year, (float,int)):
        test_year=str(test_year)
    training=[file for file in files if test_year not in file.split('/')[-1]]
    testing=[file for file in files if test_year in file.split('/')[-1]]
    DStr= xr.open_mfdataset(training)
    DSts= xr.open_mfdataset(testing)
    spatial_dim=(DStr['x'].shape[0], DStr['y'].shape[0])
    return DStr, DSts, spatial_dim


def valid_images(DS,alignment_dict,vi='NDVI', testing_season=False,start_date=None,end_date=None):
    if vi=='NDVI':
        DS['NDVI']=(DS['B08']-DS['B04'])/(DS['B08']+DS['B04'])*10000
    null_count={idx:np.sum(np.isnan(DS[vi].values[idx,:,:]))\
                /(DS[vi].values.shape[1]*DS[vi].values.shape[2]) for idx in range(len(DS['time'].values))}
    min_val=min(null_count.values())
    dates_to_drop=[DS['time'].values[key] for key,val in null_count.items() if val > min_val]
    if testing_season:
        dates_to_drop=dates_to_drop
    else:
        seasons_to_drop=[date for date in DS['time'].values if 
                     all(year not in str(date) for year in alignment_dict['deltas'].keys())]
        dates_to_drop=dates_to_drop+seasons_to_drop
    if start_date is not None and end_date is not None:
        years=set([str(date).split('-')[0] for date in DS['time'].values])
        window_of_extraction=[]
        for year in years:
            for date in DS['time'].values:
                if year in str(date):
                    if str(date)<str(year)+'-'+str(start_date) or str(date)>str(year)+'-'+str(end_date):
                        window_of_extraction.append(date)
        dates_to_drop=dates_to_drop+window_of_extraction
        DS=DS.drop_sel(time=dates_to_drop)
    else:
        DS=DS.drop_sel(time=dates_to_drop)
    return DS

def date_data(field_data,dictionary):
    dates_to_drop=[]
    ndoy=[]
    odoy=[]
    dates=field_data['time'].values
    for year in dictionary['deltas'].keys():
        for date in dates:
            if year in str(date) :
                new_doy=pd.to_datetime(date).dayofyear - np.round(dictionary['deltas'][year])
                if new_doy>0:                                                
                    ndoy.append(new_doy)
                    odoy.append(pd.to_datetime(date).dayofyear)
                else:
                    dates_to_drop.append(date)
            else:
                continue
    field_data=field_data.drop_sel(time=dates_to_drop)
    return field_data, np.array(odoy), np.array(ndoy)

def get_prob(test_data, prob_arr):
    arr=np.zeros(test_data.shape)
    for x in range(test_data.shape[1]):
        for y in range(test_data.shape[2]):
            for idx, val in enumerate(test_data[:,x,y]):
                arr[idx,x,y]=prob_arr[idx,val,x,y]      
    return arr

def ndvi_index_raster(test_data, ndvi_range):
    list1=[]
    for val in test_data:
        list2=[np.abs(d-val) for d in ndvi_range]
        list1.append(list2.index(min(list2)))
    return list1


def unique_cumsum(arr):
    arr[np.isnan(arr)]=0
    b=np.unique(np.sort(arr))[::-1]
    c=np.cumsum(b)
    vals_dict={x:y for x,y in zip(b, c)}
    e=np.array([vals_dict[x] for x in arr])
    return e
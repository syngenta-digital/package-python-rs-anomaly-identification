import xarray as xr
import numpy as np
import pandas as pd
import itertools

def data_split(files:list, test_year='2023') -> [xr.core.dataset.Dataset,tuple]:
     """
     A function to split files of field under consideration into training and testing.
    
     Parameters:
    
     doy: Day of year (of the NDVI image).
     NDVI: NDVI array of all images.

     Returns:

     DStr: dataset of vegtitative index (training).
     DSts: dataset of vegtitative index (testing).
     spatial_dim: dimenstion of testing dataset
     """
     if isinstance(test_year, (float,int)):
         test_year=str(test_year)
     training=[file for file in files if test_year not in file.split('/')[-1]]
     testing=[file for file in files if test_year in file.split('/')[-1]]
     DStr= xr.open_mfdataset(training)
     DSts= xr.open_mfdataset(testing)
     spatial_dim=(DStr['x'].shape[0], DStr['y'].shape[0])
     return DStr, DSts, spatial_dim


def valid_images(DS:xr.core.dataset.Dataset,alignment_dict:dict,vi='NDVI', testing_season=False,start_date=None,end_date=None) -> xarray.core.dataset.Dataset:
     """
     A function to remove unwated images.
    
     Parameters:
    
     DS: dataset of vegtitative index.
     alignment_dict: dictionary containing available years for the field.
     testing_season: year which will used as testing

     Returns:

     DS: dataset of vegtitative index.
     """
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
        for year, date in itertools.product(years, DS['time'].values):
            if year in str(date):
                if str(date)<str(year)+'-'+str(start_date) or str(date)>str(year)+'-'+str(end_date):
                    window_of_extraction.append(date)
        dates_to_drop=dates_to_drop+window_of_extraction
        DS=DS.drop_sel(time=dates_to_drop)
     else:
        DS=DS.drop_sel(time=dates_to_drop)
     return DS

def date_data(field_data:xr.core.dataset.Dataset,dictionary:dict) -> [xr.core.dataset.Dataset,np.array,np.array]:
     """
     A function calculate shifted (aligned) day of year.
    
     Parameters:
    
     field_data: dataset of vegtitative index.
     dictionary: dictionary containing available years for the field with amount needed to shift the every season.

     Returns:

     field_data: dataset of vegtitative index.
     odoy: Original day of year.
     ndoy: Aligned day of year.
     """
     dates_to_drop=[]
     ndoy=[]
     odoy=[]
     dates=field_data['time'].values
     for year , date in itertools.product(dictionary['deltas'].keys(),dates):
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

def get_prob(test_data:np.array, prob_arr:np.array) -> np.array:
    """
    A function to get pixel propapbilities for every image of the testing season.
    
     Parameters:
    
     test_data: NDVI value closest to the testing data from NDVI search space.
     prop_arr: Propability values

     Returns:

     arr: Propability values for every image in the testing season.
     """
    arr=np.zeros(test_data.shape)
    for x, y in itertools.product(range(test_data.shape[1]), range(test_data.shape[2])):
        for idx, val in enumerate(test_data[:,x,y]):
            arr[idx,x,y]=prob_arr[idx,val,x,y]    
    return arr

def ndvi_index_raster(test_data:xr.core.dataset.Dataset, ndvi_range:np.array) -> list:
    """
    A function to get closest NDVI .
    
     Parameters:
    
     test_data: dataset of vegtitative index.
     NDVI_range: 50 NDVI values ranging from 0 - 10000.

     Returns:

     list1: index of the closest NDVI values in the testing data to the NDVI search values.
     """
    list1=[]
    for val in test_data:
        list2=[np.abs(d-val) for d in ndvi_range]
        list1.append(list2.index(min(list2)))
    return list1


def unique_cumsum(arr:np.array) -> np.array:
    """
    Calculating cumulative sum for all unique values in propability file .
    
     Parameters:
    
     arr: Propability file.

     Returns:

     e: array containing cumulative sum for all values.
     """
    arr[np.isnan(arr)]=0
    b=np.unique(np.sort(arr))[::-1]
    c=np.cumsum(b)
    vals_dict={x:y for x,y in zip(b, c)}
    e=np.array([vals_dict[x] for x in arr])
    return e
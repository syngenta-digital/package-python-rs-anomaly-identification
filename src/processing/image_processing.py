import numpy as np
import pandas as pd
from datetime import timedelta

class DataProcessing():
    def __init__(self, dates:list, ndvi, alignment_dict:dict,testing_season=None, to_keep=None)->None:
        """
        Class constructor which accepts dates and values of vegetative images
    
        Parameters:
    
        dates: Dates of images.
        NDVI: NDVI array of all images.
        alignment_dict: a dictionary contains the required days to align planting dates

        """
    
        self.dates=dates
        self.ndvi=ndvi
        self.alignment_dict=alignment_dict
        self.testing_season=testing_season
        self.to_remove=to_keep
    
    def data_split(self)->tuple:
        """
        A method for splitting training and testing year
    
        Returns:
    
        trdates: Dates of the training seasons.
        trvi: NDVI array of training images.
        tsdates: Dates of the testing season
        tsvi: NDVI array of testing images
        
        """
        training_dates=[]
        training_ndvi=[]
        testing_dates=[]
        testing_ndvi=[]
        dates=pd.DatetimeIndex(self.dates)
        years=np.unique(dates.year)
        if self.to_remove:
            touse_seasons=[year for year in years if year in self.to_keep]
        else:
            touse_seasons=years
        for year in touse_seasons:
            year_dates=[date for date in dates if str(year) in str(date)]
            year_index=[i for i,date in enumerate(dates) if str(year) in str(date)]
            minval=np.min(year_index)
            maxval=np.max(year_index)
            seasonal_ndvi=self.ndvi[minval:maxval+1,:,:]
            if year==self.testing_season:
                testing_dates.append(year_dates)
                testing_ndvi.append(seasonal_ndvi)
            else:
                training_dates.append(year_dates)
                training_ndvi.append(seasonal_ndvi)
        trdates=[v for item in training_dates for v in item]
        trvi=[v for item in training_ndvi for v in item]
        tsdates=[v for item in testing_dates for v in item]
        tsvi=[v for item in testing_ndvi for v in item]
        return trdates,np.array(trvi),tsdates,np.array(tsvi)
    
    @staticmethod
    def threshold_calculation(dates:list,vi:list)->tuple:
        """
        A method to calculate an acceptance threshold of null pixels and remove images with higher pixels than the accpeted threshold.

        Parameters:

        dates: Dates of the images.
        vi: vi values of the images.
    
        Returns:
    
        dates_to_keep: Dates after removing noisy dates.
        vi: vi values after removing noisy images.
        
        """
        null_count={idx:np.sum(np.isnan(vi[idx,:,:]))\
                /(vi.shape[1]*vi.shape[2]) for idx in np.arange(vi.shape[0])}
    
        min_val=min(null_count.values())
        dates_to_keep=[dates[key] for key,val in null_count.items() if val <= min_val]
        ndvi_to_keep=[vi[key,:,:] for key,val in null_count.items() if val <= min_val]
        return np.array(dates_to_keep), np.array(ndvi_to_keep)
    
    
    def valid_images(self)->tuple:
        """
        A method to apply threshold calculation on training and testing datasets.

        Returns:
    
        trdates_fin: Dates of the training seasons after removing noisy dates
        trvi_fin: training vi values after removing noisy images.
        tsdates_fin: Dates of the testing season after removing noisy dates
        tsvi_fin: testing vi values after removing noisy images.
        
        """
        trdates,trvi,tsdates,tsvi=self.data_split()
        trdates_fin,trvi_fin=self.threshold_calculation(trdates,trvi)
        tsdates_fin, tsvi_fin=self.threshold_calculation(tsdates,tsvi)
        return trdates_fin, trvi_fin, tsdates_fin, tsvi_fin
    
    @staticmethod
    def date_alignment(dates:list, vi:list, alignment_dict:dict)->tuple:
        """
        A method to calculate an acceptance threshold of null pixels and remove images with higher pixels than the accpeted threshold.

        Parameters:

        dates: Dates of the images.
        vi: vi values of the images.
        alignment_dict: a dictionary contains the required days to align planting dates.
    
        Returns:
    
        old_dates: original dates.
        new_dates: Dates after aligning planting dates.
        vi_final: vi values.
        
        """
        old_dates_list=[]
        new_dates_list=[]
        vi_list=[]
        years=list(alignment_dict['deltas'].keys())
        for year in years:
            year_index={i :date for i,date in enumerate(dates) if str(year) in str(date)}
            original_dates={k:v for k,v in year_index.items() \
                          if pd.to_datetime(v).dayofyear - np.round(alignment_dict['deltas'][year])>0}
            aligned_dates={k:(pd.to_datetime(v) - timedelta(np.round(alignment_dict['deltas'][year]))).to_numpy()\
                     for k,v in year_index.items() if pd.to_datetime(v).dayofyear - np.round(alignment_dict['deltas'][year])>0}
            old_dates_list.append(original_dates.values())
            new_dates_list.append(aligned_dates.values())
            vi_list.append([vi[idx,:,:] for idx in aligned_dates.keys()])
        
        old_dates=[i for item in old_dates_list for i in item]
        new_dates=[i for item in new_dates_list for i in item]
        vi_final=[i for item in vi_list for i in item]
        return np.array(old_dates), np.array(new_dates),np.array(vi_final)
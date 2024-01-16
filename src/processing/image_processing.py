import numpy as np
import pandas as pd
from datetime import timedelta

class DataProcessing():
    def __init__(self, dates, ndvi, alignment_dict,testing_season=None, to_remove=None):
        self.dates=dates
        self.ndvi=ndvi
        self.alignment_dict=alignment_dict
        self.testing_season=testing_season
        self.to_remove=to_remove
    
    def data_split(self):
        training_dates=[]
        training_ndvi=[]
        testing_dates=[]
        testing_ndvi=[]
        dates=pd.DatetimeIndex(self.dates)
        years=np.unique(dates.year)
        if self.to_remove:
            touse_seasons=[year for year in years if year not in self.to_remove]
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
    def threshold_calculation(dates,vi):
        null_count={idx:np.sum(np.isnan(vi[idx,:,:]))\
                /(vi.shape[1]*vi.shape[2]) for idx in np.arange(vi.shape[0])}
    
        min_val=min(null_count.values())
        dates_to_keep=[dates[key] for key,val in null_count.items() if val <= min_val]
        ndvi_to_keep=[vi[key,:,:] for key,val in null_count.items() if val <= min_val]
        return np.array(dates_to_keep), np.array(ndvi_to_keep)
    
    
    def valid_images(self):
        trdates,trvi,tsdates,tsvi=self.data_split()
        trdates_fin,trvi_fin=self.threshold_calculation(trdates,trvi)
        tsdates_fin, tsvi_fin=self.threshold_calculation(tsdates,tsvi)
        return trdates_fin, trvi_fin, tsdates_fin, tsvi_fin
    
    
    def date_alignment(self):
        old_dates_list=[]
        new_dates_list=[]
        vi_list=[]
        years=list(self.alignment_dict['deltas'].keys())
        dates,vi,_,_=self.valid_images()
        for year in years:
            year_index={i :date for i,date in enumerate(dates) if str(year) in str(date)}
            original_dates={k:v for k,v in year_index.items() \
                          if pd.to_datetime(v).dayofyear - np.round(self.alignment_dict['deltas'][year])>0}
            aligned_dates={k:(pd.to_datetime(v) - timedelta(np.round(self.alignment_dict['deltas'][year]))).to_numpy()\
                     for k,v in year_index.items() if pd.to_datetime(v).dayofyear - np.round(self.alignment_dict['deltas'][year])>0}
            old_dates_list.append(original_dates.values())
            new_dates_list.append(aligned_dates.values())
            vi_list.append([vi[idx,:,:] for idx in aligned_dates.keys()])
        
        old_dates=[i for item in old_dates_list for i in item]
        new_dates=[i for item in new_dates_list for i in item]
        vi_final=[i for item in vi_list for i in item]
        return np.array(old_dates), np.array(new_dates),np.array(vi_final)
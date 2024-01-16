import pandas as pd
import numpy as np
from dtaidistance import dtw

class DTWSimilarity:
    def __init__(self,dates,ndvi,reference_filepath=None, testing_season=None, field_name=None):
        self.dates=dates
        self.ndvi=ndvi
        self.reference_filepath=reference_filepath
        self.testing_season=testing_season
        self.field_name=field_name
    def calculate_similarity(self):
        dates=pd.DatetimeIndex(self.dates)
        years=np.unique(dates.year)
        df_ref=pd.read_csv(self.reference_filepath, index_col='Unnamed: 0')
        df_metric=pd.DataFrame(columns=['Similarity'],index=[year for year in years if year!=self.testing_season])
        arr2=df_ref['Reference'].values
        for year in years:
            year_index=[i for i,date in enumerate(dates) if str(year) in str(date)]
            arr1=self.ndvi[np.min(year_index):np.max(year_index)+1]
            df_metric.loc[year,'Similarity']=dtw.distance_fast(arr1,arr2)
        df_metric['Field']=self.field_name
        return df_metric
    def similar_seasons(self, threshold):
        df_metric=self.calculate_similarity()
        selected_seasons=df_metric[df_metric.loc[:,'Similarity']<threshold].index.tolist()
        return selected_seasons
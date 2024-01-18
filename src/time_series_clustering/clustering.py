import pandas as pd
import numpy as np
from dtaidistance import dtw

class DTWSimilarity:
    def __init__(self,dates:list,ndvi:[float],reference_filepath=None, testing_season=None,field_name=None)->None:
        
        """
        Class constructor which accepts dates array, vegetative index array and path to the reference file.

        Parameters:

        dates: dates array.
        ndvi: ndvi array.
        reference_filepath: path to the reference file.

        """
        self.dates=dates
        self.ndvi=ndvi
        self.reference_filepath=reference_filepath
        self.testing_season=testing_season
        self.field_name=field_name

    def calculate_similarity(self)->pd.DataFrame:
        """
        A method to compute time series similarity.

        Returns:
    
        df_metric: A dynamic time wrapping similarity value of time series of the training seasons.
        
        """
        dates=pd.DatetimeIndex(self.dates)
        years=np.unique(dates.year)
        if not self.testing_season or self.testing_season not in years:
            raise ValueError('Testing season should be a year available in the supplied in the data and expressed as integer')
        df_ref=pd.read_csv(self.reference_filepath, index_col='Unnamed: 0')
        df_metric=pd.DataFrame(columns=['Similarity'],index=[year for year in years if year !=self.testing_season])
        arr2=df_ref['Reference'].values
        for year in df_metric.index:
            year_index=[i for i,date in enumerate(dates) if str(year) in str(date)]
            arr1=self.ndvi[np.min(year_index):np.max(year_index)+1]
            df_metric.loc[year,'Similarity']=dtw.distance_fast(arr1,arr2)
        df_metric['Field']=self.field_name
        return df_metric
    
    def similar_seasons(self, threshold:float)->list[int]:
        """
        A method to obtain similar seasons in the training dataset along with the testing season. 

        Parameters:

        threshold: similarity value of 1.7 seems to be a good number. Smaller number means more conservative measurment of similarity

        Returns:
    
        selected_seasons: a list of seasons to keep.
        
        """
        df_metric=self.calculate_similarity()
        selected_seasons=df_metric[df_metric.loc[:,'Similarity']<threshold].index.tolist()
        return selected_seasons + [self.testing_season]
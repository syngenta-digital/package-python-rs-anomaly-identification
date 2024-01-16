import pandas as pd
from src.time_series_clustering.clustering import DTWSimilarity
from src.functions.interpolation_functions import interpolate

if __name__=='__main__':
    print('start')
    file='../../data/Parquet_files/id_field_213.parquet'
    reference_file='../../data/csv_files/NDVI_reference_files/corn_reference.csv'
    df=pd.read_parquet(file)
    df_ref=pd.read_csv(reference_file, index_col='Unnamed: 0')
    final_doy,final_ndvi=interpolate(df['Date'].values, df['NDVI'].values, output_dates=True)
    tss=t=DTWSimilarity(final_doy,final_ndvi,reference_filepath=reference_file, testing_season=2023, field_name=213)
    a=tss.calculate_similarity()
    b=tss.similar_seasons(threshold=1.7)
    print('Similarity metric')
    print(a)
    print(f'Seasons to keep: {b}')
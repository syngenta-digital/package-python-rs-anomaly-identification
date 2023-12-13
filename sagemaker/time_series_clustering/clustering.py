from src.time_series_clustering.clustering import time_series_processing,dtw_metric

if __name__=='__main__':
    print('start')
    file='../../data/Parquet_files/id_field_213.parquet'
    df, field=time_series_processing(file, vi='NDVI',min_date='03-01',max_date='10-31',years=[2017,2018,2019,2020,2021,2022])
    df_metric=dtw_metric(df,field,ref_file_path='../../data/csv_files/NDVI_reference_files/corn_reference.csv', ref_name='Reference')
    print(df_metric)
    print('end')

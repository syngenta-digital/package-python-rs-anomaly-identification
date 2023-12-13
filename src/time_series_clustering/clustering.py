import pandas as pd
from scipy.signal import savgol_filter
from dtaidistance import dtw

def interpolate(df, vi='NDVI', min_date='01-01', max_date='30-11',window_length=50, polyorder=3):
    cols=[vi]
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    year=df['year'].unique()[0]
    daily_date = pd.DataFrame(pd.date_range(start = min_date+'-'+str(year), end = max_date+'-'+str(year)), columns = ['Date'])
    df_daily = daily_date.merge(df, how = 'left', on = 'Date')
    if pd.isna(df_daily.iloc[0,1]):
        df_daily.iloc[0,1]=0
    if pd.isna(df_daily.iloc[-1,1]):
        df_daily.iloc[-1,1]=0
    df_daily[vi]=df_daily[vi].astype('float')
    for col in cols:
        df_daily[col + '_lnterp'] = df_daily[col].interpolate(method='slinear')
        df_daily[col + '_lnterp'] = savgol_filter(df_daily[col + '_lnterp'], window_length = window_length, polyorder = polyorder)
    return df_daily

def time_series_processing(file, vi='NDVI',min_date='01-01',max_date='11-30',years=[],reference=False, reference_year=0):
    field_name=file.split('/')[-1].split('.')[0]
    data=pd.read_parquet(file)
    data['year']=pd.to_datetime(data['Date']).dt.year
    if reference:
        data=data[data['year'].isin([reference_year])]
        df_interp=data.groupby('year').apply(interpolate,vi=vi,min_date=min_date,max_date=max_date,
                                             window_length=50, polyorder=3).reset_index(drop=True)
        df_interp['year']=pd.to_datetime(df_interp['Date']).dt.year
        df_interp['DOY']=pd.to_datetime(df_interp['Date']).dt.dayofyear
        df_interp=df_interp[~(df_interp['Date'].astype(str).str.contains('02-29'))]
    else:
        data=data[data['year'].isin(years)]
        df_interp=data.groupby('year').apply(interpolate,vi=vi,min_date=min_date,max_date=max_date,
                                             window_length=50, polyorder=3).reset_index(drop=True)
        df_interp['year']=pd.to_datetime(df_interp['Date']).dt.year
        df_interp['DOY']=pd.to_datetime(df_interp['Date']).dt.dayofyear
        df_interp=df_interp[~(df_interp['Date'].astype(str).str.contains('02-29'))]
    return df_interp, field_name

def dtw_metric(df,field_name, ref_file_path=None, ref_name='NDVI_lnterp'):
    df=df.sort_values(by='Date')
    df=df[~(df['Date'].astype(str).str.contains('02-29'))]
    years=df['year'].unique().tolist()
    df_metric=pd.DataFrame(columns=['2017','2018','2019','2020','2021','2022'])
    df_ref=pd.read_csv(ref_file_path)
    arr2=df_ref[ref_name].values
    for i in years:
        arr1=df[df['year']==i]['NDVI_lnterp'].values
        df_metric.loc[0,str(i)]=dtw.distance_fast(arr1,arr2)
    df_metric['Field']=field_name
    return df_metric
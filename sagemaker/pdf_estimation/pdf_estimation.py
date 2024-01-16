from datetime import datetime
from datetime import datetime
import pickle
from glob import glob
from src.processing.KDE_estimation import PDF
from src.processing.image_processing import DataProcessing
from src.functions.interpolation_functions import field_interpolation
import xarray as xr


if __name__=='__main__':
    
    files=glob(f'../../data/NC_files/Field_213/*field_213.*.nc')

    with open(f'../../data/Pickled_files/norm_213.pickle','rb') as f:
        data_alignment=pickle.load(f)

    data= xr.open_mfdataset(files)
    data['NDVI']=(data['B08']-data['B04'])/(data['B08']+data['B04'])*10000
    dp=DataProcessing(data['time'].values,data['NDVI'].values, data_alignment,testing_season=2023, to_remove=[])
    trdates,trvi,tsdates,tsvi=dp.data_split()
    trdates_fin,trvi_fin,tsdates_fin,tsvi_fin=dp.valid_images()
    original_dates, aligned_dates, ndvi=dp.date_alignment()
    doy,ndvi_interp=field_interpolation(original_dates, ndvi)
    start=datetime.now()
    pdfc=PDF(doy[:100], ndvi_interp[:100,:,:], ndvi_interp[:100,:,:].shape)
    data_dict=pdfc.PDF_estimation()
    with open(f'../../data/Pickled_files/Field_213_original_doy_daily_interpolation_silverman_2017_2023_new_code.pkl', mode='wb') as file:
        pickle.dump(data_dict, file)
    end=datetime.now()
    print(end-start)
    print(data_dict)

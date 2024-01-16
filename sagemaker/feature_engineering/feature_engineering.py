from src.processing.image_processing import DataProcessing
from glob import glob 
import pickle
from collections import defaultdict
import xarray as xr

def main(field_num):

    field_dict=defaultdict(dict)

    files=glob(f'../../data/NC_files/Field_213/*field_{field_num}.*.nc')

    with open(f'../../data/Pickled_files/norm_{field_num}.pickle','rb') as f:
        data_alignment=pickle.load(f)

    data= xr.open_mfdataset(files)
    data['NDVI']=(data['B08']-data['B04'])/(data['B08']+data['B04'])*10000
    dp=DataProcessing(data['time'].values,data['NDVI'].values, data_alignment,testing_season=2023, to_remove=[])
    trdates,trvi,tsdates,tsvi=dp.data_split()
    trdates_fin,trvi_fin,tsdates_fin,tsvi_fin=dp.valid_images()
    original_dates, aligned_dates, ndvi=dp.date_alignment()
    field_dict[field_num]['original_doy']=original_dates
    field_dict[field_num]['new_doy']=aligned_dates
    field_dict[field_num]['NDVI']=ndvi
    with open(f'../../output/Pickled_files/Field_data/Field_{field_num}_data.pickle','wb') as f:
        pickle.dump(field_dict,f)
    print(f'Original DOY:{original_dates[:5]}')
    print(f'Shifted DOY:{aligned_dates[:5]}')
    print(data_alignment['deltas'])

if __name__=='__main__':
    main(213)
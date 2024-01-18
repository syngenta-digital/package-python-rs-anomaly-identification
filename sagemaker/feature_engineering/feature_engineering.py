from src.processing.image_processing import data_split, valid_images,date_data
from glob import glob 
import pickle
from collections import defaultdict

def main(field_num):

    field_dict=defaultdict(dict)

    files=glob(f'../../data/NC_files/Field_213/*field_{field_num}.*.nc')

    with open(f'../../data/Pickled_files/norm_{field_num}.pickle','rb') as f:
        data_alignment=pickle.load(f)

    field_tr,field_ts, spatial_dim=data_split(files)
    field_tr=valid_images(field_tr,data_alignment, vi='NDVI', testing_season=False, start_date='03-01', end_date='10-31')
    field_ts=valid_images(field_ts, data_alignment,vi='NDVI',testing_season=True, start_date='03-01', end_date='10-31')
    field_tr,original_doy_tr, new_doy_tr=date_data(field_tr, data_alignment)
    field_ts,original_doy_ts, new_doy_ts=date_data(field_ts, data_alignment)
    field_dict[field_num]['original_doy_tr']=original_doy_tr
    field_dict[field_num]['new_doy_tr']=new_doy_tr
    field_dict[field_num]['original_doy_ts']=original_doy_ts
    field_dict[field_num]['new_doy_ts']=new_doy_ts
    field_dict[field_num]['NDVI_tr']=field_tr['NDVI'].values
    field_dict[field_num]['NDVI_ts']=field_ts['NDVI'].values
    with open(f'../../output/Pickled_files/Field_data/Field_{field_num}_data.pickle','wb') as f:
        pickle.dump(field_dict,f)
    print(f'Original DOY:{original_doy_tr[:5]}')
    print(f'Shifted DOY:{new_doy_ts[:5]}')

if __name__=='__main__':
    main(213)
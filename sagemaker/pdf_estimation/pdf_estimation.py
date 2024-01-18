import numpy as np
from numba import njit, prange
from datetime import datetime
import pickle
from src.processing.KDE_estimation import get_finaloutput_numba


if __name__=='__main__':
    field_num=213
    with open(f'../../output/Pickled_files/Field_data/Field_{field_num}_data.pickle', 'rb') as f:
        field_dict=pickle.load(f)
    ndim=field_dict[field_num]['NDVI_tr'].shape
    start=datetime.now()
    data_dict=get_finaloutput_numba(field_dict[field_num]['original_doy_tr'], field_dict[field_num]['NDVI_tr'], ndim)
    end=datetime.now()
    print(end-start)
    with open(f'../../output/Pickled_files/PDFs/Field_{field_num}_original_doy.pkl', mode='wb') as file:
        pickle.dump(data_dict, file)
import numpy as np
import sys
sys.path.append("..")

import src.utils as utils
import src.config as config
from src.normalization_builder import NormalizationBuilder


if __name__ == "__main__":
    data = utils.load_data(config.NETCDF4_INPUT_FOLDER)
    FIELD_ID = "field_126"
    selected_field = [x for x in list(data.keys()) if FIELD_ID in x]

    # Compute the (lon, lat) of the specified FIELD_ID
    (lon, lat) = (np.mean(list(data[selected_field[0]]['x'].data)), np.mean(list(data[selected_field[0]]['y'].data)))

    # Compute NDVI rasters and dates of all seasons
    dates, all_rasters = [], []
    for i in range(0, len(selected_field)):
        dates += utils.compute_dates(data[selected_field[i]])
        all_rasters += utils.create_rasters(data[selected_field[i]])

    # Compute NDVI time_series of all seasons
    ndvi_time_series = [np.nanmedian(x) for x in all_rasters]

    # Normalize the seasons
    normalizer = NormalizationBuilder()
    planting_date_input_df = normalizer.perform_normalization(ndvi_time_series, dates, (lon, lat))
    print(planting_date_input_df)

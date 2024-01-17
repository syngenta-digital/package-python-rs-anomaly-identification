# General imports
import sys
import os

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
import numpy as np
import matplotlib.pyplot as plt

# Class specific imports
from src import normalization_builder
from scripts import user_config as user_config, utils

if __name__ == "__main__":
    # ---------------------------------------------------
    # Load NETCDF4 files for the specified FIELD_ID
    # ---------------------------------------------------
    data = utils.load_data(user_config.NETCDF4_INPUT_FOLDER)
    # FIELD_ID = "field_148"
    FIELD_ID = "field_126"
    selected_field = [x for x in list(data.keys()) if FIELD_ID in x]

    # ---------------------------------------------------
    # Compute the (lon, lat) of the specified FIELD_ID
    # ---------------------------------------------------
    (lon, lat) = (np.mean(list(data[selected_field[0]]['x'].data)), np.mean(list(data[selected_field[0]]['y'].data)))

    # ---------------------------------------------------
    # Compute NDVI rasters and dates for all seasons
    # ---------------------------------------------------
    dates, all_rasters = [], []
    for i in range(0, len(selected_field)):
        dates += utils.compute_dates(data[selected_field[i]])
        all_rasters += utils.create_rasters(data[selected_field[i]])

    # -------------------------------------------------------
    # Compute NDVI time_series for all seasons
    # -------------------------------------------------------
    ndvi_time_series = [np.nanmedian(x) for x in all_rasters]

    # -------------------------------------------------------------------------
    # Normalize the seasons: this is main interface with Normalization Class!
    # -------------------------------------------------------------------------
    print("-> Estimating planting dates:")
    normalizer = normalization_builder.NormalizationBuilder()
    shift_dict = normalizer.normalize_time_series(ndvi_time_series, dates, (lon, lat))
    print(f"\n-> Planting_dates:")
    print(f"{shift_dict['planting_dates']}\n")
    print(f"-> Deltas:")
    print(f"{shift_dict['deltas']}\n")

    # -------------------------------------------------------------------------
    # Using the normalization computed
    # -------------------------------------------------------------------------
    df = normalizer.create_planting_date_dataframe(ndvi_time_series, dates, (lon, lat))
    years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
    plt.subplots(1, 2, figsize=(15, 5))
    ax1 = plt.subplot(121)
    for season in years:
        x = [i for i in range(1, 366)]
        y = list(df[df['cropzone'] == season]['NDVI'])
        ax1.plot(x, y)

    ax2 = plt.subplot(122)
    for season in years:
        x = [(i - shift_dict['deltas'][season]) for i in range(1, 366)]
        y = list(df[df['cropzone'] == season]['NDVI'])
        ax2.plot(x, y)
    plt.show()

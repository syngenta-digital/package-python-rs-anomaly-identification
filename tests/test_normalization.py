# General imports
import sys
import os

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
import numpy as np

# Class specific imports
from src import normalization_builder, config
from scripts import user_config as user_config, utils


def test_normalization():
    """
    Tests if deltas computed for pairs os season match with predefined values.
    This test is performed for multiple fields
    :return: (None)
    """

    for field in config.TESTING_DELTAS:
        # Recovers predefined deltas for the field specified by "field"
        deltas = config.TESTING_DELTAS[field]

        # loads the data for "field"
        data = utils.load_data(user_config.NETCDF4_INPUT_FOLDER)
        selected_field = [x for x in list(data.keys()) if field in x]

        # Computes longitude and latitude for "field"
        (lon, lat) = (np.mean(list(data[selected_field[0]]['x'].data)),
                      np.mean(list(data[selected_field[0]]['y'].data)))

        # Extracts ndvi_time_series for all seasons, as well as the correspondent dates of collection
        dates, all_rasters = [], []
        for i in range(0, len(selected_field)):
            dates += utils.compute_dates(data[selected_field[i]])
            all_rasters += utils.create_rasters(data[selected_field[i]])
        ndvi_time_series = [np.nanmedian(x) for x in all_rasters]

        # 1. Create and object of the class normalization
        # 2. Computes de delta for all seasons for "field"
        normalizer = normalization_builder.NormalizationBuilder()
        shift_dict = normalizer.normalize_time_series(ndvi_time_series, dates, (lon, lat))

        # TEST the computed delta for each season against the predefined delta for the same season
        for season in shift_dict['deltas']:
            assert (int(np.round(shift_dict['deltas'][season])) == deltas[season])


if __name__ == "__main__":
    test_normalization()
